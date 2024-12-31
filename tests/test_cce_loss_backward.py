# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import pytest
import torch

from cut_cross_entropy import linear_cross_entropy
from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.utils import softcapping

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


def _grads(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    softcap: float | None,
    shift: bool,
    reduction: str,
    fp32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_e, orig_c = e, c
    e.grad = c.grad = None

    N, T = targets.size()
    if shift:
        e = e[:, :-1]
        targets = targets[:, 1:]
        T = T - 1

    e = e.flatten(0, -2)
    targets = targets.flatten()

    if fp32:
        e = e.float()
        c = c.float()

    logits = e @ c.T
    if softcap is not None:
        logits = softcapping(logits, softcap)

    loss = torch.nn.functional.cross_entropy(
        logits.float(), targets, ignore_index=IGNORE_INDEX, reduction=reduction
    )

    if reduction == "sum":
        loss = loss / (targets != IGNORE_INDEX).count_nonzero()

    loss.mean().backward()

    assert orig_e.grad is not None
    assert orig_c.grad is not None

    return orig_e.grad.detach().clone(), orig_c.grad.detach().clone()


@skip_no_cuda
@pytest.mark.parametrize("impl", ["cce", "torch_compile"])
@pytest.mark.parametrize("dtype,error_tol", [(torch.float16, 1e-3), (torch.bfloat16, 1e-2)])
@pytest.mark.parametrize("softcap", [None, 20.0])
@pytest.mark.parametrize("shift", [False, True])
@pytest.mark.parametrize("invalids", [False, True])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("shape", [(256, 512, 128), (252, 507, 128), (252, 507, 123)])
def test_loss_backward(
    impl: str,
    dtype: torch.dtype,
    error_tol: float,
    softcap: float | None,
    shift: bool,
    invalids: bool,
    reduction: str,
    shape: tuple[int, int, int],
):
    torch.set_float32_matmul_precision("highest")
    torch._dynamo.config.cache_size_limit = 256
    torch.cuda.manual_seed(0)

    if dtype == torch.bfloat16 and not torch.cuda.is_available():
        pytest.skip(reason="BF16 not avaliable")

    N, V, D = shape
    e = torch.randn((N, D), device="cuda", dtype=dtype, requires_grad=False) / (D**0.5)
    c = torch.randn((V, D), device="cuda", dtype=dtype, requires_grad=False)

    c[0 : min(N, V) // 2] = e[0 : min(N, V) // 2]

    targets = torch.randint(0, V, size=(N,), device="cuda")

    if invalids:
        inds = torch.randperm(len(targets), device="cuda")[0 : int(0.2 * len(targets))]
        targets[inds] = IGNORE_INDEX

    e = e.view(4, -1, D)
    targets = targets.view(e.size()[0:-1])

    e.requires_grad_(True)
    c.requires_grad_(True)

    gt = _grads(e, c, targets, softcap, shift, reduction, fp32=True)

    ref = _grads(e, c, targets, softcap, shift, reduction)

    e.grad = c.grad = None
    loss = linear_cross_entropy(
        e, c, targets, softcap=softcap, shift=shift, reduction=reduction, impl=impl
    )
    if reduction == "sum":
        loss = loss / (targets != IGNORE_INDEX).count_nonzero()
    loss.mean().backward()
    assert e.grad is not None
    assert c.grad is not None

    expected_error = tuple((vgt - vref).abs() for vgt, vref in zip(gt, ref))
    cce_error = tuple((vgt - vcce).abs() for vgt, vcce in zip(gt, (e.grad, c.grad)))

    for i in range(len(expected_error)):
        assert (
            cce_error[i] <= (expected_error[i] + error_tol)
        ).all(), f"{(cce_error[i] - expected_error[i]).relu().max()=}"
