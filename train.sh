export HF_HOME=~/
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=2,3,4,5
export TORCH_USE_CUDA_DSA=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=TRACE
export NCCL_IB_HCA=irdma1:1
export NCCL_IB_GID_INDEX=0
export NCCL_SOCKET_IFNAME=^lo,docker,virbr,vmnet,vboxnet,eth0
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_SHM_DISABLE=1
export MASTER_ADDR=10.10.0.102
export MASTER_PORT=29500
export NCCL_DEBUG_FILE="/tmp/nccl_debug_rank_%r.log"

deepspeed --num_nodes=1 --num_gpus=4 finetune.py --deepspeed ds_config.json