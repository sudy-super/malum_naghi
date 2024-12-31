from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import transformers
import torch
from trl import SFTTrainer
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.utils import shuffle
import pandas as pd
import wandb
import os
from typing import Any, Sequence, cast
import torch.distributed as dist

#if dist.get_rank() == 0:
wandb.init(project="naghi_sft", name="2e-5_qwen2.5-7b_eps-1e-15", entity="sudy_super")

torch.manual_seed(42)


device_map = {
    'model.embed_tokens': 0,
    'model.rotary_emb': 0,
    'model.layers.0': 0,
    'model.layers.1': 0,
    'model.layers.2': 0,
    'model.layers.3': 0,
    'model.layers.4': 0,
    'model.layers.5': 0,
    'model.layers.6': 0,

    'model.layers.7': 1,
    'model.layers.8': 1,
    'model.layers.9': 1,
    'model.layers.10': 1,
    'model.layers.11': 1,
    'model.layers.12': 1,
    'model.layers.13': 1,

    'model.layers.14': 1,
    'model.layers.15': 1,
    'model.layers.16': 2,
    'model.layers.17': 2,
    'model.layers.18': 2,
    'model.layers.19': 2,
    'model.layers.20': 2,

    'model.layers.21': 2,
    'model.layers.22': 2,
    'model.layers.23': 2,
    'model.layers.24': 2,
    'model.layers.25': 3,
    'model.layers.26': 3,
    'model.layers.27': 3,
    'model.norm': 3,
    'lm_head': 3
}

model_name = "Qwen/Qwen2.5-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
model.gradient_checkpointing_enable()
#print(model)
"""
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 131072,
    dtype="bfloat16",
    load_in_8bit=False,
    load_in_4bit=False,
)
"""

dataset = load_dataset("Manual-Dataset-Creation-Project/Naghi-SFT")

dataset = dataset['train'].shuffle(seed=42)

def tokenize(batch):
    # batch["prompt"] は文字列のリストになっている
    tokenized = tokenizer(
        batch["prompt"],
        truncation=True,
        max_length=32768,
        padding=False,
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }


def generate_prompt(batch):
    results = []
    for conversations_of_one_sample in batch["conversations"]:
        result = "<|im_start|>system\nあなたはManual Dataset Creation Projectによって作られた凪 (Naghi) です。あなたは頼りになるアシスタントです。<|im_end|>\n<|im_start|>user\n"
        count = 1
        for turn in conversations_of_one_sample:
            system_flag = False
            if turn["role"] == "system":
                turn_value = turn["content"]
                result = f"<|im_start|>system\n{turn_value}<|im_end|>\n<|im_start|>user\n"
                system_flag = True
            if turn["role"] == "user":
                turn_value = turn["content"]
                result += f"{turn_value}<|im_end|>\n<|im_start|>assistant\n"
            elif turn["role"] == "assistant":
                turn_value = turn["content"]
                if system_flag:
                    if (len(conversations_of_one_sample) / 2) == (count + 0.5):
                        result += f"{turn_value}<|im_end|>"
                    else:
                        result += f"{turn_value}<|im_end|>\n<|im_start|>user\n"
                else:
                    if (len(conversations_of_one_sample) / 2) == count:
                        result += f"{turn_value}<|im_end|>"
                    else:
                        result += f"{turn_value}<|im_end|>\n<|im_start|>user\n"
            count += 1
        results.append(result)

    return {"prompt": results}


val_data = dataset.select(range(1000))
train_data = dataset.select(range(1000, len(dataset)))

train_data = train_data.map(
        generate_prompt,
        batched=True,
        num_proc=8,
        desc=f"Generating inputs for training",
        load_from_cache_file=True
    ).filter(lambda x: x["prompt"] != '', num_proc=8)
val_data = val_data.map(
        generate_prompt,
        batched=True,
        num_proc=8,
        desc=f"Generating inputs for validation",
        load_from_cache_file=True
    ).filter(lambda x: x["prompt"] != '', num_proc=8)

train_data = train_data.map(
        tokenize,
        batched=True,
        num_proc=8,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing for training",
        load_from_cache_file=True
    )
val_data = val_data.map(
        tokenize,
        batched=True,
        num_proc=8,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing for validation",
        load_from_cache_file=True
    )


class DataCollatorOnlyAssistantResponses:
    """
    <|im_start|>assistant ~ <|im_end|> の区間のみを学習対象（=ラベル）として残し、
    それ以外（system / user の部分 + その他特殊トークン）はラベルを -100 で埋めるデータコレーター。
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        # 必要な特殊トークンのIDを取得
        self.start_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.end_token_id   = tokenizer.convert_tokens_to_ids("<|im_end|>")
        
        # "assistant" のトークン ID
        # これが実際に tokenizer でどう分割されるか確認が必要
        self.assistant_token_id = 77091
        
        # 改行のトークンID
        self.newline_token_id = 198

    def __call__(self, features):
        """
        features は下記形式のリスト:
        [
          {
            "input_ids": [...],
            "attention_mask": [...]
          },
          ...
        ]
        """
        # トークナイザでパディング
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt"
        )
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        # labels を複製して作成 (初期値は input_ids と同じ)
        labels = input_ids.clone()

        # それぞれのサンプルに対して「assistantの区間」以外を -100 に置き換える
        for idx in range(len(features)):
            token_ids = input_ids[idx]

            in_assistant = False
            length = len(token_ids)
            
            i = 0
            while i < length:
                tid = token_ids[i]

                # <|im_start|> に遭遇したら、次トークンが "assistant" の場合のみ in_assistant = True
                if tid == self.start_token_id:
                    # i+1 が範囲内 & 次が assistant_token_id のとき
                    if (i+1 < length) and (token_ids[i+1] == self.assistant_token_id):
                        # (例) <|im_start|>assistant なので in_assistant 開始
                        in_assistant = True
                        # この特殊トークン自体は学習させない
                        labels[idx, i]   = -100  # <|im_start|>
                        labels[idx, i+1] = -100  # assistant
                        
                        # もしその直後の改行 (ID=198) も無視したいなら
                        if (i+2 < length) and (token_ids[i+2] == self.newline_token_id):
                            labels[idx, i+2] = -100
                            # 改行も「assistant開始前にある特殊トークン」とみなし i+2 だけ -100 にする
                            # in_assistant = True は変わらない
                        i += 2  # <|im_start|>, assistant を一気に飛ばす
                        continue
                    else:
                        # "assistant" 以外なら普通に -100
                        labels[idx, i] = -100
                        in_assistant = False

                # <|im_end|> に遭遇したらアシスタント区間終了
                elif tid == self.end_token_id:
                    in_assistant = False
                    # labels[idx, i] = -100  # 終了トークンも -100

                else:
                    # アシスタント区間外は -100
                    if not in_assistant:
                        labels[idx, i] = -100

                i += 1

        batch["labels"] = labels
        return batch

model.requires_grad_(True)

# model = cast(transformers.PreTrainedModel, model)
from cut_cross_entropy.transformers import cce_patch
model = cce_patch(model, impl="cce", reduction="mean", gradient_accumulation_steps=32)

args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=32,
    learning_rate=2e-5,
    adam_epsilon=1e-15,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    adam_beta2=0.95,
    weight_decay=0.0,
    logging_steps=1,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=199,
    save_steps=597,
    output_dir="output",
    report_to="wandb",
    save_total_limit=3,
    push_to_hub=False,
    seed=42,
    bf16=True,  # bf16を有効化
    bf16_full_eval=True,
    #deepspeed="ds_config.json",  # DeepSpeed設定ファイルの指定
    gradient_checkpointing=True,
    optim="adafactor",
    dataloader_pin_memory=True,
    dataloader_num_workers=8,
    local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    torch_compile=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=DataCollatorOnlyAssistantResponses(tokenizer),
)

trainer.train()

trainer.save_model("output_model")
model.save_pretrained("output_model_aux", safe_serialization=True)
