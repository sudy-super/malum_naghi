from datasets import load_dataset
import json
import random

# Hugging Faceデータセットをロードし、辞書形式で保持
def get_dataset_as_dict(dataset_name, split):
    # データセットをロード
    print(f"Loading {dataset_name} dataset...")
    if split in ["train", "validation", "test", "v1.0_cleaned", "20240807beginwith_commands", "20240806filtered"]:
        dataset = load_dataset(dataset_name, split=split, token="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")
    else:
        dataset = load_dataset(dataset_name, split, split="train", token="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")
    
    dataset_dict = []
    for record in dataset:
        # record がすでに辞書形式の場合はそのまま append
        if isinstance(record, dict):
            dataset_dict.append(record)
        # そうでなければ {"value": record} としてラップ
        else:
            dataset_dict.append({"value": record})

    return dataset_dict

def format_1(data_point, dataset_name):
    records = []
    for record in data_point:
        new_record = {"conversations": [], "origin": dataset_name}
        new_record["conversations"] = record["conversations"]
        records.append(new_record)
    
    return records

def format_2(data_point, dataset_name):
    records = []
    for record in data_point:
        new_record = {"conversations": [], "origin": dataset_name}
        new_record["conversations"] = [
            {"role": "user", "content": record["instruction"] + "\n\n" + record["input"]},
            {"role": "assistant", "content": record["output"]},
        ]
        records.append(new_record)
    
    return records

def format_3(data_point, dataset_name):
    records = []
    for record in data_point:
        new_record = {"conversations": [], "origin": dataset_name}
        new_record["conversations"] = record["messages"]
        records.append(new_record)
    
    return records

def format_4(data_point, dataset_name):
    instruction_list = [
        "この文を短くまとめてください。",
        "この情報を手短に説明してください。",
        "上記の内容について簡潔に教えてください。",
        "これの要点だけ教えてもらえますか？",
        "このコンテキストを簡単に説明していただけると助かります。",
        "これを要約していただけますか？",
        "コンパクトにお願いします。",
        "この情報についてサクッと教えて",
        "上記の文章を短くしてください。",
    ]
    
    records = []
    for record in data_point:
        instruction = random.choice(instruction_list)
        new_record = {"conversations": [], "origin": dataset_name}
        new_record["conversations"] = [
            {"role": "user", "content": record["input"] + "\n\n" + instruction},
            {"role": "assistant", "content": record["generated"]},
        ]
        records.append(new_record)
    
    return records

def format_5(data_point, dataset_name):
    records = []
    count = 0
    for record in data_point:
        new_record = {"conversations": [], "origin": dataset_name}
        if count % 4 == 0:
            new_record["conversations"] = [
                {"role": "user", "content": record["input"] + "\n" + record["instruction"]},
                {"role": "assistant", "content": record["response"]},
            ]
        elif count % 4 == 1:
            new_record["conversations"] = [
                {"role": "user", "content": record["input"] + "\n\n" + record["instruction"]},
                {"role": "assistant", "content": record["response"]},
            ]
        elif count % 4 == 2:
            new_record["conversations"] = [
                {"role": "user", "content": record["instruction"] + "\n" + record["input"]},
                {"role": "assistant", "content": record["response"]},
            ]
        else:
            new_record["conversations"] = [
                {"role": "user", "content": record["instruction"] + "\n\n" + record["input"]},
                {"role": "assistant", "content": record["response"]},
            ]
        records.append(new_record)
        count += 1
    
    return records

def format_6(data_point, dataset_name):
    records = []
    for record in data_point:
        new_record = {"conversations": [], "origin": dataset_name}
        new_record["conversations"] = [
            {"role": "user", "content": record["instruction"]},
            {"role": "assistant", "content": record["output"]},
        ]
        records.append(new_record)
    
    return records

def format_7(data_point, dataset_name):
    records = []
    count = 0
    for record in data_point:
        new_record = {"conversations": [], "origin": dataset_name}
        if count % 4 == 0:
            new_record["conversations"] = [
                {"role": "user", "content": record["text"] + "\n" + record["instruction"]},
                {"role": "assistant", "content": record["output"]},
            ]
        elif count % 4 == 1:
            new_record["conversations"] = [
                {"role": "user", "content": record["text"] + "\n\n" + record["instruction"]},
                {"role": "assistant", "content": record["output"]},
            ]
        elif count % 4 == 2:
            new_record["conversations"] = [
                {"role": "user", "content": record["instruction"] + "\n" + record["text"]},
                {"role": "assistant", "content": record["output"]},
            ]
        else:
            new_record["conversations"] = [
                {"role": "user", "content": record["instruction"] + "\n\n" + record["text"]},
                {"role": "assistant", "content": record["output"]},
            ]
        records.append(new_record)
        count += 1
    
    return records

def format_8(data_point, dataset_name):
    records = []
    for record in data_point:
        new_record = {"conversations": [], "origin": dataset_name}
        new_record["conversations"] = record["prompt"]
        new_record["conversations"].append({"role": "assistant", "content": record["chosen"]})
        records.append(new_record)
    
    return records

def format_9(data_point, dataset_name):
    records = []
    for record in data_point:
        new_record = {"conversations": [], "origin": dataset_name}
        new_record["conversations"] = [
            {"role": "user", "content": record["instruction"]},
            {"role": "assistant", "content": record["response"]},
        ]
        records.append(new_record)

    return records

def format_10(data_point, dataset_name):
    records = []
    for record in data_point:
        new_record = {"conversations": [], "origin": dataset_name}
        new_record["conversations"] = [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["chosen"]},
        ]
        records.append(new_record)
    
    return records

def format_11(data_point, dataset_name):
    records = []
    for record in data_point:
        if record["language"] == "Japanese":
            new_record = {"conversations": [], "origin": dataset_name}
            new_record["conversations"] = record["messages"]
            records.append(new_record)
    
    return records

def format_12(data_point, dataset_name):
    records = []
    count = 0
    for record in data_point:
        if count % 2 == 0:
            new_record = {"conversations": [], "origin": dataset_name}
            new_record["conversations"] = record["messages"][:11]
            records.append(new_record)
        count += 1
    
    return records

def format_13(data_point, dataset_name):
    records = []
    for record in data_point:
        new_record = {"conversations": [], "origin": dataset_name}
        new_conversations = []
        for c in record["translated_conversations"]:
            if c["from"] == "system":
                new_conversations.append({"role": "system", "content": c["value"]})
            elif c["from"] == "human":
                new_conversations.append({"role": "user", "content": c["value"]})
            else:
                new_conversations.append({"role": "assistant", "content": c["value"]})
        records.append(new_record)
    
    return records

def format_14(data_point, dataset_name):
    records = []
    for record in data_point:
        new_record = {"conversations": [], "origin": dataset_name}
        new_record["conversations"] = [
            {"role": "user", "content": record["question"]},
            {"role": "assistant", "content": record["answer"]},
        ]
        records.append(new_record)
    
    return records

def format_15(data_point, dataset_name):
    records = []
    for record in data_point:
        new_record = {"conversations": [], "origin": dataset_name}
        new_record["conversations"] = [
            {"role": "user", "content": record["question_ja"]},
            {"role": "assistant", "content": record["generated_solution_ja"]},
        ]
        records.append(new_record)
    
    return records

def format_16(data_point, dataset_name):
    records = []
    for record in data_point:
        new_record = {"conversations": [], "origin": dataset_name}
        new_conv = []
        for c in record["conversations"]:
            if c["from"] == "system":
                new_conv.append({"role": "system", "content": c["value"]})
            elif c["from"] == "human":
                new_conv.append({"role": "user", "content": c["value"]})
            elif c["from"] == "gpt":
                new_conv.append({"role": "assistant", "content": c["value"]})
            else:
                raise ValueError("Invalid role.")
        new_record["conversations"] = new_conv
        records.append(new_record)
    
    return records

def format_17(data_point, dataset_name):
    records = []
    for record in data_point:
        new_record = {"conversations": [], "origin": dataset_name}
        new_record["conversations"] = [
            {"role": "syatem", "content": record["system"]},
            {"role": "user", "content": record["instruction"]},
            {"role": "assistant", "content": record["output"]},
        ]
        records.append(new_record)
    
    return records

def format_baobabu(data_point, dataset_name):
    records = []
    count = 0
    for record in data_point:
        new_record = {"conversations": [], "origin": dataset_name}
        if count % 4 == 0:
            new_record["conversations"] = [
                {"role": "user", "content": record["input"] + "\n" + record["question"]},
                {"role": "assistant", "content": record["answer"]},
            ]
        elif count % 4 == 1:
            new_record["conversations"] = [
                {"role": "user", "content": record["input"] + "\n\n" + record["question"]},
                {"role": "assistant", "content": record["answer"]},
            ]
        elif count % 4 == 2:
            new_record["conversations"] = [
                {"role": "user", "content": record["question"] + "\n" + record["input"]},
                {"role": "assistant", "content": record["answer"]},
            ]
        else:
            new_record["conversations"] = [
                {"role": "user", "content": record["question"] + "\n\n" + record["input"]},
                {"role": "assistant", "content": record["answer"]},
            ]
        records.append(new_record)
        count += 1
    
    return records

# 使用例
if __name__ == "__main__":
    dataset_name = "imdb"
    split = "train"
    dataset_list = [
        ("Manual-Dataset-Creation-Project/Malum-230", "train", 1),
        ("sudy-super/CoTangent", "train", 2),
        ("llm-jp/oasst1-21k-ja", "train", 16),
        ("llm-jp/oasst2-33k-ja", "train", 1),
        ("Aratako/Rosebleu-1on1-Dialogues-RP", "v2", 3),
        # "baobab-trees/wikipedia-human-retrieval-ja",
        ("aixsatoshi/Longcontext-aozora-summary", "train", 4),
        ("aixsatoshi/Longcontext-aozora-instruction", "train", 5),
        ("kunishou/amenokaku-code-instruct", "train", 6),
        ("HachiML/Evol-hh-rlhf-gen3-1k", "train", 6),
        # "minnade/chat-daily",
        ("HachiML/Hachi-Alpaca", "v1.0_cleaned", 2),
        ("Kendamarron/jimba-wiki-instruction-calm3", "train", 7),
        ("weblab-GENIAC/aya-ja-evol-instruct-calm3-dpo-masked", "train", 8),
        ("weblab-GENIAC/Open-Platypus-Japanese-masked", "train", 9),
        ("weblab-GENIAC/aya-ja-nemotron-dpo-masked", "train", 10),
        ("Aratako/Synthetic-JP-EN-Coding-Dataset-801k", "train", 11),
        ("Aratako/Synthetic-JP-Conversations-Magpie-Nemotron-4-10k", "train", 3),
        ("Aratako/Synthetic-JP-10-Turns-Roleplay-Dialogues-Nemotron-4-1k", "train", 12),
        ("Aratako/LimaRP-augmented-ja-karakuri", "train", 13),
        ("llm-jp/magpie-sft-v1.0", "train", 1),
        ("GENIAC-Team-Ozaki/WikiHowNFQA-ja_cleaned", "train", 14),
        ("GENIAC-Team-Ozaki/OpenMathInstruct-1-1.8m-ja_10k", "train", 15),
        ("Aratako/WIP-Dataset-For-Self-Taught-Evaluators", "train", 3),
        ("Aratako/gemma-2-27b-evol-instruct-88k-sft", "train", 3),
        ("Aratako/gemma-2-27b-evol-instruct-88k-sft", "test", 3),
        ("kanhatakeyama/wizardlm8x22b-logical-math-coding-sft-ja", "train", 3),
        ("kanhatakeyama/wizardlm8x22b-logical-math-coding-sft_additional-ja", "train", 3),
        ("kanhatakeyama/ramdom-to-fixed-multiturn-Calm3", "20240806filtered", 3),
        ("kanhatakeyama/logicaltext-wizardlm8x22b-api", "train", 17),
        ("kanhatakeyama/multiturn-Calm3-manual", "20240807beginwith_commands", 3),
        # ("kanhatakeyama/0804calm3-logical-multiturn-pretrain", "train", 3),
    ]

    total_records = []
    for dataset_name, split, format_num in dataset_list:
        dataset_dict = get_dataset_as_dict(dataset_name, split)
        # format_numで指定された番号の関数を呼び出し、データを整形
        print(f"Formatting {dataset_name} dataset...")
        formatted_data = globals()[f"format_{format_num}"](dataset_dict, dataset_name)
        print(len(formatted_data))
        
        total_records.extend(formatted_data)
    
    with open("baobabu_train.json", "r", encoding="utf-8") as f:
        print("Loading baobabu_train dataset...")
        baobabu_train = json.load(f)
        print("Formatting baobabu_train dataset...")
        baobabu_train = format_baobabu(baobabu_train, "baobab-trees/wikipedia-human-retrieval-ja")
    with open("baobabu_val.json", "r", encoding="utf-8") as f:
        print("Loading baobabu_val dataset...")
        baobabu_val = json.load(f)
        print("Formatting baobabu_val dataset...")
        baobabu_val = format_baobabu(baobabu_val, "baobab-trees/wikipedia-human-retrieval-ja")
    with open("null_instruct.json", "r", encoding="utf-8") as f:
        print("Loading null_instruct dataset...")
        null_instruct = json.load(f)
        print("Formatting null_instruct dataset...")
        null_instruct = format_1(null_instruct, "neody/null-instruct-ja")
    with open("minnade.json", "r", encoding="utf-8") as f:
        print("Loading minnade dataset...")
        minnade = json.load(f)
        print("Formatting minnade dataset...")
        minnade = format_1(null_instruct, "minnade/chat-daily")
    
    total_records.extend(baobabu_train)
    total_records.extend(baobabu_val)
    total_records.extend(null_instruct)
    total_records.extend(minnade)

    print(len(total_records))

    for record in total_records:
        for c in record["conversations"]:
            try:
                if c["role"] == "user":
                    pass
                elif c["role"] == "assistant":
                    pass
                elif c["role"] == "system":
                    pass
            except:
                raise ValueError(f"Invalid role at {record['origin']} dataset.") 
    print("All records are valid.")

    from huggingface_hub import login
    from datasets import Dataset

    login(token="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")

    dataset = Dataset.from_list(total_records)

    dataset.push_to_hub(
        repo_id="Manual-Dataset-Creation-Project/Naghi-SFT",
        private=True,
        token="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",
    )