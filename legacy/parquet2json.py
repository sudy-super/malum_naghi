import pyarrow.parquet as pq
import json

def parquet_to_jsonl(parquet_file_path, json_file_path):
    # Parquetファイルを読み込む
    table = pq.read_table(parquet_file_path)

    # Parquetの各列をリストに変換
    user_column = table['text'].to_pylist()
    user_column2 = table['instruction'].to_pylist()
    model_column = table['output'].to_pylist()

    records = []

    # JSONLファイルに書き込む
    with open(json_file_path, 'w', encoding="utf-8") as json_file:
        count = 0
        for i in range(len(user_column)):
            count += 1
            if count % 1 == 0:
                record = {"conversations": [{"from": "human", "value": user_column[i] + "\n\n" + user_column2[i]}, {"from": "gpt", "value": model_column[i]}]}
                records.append(record)
        
        print(len(records))
        json_file.write(json.dumps(records, ensure_ascii=False, indent=4) + "\n")

# 使用例
parquet_file_path = "C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\jimba_calm3.parquet"  # 入力のParquetファイルパス
json_file_path = "C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\jimba_calm3.json"  # 出力のJSONファイルパス

parquet_to_jsonl(parquet_file_path, json_file_path)
