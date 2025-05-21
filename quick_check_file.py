import json

def read_jsonl(file_path):
    """This function reads a jsonl file and returns a list of dictionaries"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

data = read_jsonl('ENEXA_Demo2/disambiguation_output/raged_output_adidas/disambiguation_output.jsonl')


for idx, item in enumerate(data):
    if idx != item["index"]:
        print(f"Index mismatch at index {idx}: {item['index']}")