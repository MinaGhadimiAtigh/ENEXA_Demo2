import unsloth
import json
import os
from collections import defaultdict
import argparse
from huggingface_hub import login
from datasets import Dataset
from transformers import AutoTokenizer #, TextStreamer
from unsloth import FastLanguageModel
import torch
print(torch.cuda.is_available())  # Should return True if GPUs are available
print(torch.cuda.device_count())  # Number of GPUs available


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# Login directly with your token
login(token="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")


def read_jsonL(file):
    data = []
    with open(file, "r", encoding='utf-8') as read_file:
        lines = read_file.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data

def read_json(file):
    """Read from an external file: json file
    """	
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
   
    return data

def write_jsonL(data, file):
    with open(file, "w", encoding='utf-8') as write_file:
        for line in data:
            json.dump(line, write_file, ensure_ascii=False)
            write_file.write("\n")

def write_json(data, file):
    """Write to an external file: json file
    """	
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def create_message_column(row):
    """Create a conversational message column for the dataset."""

    # Initialize an empty list to store the messages.
    messages = []

    # Create a 'system' message dictionary with 'content' and 'role' keys.
    system = {
        "content": "You are a helpful AI assistant specializing in Entity Disambiguation. You will be given a mention and some context. Your task is to select the correct entity from the given candidates.",
        "role": "system"
    }

    # Append the 'system' message to the 'messages' list.
    messages.append(system)
    
    # Create a 'user' message dictionary with 'content' and 'role' keys.
    user = {
        "content": row["prompt"],
        "role": "user"
    }
    
    # Append the 'user' message to the 'messages' list.
    messages.append(user)
    
    # Return a dictionary with a 'messages' key and the 'messages' list as its value.
    return {"messages": messages}


def format_dataset_chatml(row, tokenizer):
    return {"text": tokenizer.apply_chat_template(row["messages"], add_generation_prompt=True, tokenize=False)}


#################################################################################################

def main():
    parser = argparse.ArgumentParser(description='Test the model with the shadowlinks dataset')
    parser.add_argument('--input_file', type=str, default="ENEXA_Demo2/IE_extraction_output/ft_phi4_semiclosed_with_example_test/LLM_answers.jsonl", help='The input file where the context is stored')
    parser.add_argument('--prompt_template', type=str, default="ENEXA_Demo2/input_files/entity_disambiguation_temaplate_can_reject.json", help='The prompt template to use for the test')
    parser.add_argument('--LLM', type=str, default="FinaPolat/unsloth_llama3_8B_for_ED", help='The model to use for the test')
    parser.add_argument('--temperature', type=int, default=0.001, help='The temperature to use for the test')
    parser.add_argument('--max_tokens', type=int, default=5, help='The maximum number of tokens to use for the test')
    parser.add_argument('--output_folder', type=str, default="inference/api_candidates_HF", help='The output file where the errors are stored')
    args = parser.parse_args()

    args_dict = vars(args)
    for key in args_dict:
        print(f"{key}: {args_dict[key]}")

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    template = read_json(args.prompt_template)

    input_data = read_jsonL(args.input_file)
    prompts = []
    for indx, i in enumerate(input_data):
        for item in i["candidates"]:
            cands = []
            for c in item["candidates"]:
                cands.append(f'{c["id"]} -- {c["label"]}: {c["description"]}')
            prompt = template["formatter"].format(mention= item["item"],
                                                  text = i["input text"].replace(item["item"], f'#{item["item"]}#'),
                                                  candidates = json.dumps(cands),
                                                  )
            prompts.append({"index": indx,
                               "item to disambiguate": item["item"],
                               "prompt": prompt,
                               })

    # Load dataset
    data = Dataset.from_list(prompts)
    print("Test Dataset loaded", flush=True)
    print(len(data), flush=True)
    print(data[0], flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.LLM)
    data = data.map(create_message_column)
    data = data.map(format_dataset_chatml, fn_kwargs={"tokenizer": tokenizer})
    columns_to_remove = ["messages", "prompt"]
    data = data.map(lambda x: x, remove_columns=columns_to_remove)

    max_seq_length = 3072
    model, tokenizer = FastLanguageModel.from_pretrained(
                                                            model_name = args.LLM, # YOUR MODEL YOU USED FOR TRAINING
                                                            max_seq_length = max_seq_length,
                                                            dtype = None,
                                                            load_in_4bit = True,
                                                            )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add to tokenizer
    model.resize_token_embeddings(len(tokenizer))

    FastLanguageModel.for_inference(model) # Enable native 2x faster inference


    LLM_answers = []
    #for i in range(len(data)):
    for i in range(10):
        print(f"Generating answer for row {i}...", flush=True)
        row = data[i]   
        inputs = tokenizer(row["text"], return_tensors="pt").to("cuda")
        outputs = model.generate(input_ids=inputs['input_ids'], 
                                  attention_mask=inputs["attention_mask"], 
                                  max_new_tokens = args.max_tokens,
                                  temperature=args.temperature,
                                  do_sample=True,
                                  ) 
        prompt_length = inputs['input_ids'].shape[1]
        generated_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        #print("LLM answer", generated_text, flush=True)
        LLM_answers.append({"index": row["index"],
                            "item to disambiguate": row["item to disambiguate"],
                            "disambiguation output": generated_text,
                            "candidates": input_data[row["index"]]["candidates"],
                            })

    write_json(LLM_answers[:10], os.path.join(args.output_folder, "LLM_answers_to_inspect.json"))
    # Group by index
    grouped = defaultdict(list)
    for entry in LLM_answers:
        grouped[entry["index"]].append({
            "item": entry["item to disambiguate"],
            "output": entry["disambiguation output"]
        })

    # Write to JSONL (one line per index)
    with open(os.path.join(args.output_folder, "disambiguation_output.jsonl"), "w", encoding="utf-8") as out:
        for idx in sorted(grouped.keys()):
            line = {
                "index": idx,
                "disambiguations": grouped[idx]
            }
            out.write(json.dumps(line, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
