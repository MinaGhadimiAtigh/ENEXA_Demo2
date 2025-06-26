import json
import os
import argparse
import math
import time
from tqdm import tqdm
import unsloth
from huggingface_hub import login
from datasets import Dataset
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
import torch
print(torch.cuda.is_available())  # Should return True if GPUs are available
print(torch.cuda.device_count())  # Number of GPUs available


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# Login directly with your token
login(token="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")

def load_json(file):
    """Load a json file
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
        "content": "You are a helpful AI assistant specializing in Information Extraction (IE) tasks such as Named Entity Recognition (NER) or Relation Extraction (RE). Analyze the following text and provide the requested information.",
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
    parser.add_argument('--input_folder', type=str, default="ENEXA_Demo2/wiki_downloads", help='The input folder where the dowloaded Wikipedia files are stored')
    parser.add_argument('--prompt_template', type=str, default="ENEXA_Demo2/input_files/prompt_template_with example.json", help='The prompt template to fill in the inputs.')
    parser.add_argument('--target_entity_types', type=str, default="ENEXA_Demo2/input_files/target_entity_types.json", help='The target entity types to extract from the Wikipedia files.')
    parser.add_argument('--target_relations', type=str, default="ENEXA_Demo2/input_files/target_relations_shorter.json", help='The target relations to extract from the Wikipedia files.')
    parser.add_argument('--NER_example', type=str, default="ENEXA_Demo2/input_files/NER_example.json", help='NER example to use in the prompt template.')
    parser.add_argument('--RE_example', type=str, default="ENEXA_Demo2/input_files/RE_example.json", help='RE example to use in the prompt template.')
    parser.add_argument('--LLM', type=str, default="FinaPolat/phi4_adaptable_IE", help='The model to use for the test')
    #parser.add_argument('--batch_size', type=int, default=1, help='The batch size to use for the test')
    parser.add_argument('--temperature', type=int, default=0.001, help='The temperature to use for the test')
    parser.add_argument('--max_tokens', type=int, default=8192, help='The maximum number of tokens to use for the test')
    parser.add_argument('--output_folder', type=str, default="ENEXA_Demo2/IE_extraction_output", help='The output folder where the resulting triples will be stored')
    args = parser.parse_args()

    args_dict = vars(args)
    for key in args_dict:
        print(f"{key}: {args_dict[key]}")

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    template = load_json(args.prompt_template)
    target_entity_types = load_json(args.target_entity_types)
    print("Target entity type categories", flush=True)
    print(len(target_entity_types), flush=True)
    target_relations = load_json(args.target_relations)
    print("Target relation categories", flush=True)
    print(len(target_relations), flush=True)
    NER_example = load_json(args.NER_example)
    RE_example = load_json(args.RE_example)

    input_data = []
    prompts = []
    for file in os.listdir(args.input_folder):
        if file.endswith(".json"):
            file_path = os.path.join(args.input_folder, file)
            data = load_json(file_path)
            print(f"Processing file: {file}", flush=True)
            print(f"Number of articles in {file}: {len(data)}", flush=True)
            # If the file is a Wikipedia dump, it should contain a list of articles
            # Each article is a dictionary with "url", "heading", "paragraphs", and "table" keys
            for item in data:
                if "url" in item:
                    url = item["url"]
                #NER: Named Entity Recognition
                if "heading" and "paragraphs" in item:
                    merges_parags = " ".join(item["paragraphs"])
                    #print(merges_parags, flush=True)
                    for t in target_entity_types:
                        #print(f"Target entity types: {t['types']}", flush=True)
                        text = f'article: {file.split(".")[0]}, {item["heading"]}: {merges_parags}'
                        prompt = template["formatter"].format(task= "NER: Named Entity Recognition",
                                                                example= NER_example,
                                                                schema=json.dumps(t["types"], ensure_ascii=False),
                                                                inputs= text,
                                                                output_format = '[["Entity", "Type"], ...]')
                        #print(prompt, flush=True)
                        
                        input_data.append({"article": file.split(".")[0],
                                            "heading": item["heading"],
                                            "input type": "paragraph",
                                            "url": url,
                                            "task": "NER",
                                            "schema": t["types"],
                                            "input text": text,
                                            "prompt": prompt})
                        prompts.append({"prompt": prompt})

                if "heading" and "table" in item:
                    for t in target_entity_types:
                        text = f'article: {file.split(".")[0]}, {item["heading"]}: {json.dumps(item["table"], ensure_ascii=False)}'
                        prompt = template["formatter"].format(task= "NER: Named Entity Recognition",
                                                            example= NER_example,
                                                            schema=json.dumps(t["types"], ensure_ascii=False),
                                                            inputs= text,
                                                            output_format = '[["Entity", "Type"], ...]')
                        input_data.append({"article": file.split(".")[0],
                                            "heading": item["heading"],
                                           "input type": "table",
                                            "url": url,
                                            "task": "NER",
                                            "schema": t["types"],
                                            "input text": text,
                                            "prompt": prompt})
                        prompts.append({"prompt": prompt})

                #RE: Relation Extraction
                if "heading" and "paragraphs" in item:
                    merges_parags = " ".join(item["paragraphs"])
                    #print(merges_parags, flush=True)
                    for p in target_relations:
                        #print(f"Target relations: {p['relations']}", flush=True)
                        text = f'article: {file.split(".")[0]}, {item["heading"]}: {" ".join(item["paragraphs"])}'
                        prompt = template["formatter"].format(task= "RE: Relation Extraction",
                                                                  example= RE_example,
                                                                  schema=json.dumps(p["relations"], ensure_ascii=False),
                                                                  inputs= text,
                                                                  output_format = '[["Subject", "Relation", "Object"], ...]')
                        #print(prompt, flush=True)
                        input_data.append({"article": file.split(".")[0],
                                            "heading": item["heading"],
                                            "input type": "paragraph",
                                            "url": url,
                                            "task": "RE",
                                            "schema": p["relations"],
                                            "input text": text,
                                            "prompt": prompt})
                        prompts.append({"prompt": prompt})

                if "heading" and "table" in item:
                    for p in target_relations: 
                        text = f'article: {file.split(".")[0]}, {item["heading"]}: {json.dumps(item["table"], ensure_ascii=False)}'  
                        prompt = template["formatter"].format(task="Relation Extraction",
                                                              example= RE_example,
                                                              schema=json.dumps(p["relations"], ensure_ascii=False),
                                                              inputs= text,
                                                              output_format = '[["Subject", "Relation", "Object"], ...]')
                        input_data.append({"article": file.split(".")[0],
                                            "heading": item["heading"],
                                           "input type": "table",
                                            "url": url,
                                            "task": "RE",
                                            "schema": p["relations"],
                                            "input text": text,
                                            "prompt": prompt})
                        prompts.append({"prompt": prompt})

    print("Prompts are created", flush=True)
    print(len(input_data), flush=True)

    # Load dataset
    dataset = Dataset.from_list(prompts)
    print("Data for inference is loaded", flush=True)
    print(len(dataset), flush=True)
    print(dataset[0], flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.LLM, padding_side="left")
    data = dataset.map(create_message_column)
    data = data.map(format_dataset_chatml, fn_kwargs={"tokenizer": tokenizer})
    columns_to_remove = ["messages", "prompt"]
    data = data.map(lambda x: x, remove_columns=columns_to_remove)
    print("Data for inference is formatted", flush=True)
    print(len(data), flush=True)
    print(data[0], flush=True)

    max_seq_length = args.max_tokens
    model, tokenizer = FastLanguageModel.from_pretrained(
                                                            model_name = args.LLM, 
                                                            max_seq_length = max_seq_length,
                                                            dtype = None,
                                                            load_in_4bit = True,
                                                            )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add to tokenizer
    tokenizer.padding_side = "left"
    model.resize_token_embeddings(len(tokenizer))
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    #text_streamer = TextStreamer(tokenizer)

    # batch_size = args.batch_size if hasattr(args, 'batch_size') else 1
    # data_list = data.to_list()
    # num_batches = math.ceil(len(data_list) / batch_size)  # Limit to first 3 batches for testing
    # print(f"Number of batches: {num_batches}", flush=True)
    # print(f"Batch size: {batch_size}", flush=True)
    # start_time = time.time()  # Start timing

    LLM_answers = []
    for i in range(len(data)):
    #for i in range(5):
        print(f"Generating answer for row {i}...", flush=True)   
        print(data[i]["text"], flush=True)
        inputs = tokenizer(data[i]["text"], return_tensors="pt").to("cuda")
        outputs = model.generate(input_ids=inputs['input_ids'], 
                                  attention_mask=inputs["attention_mask"], 
                                  max_new_tokens = args.max_tokens,
                                  temperature=args.temperature,
                                  do_sample=True,
                                  ) 
        prompt_length = inputs['input_ids'].shape[1]
        generated_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        #print("LLM answer", generated_text, flush=True)
        row = input_data[i]
        LLM_answers.append({
                    "article": row["article"],
                    "heading": row["heading"],
                    "url": row["url"],
                    "input type": row["input type"],
                    "task": row["task"],
                    "schema": row["schema"],
                    "input text": row["input text"],
                    "prompt": row["prompt"],
                    "LLM answer": generated_text,
                })

    # with tqdm(total=num_batches, desc="Generating", unit="batch") as pbar:
    #     for batch_idx in range(num_batches): #range(2):
    #         start = batch_idx * batch_size
    #         end = min(start + batch_size, len(data_list))
    #         batch = data_list[start:end]

    #         prompts = [item["text"] for item in batch]
    #         encoded = tokenizer(
    #                         prompts,
    #                         return_tensors="pt",
    #                         padding=True,
    #                         truncation=True,
    #                     ).to("cuda")
            
    #         with torch.no_grad():
    #             outputs = model.generate(
    #             input_ids=encoded["input_ids"],
    #             attention_mask=encoded["attention_mask"],
    #             max_new_tokens=2048,
    #             temperature=args.temperature,
    #             do_sample=True,
    #             pad_token_id=tokenizer.pad_token_id,
    #             eos_token_id=tokenizer.eos_token_id,
    #             )

    #         for i, output in enumerate(outputs):
    #             input_len = (encoded["attention_mask"][i] == 1).sum().item()
    #             generated_text = tokenizer.decode(output[input_len:], skip_special_tokens=True)
    #             item = batch[i]


    #        pbar.update(1)

    # end_time = time.time()  # End timing
    # elapsed_time = end_time - start_time
    # print(f"Total time taken: {elapsed_time:.2f} seconds")
    # print("LLM answers are generated", flush=True)
      
    write_jsonL(LLM_answers, f"{args.output_folder}/LLM_answers.jsonl")
    write_json(args_dict, f"{args.output_folder}/args.json")


if __name__ == "__main__":
    main()
