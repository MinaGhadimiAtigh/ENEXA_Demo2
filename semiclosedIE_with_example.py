import json
import os
import argparse
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
    parser.add_argument('--temperature', type=int, default=0.001, help='The temperature to use for the test')
    parser.add_argument('--max_tokens', type=int, default=3072, help='The maximum number of tokens to use for the test')
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
    for file in os.listdir(args.input_folder):
        if file.endswith(".json"):
            file_path = os.path.join(args.input_folder, file)
            data = load_json(file_path)
            for item in data:
                if "url" in item:
                    url = item["url"]
                #NER: Named Entity Recognition
                if "heading" and "paragraphs" in item:
                    for paragraph in item["paragraphs"]:
                        for t in target_entity_types:
                            text = f'article: {file.split(".")[0]}, {item["heading"]}: {paragraph}'
                            prompt = template["formatter"].format(task= "NER: Named Entity Recognition",
                                                                example= NER_example,
                                                                schema=json.dumps(t["types"], ensure_ascii=False),
                                                                inputs= text,
                                                                output_format = '[["Entity", "Type"], ...]')
                            input_data.append({"article": file.split(".")[0],
                                                "heading": item["heading"],
                                               "input type": "paragraph",
                                               "url": url,
                                                "task": "NER",
                                                "schema": t["types"],
                                                "input text": text,
                                                "prompt": prompt})
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

                #RE: Relation Extraction
                if "heading" and "paragraphs" in item:
                    for paragraph in item["paragraphs"]:
                        for p in target_relations:
                            text = f'article: {file.split(".")[0]}, {item["heading"]}: {paragraph}'
                            prompt = template["formatter"].format(task= "RE: Relation Extraction",
                                                                  example= RE_example,
                                                                  schema=json.dumps(p["relations"], ensure_ascii=False),
                                                                  inputs= text,
                                                                  output_format = '[["Subject", "Relation", "Object"], ...]')
                            input_data.append({"article": file.split(".")[0],
                                               "heading": item["heading"],
                                               "input type": "paragraph",
                                               "url": url,
                                               "task": "RE",
                                               "schema": p["relations"],
                                               "input text": text,
                                               "prompt": prompt})
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

    print("Input data is created", flush=True)
    print(len(input_data), flush=True)

    # Load dataset
    dataset = Dataset.from_list(input_data)
    print("Data for inference is loaded", flush=True)
    print(len(dataset), flush=True)
    print(dataset[0], flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.LLM)
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
    model.resize_token_embeddings(len(tokenizer))
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    #text_streamer = TextStreamer(tokenizer)

    LLM_answers = []
    for i in range(len(data)):
    #for i in range(5):
        print(f"Generating answer for row {i}...", flush=True)
        row = data[i]   
        inputs = tokenizer(row["text"], return_tensors="pt", padding=True, truncation=True).to("cuda")
        outputs = model.generate(input_ids=inputs['input_ids'], 
                                 attention_mask=inputs["attention_mask"], 
                                 max_new_tokens = 2048,
                                 temperature=args.temperature,
                                 do_sample=True,
                                 ) #streamer = text_streamer, add steamer if necessary
        prompt_length = inputs['input_ids'].shape[1]
        generated_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        #print("LLM answer", generated_text, flush=True)
        LLM_answers.append({"article": row["article"],
                            "heading": row["heading"],
                            "url": row["url"],
                            "input type": row["input type"],
                            "task": row["task"],
                            "schema": row["schema"],
                            "input text": row["input text"],
                            "LLM answer": generated_text})
      
    write_jsonL(LLM_answers, f"{args.output_folder}/LLM_answers.jsonl")
    write_json(args_dict, f"{args.output_folder}/args.json")


if __name__ == "__main__":
    main()
