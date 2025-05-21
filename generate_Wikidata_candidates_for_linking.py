import requests_cache
import requests
import time
import json
import os
from rapidfuzz import fuzz
from retry import retry
import argparse
import re
from tqdm import tqdm
requests_cache.install_cache('wikipedia_cache', backend='sqlite', expire_after=360000)

WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_CLAIMS_URL = "https://www.wikidata.org/w/api.php"
DISAMBIGUATION_QID = "Q4167410"

@retry(tries=3, delay=5, max_delay=60)
def get_wikidata_candidates_fuzzy(mention, search_item="", top_k=5, search_limit=128, min_score=10, language="en"):
    """
    Search Wikidata for candidate entities matching a mention using fuzzy matching,
    and exclude disambiguation pages.

    Args:
        mention (str): The input entity mention.
        top_k (int): Number of top candidates to return after fuzzy ranking.
        search_limit (int): Number of initial candidates to fetch from the API.
        min_score (int): Minimum fuzzy match score to keep a candidate.
        language (str): Language for the label/description search.

    Returns:
        list of dict: Filtered and ranked entity candidates.
    """
    # Step 1: Search
    if search_item == "" or search_item is None or search_item == "None" or search_item == "entity":
        search_item = "item"

    elif search_item == "property" or search_item == "prop" or search_item == "relation":
        search_item = "property"

    else:
        search_item = "item"
    
    params = {
            "action": "wbsearchentities",
            "search": mention,
            "language": language,
            "format": "json",
            "limit": search_limit,
            "type": search_item
        }

    response = requests.get(WIKIDATA_SEARCH_URL, params=params)
    data = response.json()
    raw_results = data.get("search", [])

    if not raw_results:
        return []

    scored_results = []
    for result in raw_results:
        label = result.get("label", "")
        aliases = result.get("aliases", [])
        description = result.get("description", "")

        # Compute fuzzy score between mention and label
        label_score = fuzz.ratio(mention.lower(), label.lower())
        max_alias_score = max((fuzz.ratio(mention.lower(), a.lower()) for a in aliases), default=0)

        score = max(label_score, max_alias_score)

        if score >= min_score:
            scored_results.append({
                "id": result.get("id"),
                "label": label,
                "description": description,
                "score": score
            })

    # Step 2: Sort by fuzzy score
    scored_results.sort(key=lambda x: x["score"], reverse=True)

    # Step 3: Check and exclude disambiguation pages
    final_results = []
    for candidate in scored_results:
        entity_id = candidate["id"]
        claim_params = {
            "action": "wbgetclaims",
            "entity": entity_id,
            "property": "P31",
            "format": "json"
        }

        claim_response = requests.get(WIKIDATA_CLAIMS_URL, params=claim_params)
        claim_data = claim_response.json()
        claims = claim_data.get("claims", {}).get("P31", [])

        is_disambiguation = any(
            c.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id") == DISAMBIGUATION_QID
            for c in claims
        )

        if not is_disambiguation:
            final_results.append(candidate)

        if len(final_results) >= top_k:
            break

        time.sleep(0.05)  # polite rate limiting

    return final_results

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

def extract_bracket_content(text):
    pattern = r"\[\s*'([^']+)',\s*'([^']+)',\s*'([^']+)'\s*\]"  # Regex pattern to find content inside []
    extraction = re.findall(pattern, text)  # Returns a list of matches
    if extraction == []:
        pattern = pattern = r'\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]'
        extraction = re.findall(pattern, text)
        if extraction == []:
            pattern = r'\[\s*"\s*([^"]+)\s*"\s*,\s*"\s*([^"]+)\s*"\s*\]'
            extraction = re.findall(pattern, text)
            if extraction == []:
                pattern = r'\[\s*"\s*([^"]+)\s*"\s*,\s*"\s*([^"]+)\s*"\s*,\s*(?:"([^"]+)"|([\d.]+))\s*\]'
                extraction = re.findall(pattern, text)
    return extraction
      

def remove_duplicates(extracted_list):
    unique_items = []
    for item in extracted_list:
        if item not in unique_items:
            unique_items.append(item)
    return unique_items

def main():
    parser = argparse.ArgumentParser(description='Query Wikidata for entity/property candidates.')
    parser.add_argument('--input_file', type=str, default= "ENEXA_Demo2/IE_extraction_output/extraction_one_page/LLM_answers.jsonl", help='Input file with mentions.')
    parser.add_argument('--output_folder', type=str, default= "ENEXA_Demo2/candidates_adidas", help='Output folder for results.')
    parser.add_argument('--num_candidates', type=int, default=8, help='Number of candidates to return.')

    args = parser.parse_args()
    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f"{key}: {value}")

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        print(f"Output folder {args.output_folder} created.")

    input_data = read_jsonL(args.input_file)
    generated_candidates = []
    for i in tqdm(input_data, desc="Processing", unit="item"):
        extraction = extract_bracket_content(i["LLM answer"])
        extraction = remove_duplicates(extraction)

        if i["task"] == "NER":
            entities = []
            candidates = []
            for entity in extraction:
                if len(entity) == 2:
                    entities.append({"entity": entity[0], "type": entity[1]})
                    candidates.append({"item": entity[0], "candidates": get_wikidata_candidates_fuzzy(entity[0], search_item="item", top_k=args.num_candidates)})
            i["entities"] = entities
            i["candidates"] = candidates

        if i["task"] == "RE":
            triples = []
            candidates = []
            for triple in extraction:
                if len(triple) == 3:
                    triples.append({"subject": triple[0], "predicate": triple[1], "object": triple[2]})
                    sbj_candidates = get_wikidata_candidates_fuzzy(triple[0], search_item="item", top_k=args.num_candidates)
                    candidates.append({"item": triple[0], "candidates": sbj_candidates})
                    obj_candidates = get_wikidata_candidates_fuzzy(triple[2], search_item="item", top_k=args.num_candidates)
                    candidates.append({"item": triple[2], "candidates": obj_candidates})
                    relation_candidates = get_wikidata_candidates_fuzzy(triple[1], search_item="property", top_k=args.num_candidates)
                    candidates.append({"item": triple[1], "candidates": relation_candidates})
            i["triples"] = triples
            i["candidates"] = candidates
        del i["LLM answer"]
        generated_candidates.append(i)
    print("Generated candidates for all entities and relations.")
    print("Saving results...")

    write_jsonL(generated_candidates, args.output_folder + "/extraction_and_candidates.jsonl")
    #with open(args.output_folder + "/extraction_and_candidates.json", "w", encoding='utf-8') as out_file:
        #json.dump(input_data, out_file, ensure_ascii=False, indent=4)
    print(f"Results saved to {args.output_folder}/extraction_and_candidates.jsonl")
           
if __name__ == "__main__":
    main()