import requests
import requests_cache
requests_cache.install_cache('wikidata_cache', expire_after=360000)  
import json
import os
from retry import retry
import argparse
import re
from tqdm import tqdm


def read_jsonL(file):
    data = []
    with open(file, "r", encoding='utf-8') as read_file:
        lines = read_file.readlines()
        for line in lines:
            data.append(json.loads(line))
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

def get_extracted_items(input_data):
    """
    Extracts items from the input data that contain entities or triples.

    Args:
        input_data (list): List of dictionaries containing LLM answers.

    Returns:
        list: List of dictionaries with extracted entities or triples.
    """
    extracted_items = []
    for i in tqdm(input_data, desc="Processing", unit="item"):
        extraction = extract_bracket_content(i["LLM answer"])
        extraction = remove_duplicates(extraction)

        if i["task"] == "NER":
            entities = []
            for entity in extraction:
                if len(entity) == 2:
                    entities.append({"entity": entity[0], "type": entity[1]})
            i["entities"] = entities

        if i["task"] == "RE":
            triples = []
            for triple in extraction:
                if len(triple) == 3:
                    triples.append({"subject": triple[0], "predicate": triple[1], "object": triple[2]})
            i["triples"] = triples
        del i["LLM answer"]
        
        extracted_items.append(i)

    return extracted_items

def is_WD_disambiguation_page(qid):
    """
    Check if a Wikidata item is a disambiguation page.

    Args:
        qid (str): The Wikidata Q-ID (e.g., 'Q12345').

    Returns:
        bool: True if the item is a disambiguation page, False otherwise.
    """
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Get entity data
        entity = data["entities"].get(qid)
        if not entity:
            return False

        claims = entity.get("claims", {})
        p31_claims = claims.get("P31", [])

        # Check if any 'instance of' value is Q4167410 (disambiguation page)
        for claim in p31_claims:
            datavalue = claim.get("mainsnak", {}).get("datavalue", {})
            if datavalue.get("value", {}).get("id") == "Q4167410":
                return True
    except requests.RequestException as e:
        print(f"Request error: {e}")
    except Exception as e:
        print(f"Error processing data: {e}")
    
    return False


@retry(tries=3, delay=5, max_delay=60)
def get_wikidata_candidates(mention, language='en', limit=10, search_type='item'):
    import requests

    if search_type not in ['item', 'property']:
        raise ValueError("search_type must be either 'item' or 'property'")

    endpoint = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": mention,
        "language": language,
        "limit": limit,
        "type": search_type,
        "format": "json"
    }

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        results = data.get("search", [])
        candidates = []
        for item in results:
            candidates.append({
                "id": item.get("id"),
                "label": item.get("label"),
                "description": item.get("description"),
                "type": search_type
            })
        return candidates
    except Exception as e:
        print(f"Error fetching Wikidata candidates: {e}")
        return []
    
def get_candidates_list_as_string(candidates):
    cands = []
    for candidate in candidates:
        disambiguation_page = is_WD_disambiguation_page(candidate["id"])
        if disambiguation_page == False:
            cands.append(f"{candidate['id']} - {candidate['label']}: {candidate['description']}")
    
    return json.dumps(cands, ensure_ascii=False, indent=4)

def trim_property_name(property_name):
    
    # 1. Convert to lowercase
    temp_name = property_name.lower()

    # 2. Replace underscores with spaces
    temp_name = temp_name.replace('_', ' ')

    # 3. Remove common leading phrases (case-insensitive due to step 1)
    # Using regex to ensure it's at the beginning of the string
    temp_name = re.sub(r"^(is |was |are |were |been |has |have |had |having )", "", temp_name).strip()

    # 4. Remove common trailing phrases (case-insensitive due to step 1)
    # Using regex to ensure it's at the end of the string
    temp_name = re.sub(r"( by| for| with| from| on| in| of| at| to)$", "", temp_name).strip()
    
    # 5. Strip any extra whitespace that might have accumulated
    temp_name = temp_name.strip()

    return temp_name


def main():
    parser = argparse.ArgumentParser(description='Query Wikidata for entity/property candidates.')
    parser.add_argument('--input_file', type=str, default= "ENEXA_Demo2/IE_extraction_output/extraction_one_schema_per_task/LLM_answers.jsonl", help='Input file with mentions.')
    parser.add_argument('--output_folder', type=str, default= "ENEXA_Demo2/candidates_adidas_v2", help='Output folder for results.')
    parser.add_argument('--num_candidates', type=int, default=5, help='Number of candidates to return.')

    args = parser.parse_args()
    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f"{key}: {value}")

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        print(f"Output folder {args.output_folder} created.")

    input_data = read_jsonL(args.input_file)

    extracted_items = get_extracted_items(input_data)
    print("Generated candidates for all entities and relations.")

    candidates_added = []
    for item in tqdm(extracted_items, desc="Generating candidates", unit="item"):
        if "entities" in item:
            for entity in item["entities"]:
                candidates = get_wikidata_candidates(entity["entity"], limit=args.num_candidates, search_type='item')
                entity["candidates"] = get_candidates_list_as_string(candidates)
        
        if "triples" in item:
            for triple in item["triples"]:
                subject_candidates = get_wikidata_candidates(triple["subject"], limit=args.num_candidates, search_type='item')
                triple["subject candidates"] = get_candidates_list_as_string(subject_candidates)
                pred = trim_property_name(triple["predicate"])
                predicate_candidates = get_wikidata_candidates(pred, limit=args.num_candidates, search_type='property')
                triple["predicate candidates"] = get_candidates_list_as_string(predicate_candidates)
                object_candidates = get_wikidata_candidates(triple["object"], limit=args.num_candidates, search_type='item')
                triple["object candidates"] = get_candidates_list_as_string(object_candidates)
        candidates_added.append(item)
    print("Saving results...")

    write_jsonL(candidates_added, args.output_folder + "/extraction_and_candidates.jsonl")
    print(f"Results saved to {args.output_folder}/extraction_and_candidates.jsonl")
           
if __name__ == "__main__":
    main()