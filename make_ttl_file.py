# we start with imports
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDFS, RDF, OWL, XSD
from urllib.parse import quote
import json
import string
import argparse
import os
import re

def read_jsonL(file_path):
    """This function reads a jsonl file and returns a list of dictionaries"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def read_json(file_path):
    """This function reads a json file and returns a dictionary"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def make_dictionary_from_list(list_of_dicts):
    my_dict = {}
    for d in list_of_dicts:
        my_dict[d['item']] = d['output']
    return my_dict


def remove_punctuation(text):
    """Utility function to remove punctuation"""
    return re.sub(r'[^\w\s]', '', text)


def shape_relation_name(rel_string, prefix):
    """Format a relation name as: prefix:relationName (camelCase, URI-encoded)"""
    rel = rel_string.replace('_', ' ')
    rel = remove_punctuation(rel)  # FIXED
    rel = rel.strip().lower().split()
    
    if len(rel) > 1:
        words = [rel[0].lower()] + [word.capitalize() for word in rel[1:]]
        rel = ''.join(words)
    elif rel:
        rel = rel[0].lower()
    else:
        rel = 'unknown'

    rel = quote(rel)
    return URIRef(prefix + rel)


def shape_entity_name(entity_string, prefix):
    """Format an entity name as: prefix:entity_name (snake_case, URI-encoded)"""
    ent = entity_string.replace('_', ' ').replace('-', ' ')
    ent = remove_punctuation(ent)
    ent = ent.strip().lower().replace(' ', '_')
    
    # Replace special characters with currency codes
    ent = ent.replace('€', 'EUR').replace('$', 'USD').replace('£', 'GBP')

    # Normalize separators
    ent = re.sub(r'[-.,()]+', '_', ent)

    ent = quote(ent)
    return URIRef(prefix + ent)


def shape_class_name(type_string, prefix):
    """Format a class name as: prefix:ClassName (PascalCase, URI-encoded)"""
    type_name = type_string.replace('_', ' ')
    type_name = remove_punctuation(type_name)
    type_name = type_name.strip().split()
    
    if not type_name:
        shaped_type = 'Unknown'
    else:
        shaped_type = ''.join(word.capitalize() for word in type_name)

    shaped_type = quote(shaped_type)
    return URIRef(prefix + shaped_type)


def get_disambiguation_ids(extraction, disambiguation):
    def format_wiki_id(label, raw_id, expected_prefix):
        """Skip bad IDs, add prefix if missing, return (label, id) tuple or None."""
        if "None of" in raw_id:
            return None
        if not raw_id.startswith(expected_prefix):
            raw_id = expected_prefix + raw_id
        return (label, raw_id)

    disamb_set = set()
    disambiguation_items = make_dictionary_from_list(disambiguation)

    # Process entities
    for entity in extraction.get("entities", []):
        ent = entity["entity"]
        if ent in disambiguation_items:
            result = format_wiki_id(ent, disambiguation_items[ent], "Q")
            if result:
                disamb_set.add(result)

    # Process triples
    for triple in extraction.get("triples", []):
        for role, prefix in [("subject", "Q"), ("object", "Q"), ("predicate", "P")]:
            term = triple.get(role)
            if term in disambiguation_items:
                result = format_wiki_id(term, disambiguation_items[term], prefix)
                if result:
                    disamb_set.add(result)

    return list(disamb_set)

                    

def get_types_dict(types_file):
    entity_types = read_json(types_file)
    types_dict = {}
    for i in entity_types:
        all_types = i["types"]
        for j in all_types:
            for k, v in j.items():
                #print(k)
                #print(v['wikiID'])
                types_dict[k] = v['wikiID']
    return types_dict


def create_ttl_file():
    parser = argparse.ArgumentParser(description='Create a TTL file from extraction and disambiguation data')
    parser.add_argument('--extraction_file', type=str, default="ENEXA_Demo2/IE_extraction_output/LLM_answers.jsonl", help='The input file where the extraction results are stored')
    parser.add_argument('--disambiguation_file', type=str, default="ENEXA_Demo2/disambiguation_output/disambiguation_output.jsonl", help='The input file where the disambuguation results are stored')
    parser.add_argument('--experiment', type=str, default="adidas_text_only", help='The name of the experiment')
    parser.add_argument('--types_file', type=str, default="ENEXA_Demo2/input_files/target_entity_types_4_adidas_wikiIDs.json", help='The input file where the types are stored')
    parser.add_argument('--output_folder', type=str, default="ENEXA_Demo2/graph_output", help='The output folder where the ttl file is stored')

    args = parser.parse_args()

    args_dict = vars(args)
    for key in args_dict:
        print(f"{key}: {args_dict[key]}")

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    extraction = read_jsonL(args.extraction_file)
    disambiguation = read_jsonL(args.disambiguation_file)
    types_dict = get_types_dict(args.types_file)

    merged_data = []
    for i in disambiguation:
        input_data = extraction[i["index"]]
        input_data["disambiguations"] = i["disambiguations"]
        merged_data.append(input_data)

    print(len(merged_data), "extractions merged with disambiguations")

    with open(f"{args.output_folder}/{args.experiment}_final_pipeline_output.jsonl", "w", encoding="utf-8") as f:
        for i in merged_data:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
    print(f"Merged data file created at: {args.output_folder}/final_pipeline_output.jsonl")

    ENX = Namespace("http://example.org/enexa-adaptableIE/")
    WIKI = Namespace("https://www.wikidata.org/wiki/")

    g = Graph() # create a graph object
    g.bind("adaptableIE", ENX,  override=True) 
    g.bind("wiki", WIKI, override=True) 

    for k, v in types_dict.items():
        class_name = shape_class_name(k, ENX)
        g.add((class_name, RDF.type, OWL.Class))
        g.add((class_name, RDFS.label, Literal(k, datatype=XSD.string)))
        g.add((class_name, OWL.sameAs, URIRef(WIKI + quote(v))))


    for i in merged_data:  
        disamb_tuples = get_disambiguation_ids(i, i["disambiguations"])

        if "entities" in i: 
            for j in i["entities"]:
                entity = shape_entity_name(j["entity"], ENX)
                #g.add((entity, RDF.type, OWL.NamedIndividual))
                g.add((entity, RDFS.label, Literal(j["entity"], datatype=XSD.string)))
                if "type" in j:
                    class_name = shape_class_name(j["type"], ENX)
                    g.add((class_name, RDF.type, OWL.Class))
                    g.add((entity, RDF.type, class_name))
            
            for tup in disamb_tuples:
                entity = shape_entity_name(tup[0], ENX)
                object_uri = URIRef(WIKI + quote(tup[1]))
                g.add((entity, OWL.sameAs, object_uri))

        if "triples" in i:
            
            for j in i["triples"]:
                subject = shape_entity_name(j["subject"], ENX)
                #g.add((subject, RDF.type, OWL.NamedIndividual))
                g.add((subject, RDFS.label, Literal(j["subject"], datatype=XSD.string)))
                
                object = shape_entity_name(j["object"], ENX)
                g.add((object, RDFS.label, Literal(j["object"], datatype=XSD.string)))
                    
                predicate = shape_relation_name(j["predicate"], ENX)
                g.add((predicate, RDF.type, OWL.ObjectProperty))
                g.add((predicate, RDFS.label, Literal(j["predicate"], datatype=XSD.string)))
                
                g.add((subject, predicate, object))

                subject_uri = None
                object_uri = None
                predicate_uri = None
                for tup in disamb_tuples:
                    if tup[0] == j["subject"]:
                        subject_uri = URIRef(WIKI + quote(tup[1]))
                        g.add((subject, OWL.sameAs, subject_uri))
                    if tup[0] == j["object"]:
                        object_uri = URIRef(WIKI + quote(tup[1]))
                        g.add((object, OWL.sameAs, object_uri))
                    if tup[0] == j["predicate"]:
                        predicate_uri = URIRef(WIKI + quote(tup[1]))
                        g.add((predicate, OWL.sameAs, predicate_uri))

                    if subject_uri and object_uri and predicate_uri:
                        g.add((subject_uri, predicate_uri, object_uri))
                        #print(f"Added triple: {subject_uri} {predicate_uri} {object_uri}")
                                     
    ttl_file = os.path.join(args.output_folder, f"{args.experiment}_extracted_KG.ttl")
    g.serialize(destination=ttl_file, format='turtle')
    print(f"TTL file created at: {ttl_file}")
    print("Done!")
    


if __name__ == "__main__":
    create_ttl_file()
