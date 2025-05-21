# we start with imports
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDFS, RDF, OWL, XSD
from urllib.parse import quote
import json
import string
import argparse
import os


def read_jsonL(file_path):
    """This function reads a jsonl file and returns a list of dictionaries"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


def remove_punctuation(text):
    """This function removes punctuation from a text"""
    return text.translate(str.maketrans('', '', string.punctuation))


def shape_relation_name(rel_string, prefix):
    """ 
    Get the generated string as the relation name and shape it to the desired format:
    => desired format: "prefix:relationName"
    """

    rel = remove_punctuation(rel_string)
    rel = rel.strip().lower().split()
    if len(rel) > 1:
        words = []
        for index, word in enumerate(rel):
            if (index % 2) == 0:
                word = word.lower()   
            else:
                word = word.capitalize()
                    
            words.append(word)
        rel = ''.join(words)
    else:
        rel = rel[0].lower()
    
    rel = quote(rel)
    shaped_rel =  URIRef(prefix + rel)
    
    return shaped_rel


def shape_entity_name(entity_string, prefix):
    """ 
    Get the generated string as the entity name and shape it to the desired format:
    => desired format: "prefix:entity_name"
    """
    ent = remove_punctuation(entity_string)
    ent = ent.strip().lower().replace(' ', '_')
    ent = quote(ent)
    ent = URIRef(prefix + ent)
    
    return ent


def shape_class_name(type_string, prefix):
    """ 
    Get the generated string as the class name and shape it to the desired format:
    => desired format: "prefix:ClassName"
    """
    type_name = remove_punctuation(type_string)
    type_name = type_name.strip().split()
    if len(type_name) > 1:
        shaped_type = []
        for x in type_name:
            if x.isupper():
                x=x.upper()
            else:
                x = x.capitalize()
            shaped_type.append(x)
        shaped_type = ''.join(shaped_type)
        shaped_type = quote(shaped_type)
        shaped_type = URIRef(prefix + shaped_type)  
    else:
        shaped_type = URIRef(prefix + type_name[0].capitalize())
    
    return shaped_type

def check_wiki_triples(extracted_triples, disambiguated_items):

    wiki_triples = []
    for triple in extracted_triples:
        if triple["subject"] in disambiguated_items and not "None of" in triple["subject"]:
            wiki_subject = disambiguated_items[triple["subject"]]
            if not wiki_subject.startswith("Q"):
                wiki_subject = "Q" + wiki_subject
        else:
            wiki_subject = None

        if triple["object"] in disambiguated_items and not "None of" in triple["object"]:
            wiki_object = disambiguated_items[triple["object"]]
            if not wiki_object.startswith("Q"):
                wiki_object = "Q" + wiki_object
        else:
            wiki_object = None

        if triple["predicate"] in disambiguated_items and not "None of" in triple["predicate"]:
            wiki_predicate = disambiguated_items[triple["predicate"]]
            if not wiki_predicate.startswith("P"):
                wiki_predicate = "P" + wiki_predicate
        else:
            wiki_predicate = None

        wiki_triple = (wiki_subject, wiki_predicate, wiki_object)
        if not None in wiki_triple:
            wiki_triples.append(wiki_triple)
    return wiki_triples

def create_ttl_file():
    parser = argparse.ArgumentParser(description='Create a TTL file from extraction and disambiguation data')
    parser.add_argument('--extraction_file', type=str, default="ENEXA_Demo2/cand_gen_output/candidates_adidas/extraction_and_candidates.jsonl", help='The input file where the extraction results are stored')
    parser.add_argument('--disambiguation_file', type=str, default="ENEXA_Demo2/disambiguation_output/raged_output_adidas/disambiguation_output.jsonl", help='The input file where the disambuguation results are stored')
    parser.add_argument('--experiment', type=str, default="adidas", help='The name of the experiment')
    parser.add_argument('--output_folder', type=str, default="ENEXA_Demo2/graph_output", help='The output folder where the ttl file is stored')

    args = parser.parse_args()

    args_dict = vars(args)
    for key in args_dict:
        print(f"{key}: {args_dict[key]}")

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    extraction = read_jsonL(args.extraction_file)
    disambiguation = read_jsonL(args.disambiguation_file)

    merged_data = []
    for i in disambiguation:
        input_data = extraction[i["index"]]
        input_data["disambiguations"] = i["disambiguations"]
        merged_data.append(input_data)

    with open(args.output_folder + "/final_pipeline_output.jsonl", "w", encoding="utf-8") as f:
        for i in merged_data:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
    print(f"Merged data file created at: {args.output_folder}/inal_pipeline_output.jsonl")

    ENX = Namespace("http://example.org/enexa-adaptableIE/")
    WIKI = Namespace("https://www.wikidata.org/wiki/")

    g = Graph() # create a graph object
    g.bind("adaptableIE", ENX,  override=True) 
    g.bind("wiki", WIKI, override=True) 

    for i in merged_data:       
        if "entities" in i: 
            for j in i["entities"]:
                entity = shape_entity_name(j["entity"], ENX)
                g.add((entity, RDF.type, OWL.NamedIndividual))
                g.add((entity, RDFS.label, Literal(j["entity"], datatype=XSD.string)))
                if "type" in j:
                    class_name = shape_class_name(j["type"], ENX)
                    g.add((class_name, RDF.type, OWL.Class))
                    g.add((entity, RDF.type, class_name))

        if "triples" in i:
            for j in i["triples"]:
                subject = shape_entity_name(j["subject"], ENX)
                g.add((subject, RDF.type, OWL.NamedIndividual))
                g.add((subject, RDFS.label, Literal(j["subject"], datatype=XSD.string)))
                
                object = shape_entity_name(j["object"], ENX)
                g.add((object, RDFS.label, Literal(j["object"], datatype=XSD.string)))
                    
                predicate = shape_relation_name(j["predicate"], ENX)
                g.add((predicate, RDF.type, OWL.ObjectProperty))
                g.add((subject, predicate, object))

            wiki_triples = check_wiki_triples(i["triples"], i["disambiguations"])
            #print(f"Wiki triples: {wiki_triples}")
            for j in wiki_triples:
                subject = URIRef(WIKI + quote(j[0]))
                predicate = URIRef(WIKI + quote(j[1]))
                object = URIRef(WIKI + quote(j[2]))

                g.add((subject, predicate, object))
                g.add((subject, RDFS.label, Literal(j[0], datatype=XSD.string)))
                g.add((object, RDFS.label, Literal(j[2], datatype=XSD.string)))
                g.add((predicate, RDFS.label, Literal(j[1], datatype=XSD.string)))
                    
        if "disambiguations" in i:
            for j in i["disambiguations"]:
                if j["output"].startswith("P"):
                    entity = shape_relation_name(j["item"], ENX)
                else:
                    entity = shape_entity_name(j["item"], ENX)

                if not j["output"].startswith("Q") and not j["output"].startswith("P"):
                    object_string = quote(f"Q{j['output']}")
                else:
                    object_string = quote(j["output"]) 
                object_uri = URIRef(WIKI + object_string)
                
                if not j["output"].startswith("None of"):
                    g.add((entity, RDFS.label, Literal(j["item"], datatype=XSD.string)))
                    g.add((entity, OWL.sameAs, object_uri))

        
                                       
    ttl_file = os.path.join(args.output_folder, f"{args.experiment}_extracted_KG.ttl")
    g.serialize(destination=ttl_file, format='turtle')
    print(f"TTL file created at: {ttl_file}")
    print("Done!")


if __name__ == "__main__":
    create_ttl_file()
