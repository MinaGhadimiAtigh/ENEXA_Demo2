# ENEXA Pipeline
## Step 1:
    ### Downdoad Wikipedia articles. 
    We have two options. 
        1. Only text
    ```python download_wiki_articles_text_only.py```
        2. Text + Tables
    ```python download_wiki_articles_with_tables.py```
    
    Downloading Wikipedia pages is relatively quick depending on the number and volume of the downloaded pages.

    This script requires:
        --url_file (json file, a list of Wikipedia  URLs)
        --output_dir (path to a directory, downloaded Wikipedia page content is saved here)

## Step 2:
    ### Extract information from downloaded Wikipedia pages.
    ```python semiclosedIE_with_example.py```
    Run on a GPU node in a cluster:
    ```sbatch extract_triples_semiclosed_with_example.sh```

    This script does NER and RE using an LLM. Requires:
        --input_folder (path to the directory where our downloaded Wikipedia pages are stored)
        --prompt_template (path to a custom prompt template. Formatter requires the following input variables: task, example, schema, inputs, output_format)
        --target_entity_types (path to a json file: list of dicts with the following keys: "category" ->str, "types"=>list of str)
        --target_relations (path to a json file: list of dicts with the following keys: "category" ->str, "relations"=>list of str)
        --NER_example (path to a json file where the NER execution example for LLM is stored. Used to fill prompt template - example variable)
        --RE_example (path to a json file where the RE execution example for LLM is stored. Used to fill prompt template - example variable)
        --LLM ("FinaPolat/phi4_adaptable_IE", finetuned version of phi-4 for NER and RE. Data: WebNLG from https://github.com/cenguix/Text2KGBench/tree/main, Method: QLoRA)
        --temperature (default=0.001, text generartion parameter)
        --max_tokens (default=3072, text generartion parameter)
        --output_folder (path to a folder to store the extraction parameters and resulting output: 2 files - args.json, LLM_answers.jsonl)
    
    Important points about the script:
        * The same LLM performs NER and RE. 
        * For each task (NER or RE) and targets (Entity type or Relation categoties) a new prompt is compiled. We run inference for each prompt. If we want to parallelize the execution of the script, the extraction tasks for NER and RE can easily be separated.
        * We loop through the dataset for each NER target types. We use three categories for this experiment.
        * We loop through the dataset for each RE target relation categories. For this experiement, we use seven. 
        * 2 (task) x 3 (NER categories) x 7 (RE categories ) = 42 (We loop through the dataset 42 times.) 
        * A100 GPU node, Job Wall-clock time: 01:27:51

## Step 3:
    ### Generate candidates for each extracted triple components using Wikidata API.
    ```python generate_Wikidata_candidates_for_linking.py```

    This script loops over the extraction file and generates entity/property linking candidates for each extracted entity and relation using Wikidata. Requires:
        --input_file (path to the extraction output which was generated in Step 2. )
        --num_candidates (int, the number of candidates per extracted entity/property)
        --output_folder (path to a folder to store candidates)

    *CPU node, Job Wall-clock time: 01:37:42
    * 17652 extracted component (entity, entity type or property)
        

## Step 4:
    ### Linking: Select the correct Wikidata entity/property ID using our disambiguation model for each extracted triple component. 
    ```python link_items.py```
    Run on a GPU node in a cluster:
    ```sbatch disambiguate.sh```
    
    This script uses an LLM to link extracted triple components to Wikidata. Given a mention, context, and candidates, the LLM responds with a selected ID or "None of the candidates" for each extracted entity/property.  Requires:
        --input_file (path to a file where generated candidates in Step 3 is stored)
        --prompt_template (path to a custom prompt template. Formatter requires the following input variables: text, mention, candidates)
        --LLM ("FinaPolat/unsloth_llama3_8B_for_ED", finetuned version of llama3-8B for Entity Disambiguation (ED). Data: 5000 random instanced from ZELDA training set, https://github.com/flairNLP/zelda, Method: QLoRA)
        --temperature (default=0.001, text generartion parameter)
        --max_tokens (default=5, text generartion parameter)
        --output_folder (path to a folder to store disambiguation output)
    * A100 GPU node, Job Wall-clock time: 01:09:50

## Step 5: 
    ### Create a ttl file with the resulting extraction. 
    ```python make_ttl_file.py```

    This script doesn't take long time. Requires:
        --extraction_file (The out file from Step 3)
        --disambiguation_file (The out file from Step 4)
        --experiment (experiment name to use in the ttl file path)
        --output_folder (a folder the to save the ttl file)
    



