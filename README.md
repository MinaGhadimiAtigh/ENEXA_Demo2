
# ENEXA Extractions Pipeline

## Step 1: Download Wikipedia articles
   ```bash
   python download_wiki_articles_text_only.py
   ```
Downloading Wikipedia page is quick.

**This script requires:**
- `--url_file` (JSON file with a list of Wikipedia URLs)
- `--output_dir` (directory to save the downloaded content)

---

## Step 2: Extract information from downloaded Wikipedia pages

Run the script:

```bash
python semiclosedIE_with_example.py
```

On a GPU node in a cluster:

```bash
sbatch extract_triples_semiclosed_with_example.sh
```

**This script does NER and RE using an LLM. It requires:**
- `--input_folder` (path to downloaded Wikipedia pages)
- `--prompt_template` (custom prompt template file)
- `--target_entity_types` (JSON: list of dicts with "category" and "types")
- `--target_relations` (JSON: list of dicts with "category" and "relations")
- `--NER_example` (JSON example for NER)
- `--RE_example` (JSON example for RE)
- `--LLM` ("FinaPolat/phi4_adaptable_IE")
- `--temperature` (default: 0.001)
- `--max_tokens` (default: 8192)
- `--output_folder` (directory to store `args.json` and `LLM_answers.jsonl`)

**Important notes:**
- Same LLM performs both NER and RE.
- For each task (NER/RE) and target category, a new prompt is used.
- Parallel execution is possible.
- Dataset is looped:
  - 1 NER category
  - 1 RE category
  - 2 tasks → 1 × 1 × 2 = 2 times

**Performance:**
- A100 GPU node
- Job wall-clock time: `00:28:49`

---

## Step 3: Generate Wikidata candidates for extracted triples

```bash
python generate_Wikidata_candidates_for_linking.py
```

**Requires:**
- `--input_file` (output from Step 2)
- `--num_candidates` (int) 
- `--output_folder`

**Performance:**
- CPU node
- Job wall-clock time: `00:08:48`
- 1167 extracted components (entities + properties)
- 5 candidates per item

---

## Step 4: Link triple components to Wikidata IDs

```bash
python link_items.py
```

On a GPU node in a cluster:

```bash
sbatch disambiguate.sh
```

**Uses an LLM to select the correct Wikidata ID for each entity/property.**

**Requires:**
- `--input_file` (from Step 3)
- `--prompt_template`
- `--LLM` ("FinaPolat/unsloth_llama3_8B_for_ED")
- `--temperature` (default: 0.001)
- `--max_tokens` (default: 5)
- `--output_folder`

**Performance:**
- A100 GPU node
- Job wall-clock time: `00:06:18`

---

## Step 5: Generate TTL file from results

```bash
python make_ttl_file.py
```

**Requires:**
- `--extraction_file` (output from Step 3)
- `--disambiguation_file` (output from Step 4)
- `--experiment` (experiment name for file naming)
- `--output_folder`
