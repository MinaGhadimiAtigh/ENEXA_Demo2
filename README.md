
# ENEXA Pipeline

## Step 1: Download Wikipedia articles

We have two options:

1. **Only text**  
   ```bash
   python download_wiki_articles_text_only.py
   ```

2. **Text + Tables**  
   ```bash
   python download_wiki_articles_with_tables.py
   ```

Downloading Wikipedia pages is relatively quick depending on the number and volume of pages.

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
- `--max_tokens` (default: 3072)
- `--output_folder` (directory to store `args.json` and `LLM_answers.jsonl`)

**Important notes:**
- Same LLM performs both NER and RE.
- For each task (NER/RE) and target category, a new prompt is used.
- Parallel execution is possible.
- Dataset is looped:
  - 3 NER categories
  - 7 RE categories
  - 2 tasks → 3 × 7 × 2 = 42 total loops

**Performance:**
- A100 GPU node
- Job wall-clock time: `01:27:51`

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
- Job wall-clock time: `01:37:42`
- 17,652 extracted components

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
- Job wall-clock time: `01:09:50`

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
