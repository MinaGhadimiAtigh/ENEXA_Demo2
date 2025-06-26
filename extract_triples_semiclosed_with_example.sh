#!/bin/bash
#SBATCH -J extract_triples
#Set job requirements
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -p gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=f.yilmazpolat@uva.nl

 
#Loading modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
 
#Copy data to scratch
cp -r $HOME/ENEXA_Demo2 "$TMPDIR"
pwd

#Create output directory on scratch
mkdir "$TMPDIR"/extraction_one_schema_per_task

#Execute the Python program.
accelerate launch --num_processes=1 --num_machines=1 $TMPDIR/ENEXA_Demo2/semiclosedIE_with_example.py \
                            --input_folder $TMPDIR/ENEXA_Demo2/wiki_downloads/wiki_downloads_no_tables \
                            --prompt_template $TMPDIR/ENEXA_Demo2/input_files/prompt_template_with_example.json \
                            --target_entity_types $TMPDIR/ENEXA_Demo2/input_files/target_entity_types_4_adidas.json \
                            --target_relations $TMPDIR/ENEXA_Demo2/input_files/target_relations_4_adidas.json \
                            --NER_example $TMPDIR/ENEXA_Demo2/input_files/NER_example.json \
                            --RE_example $TMPDIR/ENEXA_Demo2/input_files/RE_example.json \
                            --LLM FinaPolat/phi4_adaptable_IE \
                            --output_folder $TMPDIR/extraction_one_schema_per_task
 
#Copy output directory from scratch to home
cp -r "$TMPDIR"/extraction_one_schema_per_task $HOME/ENEXA_Demo2/IE_extraction_output
