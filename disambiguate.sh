#!/bin/bash
#SBATCH -J extract_triples
#Set job requirements
#SBATCH -N 1
#SBATCH -t 00:08:00
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
mkdir "$TMPDIR"/raged_output_one_schema

#Execute the Python program.
accelerate launch --num_processes=1 --num_machines=1 $TMPDIR/ENEXA_Demo2/link_items.py \
                            --input_file $TMPDIR/ENEXA_Demo2/candidates_adidas_one_schema/extraction_and_candidates.jsonl \
                            --prompt_template $TMPDIR/ENEXA_Demo2/input_files/entity_disambiguation_template_can_reject.json \
                            --LLM "FinaPolat/unsloth_llama3_8B_for_ED" \
                            --output_folder $TMPDIR/raged_output_one_schema \
 
#Copy output directory from scratch to home
cp -r "$TMPDIR"/raged_output_one_schema $HOME/ENEXA_Demo2/disambiguation_output
