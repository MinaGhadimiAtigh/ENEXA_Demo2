#!/bin/bash
#SBATCH -J test_GPT
#Set job requirements
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=f.yilmazpolat@uva.nl

#Loading modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

#Copy data to scratch
cp -r $HOME/ENEXA_Demo2 "$TMPDIR"
pwd

#Create output directory on scratch
mkdir "$TMPDIR"/candidates_adidas

#Execute the Python program.
python $TMPDIR/ENEXA_Demo2/generate_Wikidata_candidates_for_linking.py \
            --input_file $TMPDIR/ENEXA_Demo2/IE_extraction_output/extraction_one_page/LLM_answers.jsonl \
            --output_folder $TMPDIR/candidates_adidas \
            --num_candidates 8 \

#Copy output directory from scratch to home
cp -r "$TMPDIR"/candidates_adidas $HOME/ENEXA_Demo2