#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="nlg_uncertainty"
``


run_id=`python -c "import wandb; run_id = wandb.util.generate_id(); wandb.init(project='nlg_uncertainty', id=run_id); print(run_id)"`
pip install bitsandbytes
res1=$(date +%s.%N)
model='mistral-7b'
srun python semantic_uncertainty/code/generate.py --num_generations_per_prompt='5' --model=$model --run_id=$run_id --temperature='0.5' --num_beams='1' --top_p='1.0' --add_context=False; srun python semantic_uncertainty/code/clean_generated_strings.py  --generation_model=$model --run_id=$run_id; srun python semantic_uncertainty/code/get_semantic_similarities.py --generation_model=$model --run_id=$run_id; python semantic_uncertainty/code/get_likelihoods.py --evaluation_model=$model --generation_model=$model --run_id=$run_id; srun semantic_uncertainty/code/python get_prompting_based_uncertainty.py --run_id_for_few_shot_prompt=$run_id --run_id_for_evaluation=$run_id; python semantic_uncertainty/code/compute_confidence_measure.py --generation_model=$model --evaluation_model=$model --run_id=$run_id
res2=$(date +%s.%N)
echo "Start time: $res1"
echo "Stop time:  $res2"
printf "Elapsed:    %.3F\n"  $(echo "$res2 - $res1"|bc )

