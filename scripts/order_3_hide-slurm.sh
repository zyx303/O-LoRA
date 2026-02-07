#!/bin/bash
#SBATCH -N 1
#SBATCH --mem=200G
#SBATCH -p gpu_requeue
#SBATCH --constraint="a100"
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yichenwu/zyx/.cache/huggingface

cd /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yichenwu/zyx/O-LoRA


source /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yichenwu/zyx/bashrc

#nvidia-smi
srun nvidia-smi

# Activate conda environment (Python 3)
eval "$(conda shell.bash hook)"
conda activate /n/home02/ycwu/.conda/envs/lora

# Load GCC 12 for building DeepSpeed extensions
module load gcc/12.2.0-fasrc01

port=$(shuf -i25000-30000 -n1)
# export debug=1
#@@@@@@@#### bash scripts/order_3_hide.sh> logs_and_outputs/hide_cr/order_3/logs/train_and_infer.log 2>&1 &

deepspeed --master_port $port src/run_uie_hide.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path initial_model/t5-large \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order3_configs/yahoo \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/hide_cr/order_3/outputs/1-yahoo \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-02 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order3_round1 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type cosine \
   --warmup_steps 20 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.5 \
   --lamda_2 0 \
   --num_virtual_tokens 20 \
   --peft_type hide_prompt

sleep 5

deepspeed --master_port $port src/run_uie_hide.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/hide_cr/order_3/outputs/1-yahoo/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order3_configs/amazon \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/hide_cr/order_3/outputs/2-amazon \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-2 \
   --num_train_epochs 2 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order3_round2 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type cosine \
   --warmup_steps 20 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.5 \
   --lamda_2 0 \
   --num_virtual_tokens 20 \
   --peft_type hide_prompt

sleep 5

deepspeed --master_port $port src/run_uie_hide.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/hide_cr/order_3/outputs/2-amazon/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order3_configs/agnews \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/hide_cr/order_3/outputs/3-agnews \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-2 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order3_round3 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type cosine \
   --warmup_steps 20 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.5 \
   --lamda_2 0 \
   --num_virtual_tokens 20 \
   --peft_type hide_prompt

sleep 5

deepspeed --master_port $port src/run_uie_hide.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs/hide_cr/order_3/outputs/3-agnews/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order3_configs/dbpedia \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/hide_cr/order_3/outputs/4-dbpedia \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-2 \
   --num_train_epochs 2 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name order3_round4 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type cosine \
   --warmup_steps 20 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.5 \
   --lamda_2 0 \
   --num_virtual_tokens 20 \
   --peft_type hide_prompt 

