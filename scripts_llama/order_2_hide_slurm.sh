#!/bin/bash
#SBATCH -N 1
#SBATCH --mem=200G
#SBATCH -p gpu_requeue
#SBATCH --constraint="a100"
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --output=order_2_hide-%j.out
#SBATCH --error=order_2_hide-%j.err
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yichenwu/zyx/.cache/huggingface

cd /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yichenwu/zyx/O-LoRA

source /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yichenwu/zyx/bashrc

# Activate conda environment (Python 3)
eval "$(conda shell.bash hook)"
conda activate /n/home02/ycwu/.conda/envs/lora

# Load GCC 12 for building DeepSpeed extensions
module load gcc/12.2.0-fasrc01

port=$(shuf -i25000-30000 -n1)

#############!!!!!!!! bash scripts_llama/order_2_hide.sh> logs_and_outputs_llama/hide/order_2/logs/train_and_infer.log 2>&1 &

# Checkpoint file for resuming
CHECKPOINT_FILE="logs_and_outputs_llama/hide/order_2/.checkpoint"
mkdir -p logs_and_outputs_llama/hide/order_2

# Function to check if a task is completed
is_task_completed() {
    local task_name=$1
    local output_dir=$2
    # Check if adapter directory exists and contains files
    if [ -d "${output_dir}/adapter" ] && [ "$(ls -A ${output_dir}/adapter 2>/dev/null)" ]; then
        echo "Task ${task_name} already completed, skipping..."
        return 0
    fi
    return 1
}

# Function to mark task as completed
mark_task_completed() {
    local task_name=$1
    echo "${task_name}" >> "${CHECKPOINT_FILE}"
    echo "Task ${task_name} marked as completed"
}

# Task 1: dbpedia
if ! is_task_completed "1-dbpedia" "logs_and_outputs_llama/hide/order_2/outputs/1-dbpedia"; then
    echo "Starting Task 1: dbpedia"
    deepspeed --master_port $port src/run_uie_hide.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path initial_model/llama2-7b-hf \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order2_configs/dbpedia \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs_llama/hide/order_2/outputs/1-dbpedia \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name order2_round1 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.5 \
   --lamda_2 0 \
   --num_virtual_tokens 20 \
   --peft_type hide_prompt
    mark_task_completed "1-dbpedia"
    sleep 5
else
    echo "Skipping Task 1: dbpedia (already completed)"
fi

# Task 2: amazon
if ! is_task_completed "2-amazon" "logs_and_outputs_llama/hide/order_2/outputs/2-amazon"; then
    echo "Starting Task 2: amazon"
    deepspeed --master_port $port src/run_uie_hide.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs_llama/hide/order_2/outputs/1-dbpedia/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order2_configs/amazon \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs_llama/hide/order_2/outputs/2-amazon \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-04 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name order2_round2 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.5 \
   --lamda_2 0 \
   --num_virtual_tokens 20 \
   --peft_type hide_prompt
    mark_task_completed "2-amazon"
    sleep 5
else
    echo "Skipping Task 2: amazon (already completed)"
fi

# Task 3: agnews
if ! is_task_completed "3-agnews" "logs_and_outputs_llama/hide/order_2/outputs/3-agnews"; then
    echo "Starting Task 3: agnews"
    deepspeed --master_port $port src/run_uie_hide.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs_llama/hide/order_2/outputs/2-amazon/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order2_configs/agnews \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs_llama/hide/order_2/outputs/3-agnews \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-04 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name order2_round3 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.5 \
   --lamda_2 0 \
   --num_virtual_tokens 20 \
   --peft_type hide_prompt
    mark_task_completed "3-agnews"
    sleep 5
else
    echo "Skipping Task 3: agnews (already completed)"
fi

# Task 4: yahoo
if ! is_task_completed "4-yahoo" "logs_and_outputs_llama/hide/order_2/outputs/4-yahoo"; then
    echo "Starting Task 4: yahoo"
    deepspeed --master_port $port src/run_uie_hide.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs_llama/hide/order_2/outputs/3-agnews/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order2_configs/yahoo \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs_llama/hide/order_2/outputs/4-yahoo \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-04 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name order2_round4 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.5 \
   --lamda_2 0 \
   --num_virtual_tokens 20 \
   --peft_type hide_prompt
    mark_task_completed "4-yahoo"
else
    echo "Skipping Task 4: yahoo (already completed)"
fi

echo "All tasks completed for order_2_hide!"

