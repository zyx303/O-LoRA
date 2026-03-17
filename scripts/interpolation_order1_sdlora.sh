#!/bin/bash
set -euo pipefail
set -x

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data/yongxi/.cache/huggingface}"
CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}

ADAPTER_A="${ADAPTER_A:-/home/yongxi/work/O-LoRA/exp/sdlora/order_1_test/outputs/2-amazon/adapter}"
ADAPTER_B="${ADAPTER_B:-/home/yongxi/work/O-LoRA/exp/sdlora/order_1_test/outputs/4-agnews/adapter}"
RESULTS_ROOT="${RESULTS_ROOT:-exp/t5_interpolation/order1_sdlora_amazon_to_agnews}"
DATA_DIR="${DATA_DIR:-CL_Benchmark}"
TASK_CONFIG_DIR="${TASK_CONFIG_DIR:-configs/order1_configs/agnews}"
INSTRUCTION_FILE="${INSTRUCTION_FILE:-configs/instruction_config.json}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/ds_configs/stage2.config}"
ALPHAS="${ALPHAS:-0.0 0.25 0.5 0.75 1.0}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-interp_t5_order1_sdlora}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-128}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
PEFT_TYPE="${PEFT_TYPE:-SDLORA}"

require_adapter_dir() {
  local adapter_dir="$1"
  if [[ ! -f "$adapter_dir/adapter_config.json" || ! -f "$adapter_dir/adapter_model.bin" ]]; then
    echo "Missing adapter checkpoint under $adapter_dir" >&2
    echo "Current workspace does not contain the expected SDLora adapter files for logs_and_outputs/sdlora/order_1." >&2
    exit 1
  fi
}

require_adapter_dir "$ADAPTER_A"
require_adapter_dir "$ADAPTER_B"

port=$(shuf -i25000-30000 -n1)
mkdir -p "$RESULTS_ROOT/adapters" "$RESULTS_ROOT/evals"

extra_args=()
if [[ -n "${MAX_PREDICT_SAMPLES:-}" ]]; then
  extra_args+=(--max_predict_samples "$MAX_PREDICT_SAMPLES")
fi

for alpha in $ALPHAS; do
  adapter_out="$RESULTS_ROOT/adapters/alpha_${alpha}"
  if [[ ! -f "$adapter_out/adapter_model.bin" || ! -f "$adapter_out/adapter_config.json" ]]; then
    python3 scripts/interpolate_adapters.py \
      --adapter-a "$ADAPTER_A" \
      --adapter-b "$ADAPTER_B" \
      --alpha "$alpha" \
      --output-dir "$adapter_out"
  fi

  eval_out="$RESULTS_ROOT/evals/alpha_${alpha}/4-agnews"
  if [[ -z "${FORCE_RERUN:-}" && -f "$eval_out/predict_results.json" ]]; then
    continue
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}" deepspeed --master_port "$port" src/run_uie_lora.py \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path "$adapter_out" \
    --data_dir "$DATA_DIR" \
    --task_config_dir "$TASK_CONFIG_DIR" \
    --instruction_file "$INSTRUCTION_FILE" \
    --instruction_strategy single \
    --output_dir "$eval_out" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate 1e-03 \
    --num_train_epochs 1 \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --run_name "${RUN_NAME_PREFIX}_alpha_${alpha}" \
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
    --skip_predict_loss True \
    --peft_type "$PEFT_TYPE" \
    "${extra_args[@]}"
done

python3 scripts/summarize_interpolation_results.py \
  --results-root "$RESULTS_ROOT" \
  --output "$RESULTS_ROOT/summary.csv"
