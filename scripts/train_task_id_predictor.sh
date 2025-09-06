#!/bin/bash

# Task ID Predictor Training Script
# This script demonstrates how to train the task_id predictor for HiDe-Prompt

set -e

echo "Starting Task ID Predictor Training..."

# Set default values
DATA_TYPE=${DATA_TYPE:-"synthetic"}
MODEL_NAME=${MODEL_NAME:-"t5-small"}
PREDICTOR_TYPE=${PREDICTOR_TYPE:-"learned"}
NUM_TASKS=${NUM_TASKS:-5}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_EPOCHS=${NUM_EPOCHS:-50}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
OUTPUT_DIR=${OUTPUT_DIR:-"./task_id_predictor_output"}

# Training with synthetic data (default)
if [ "$DATA_TYPE" = "synthetic" ]; then
    echo "Training with synthetic data..."
    python train_task_id_predictor.py \
        --data_type synthetic \
        --num_tasks $NUM_TASKS \
        --model_name $MODEL_NAME \
        --predictor_type $PREDICTOR_TYPE \
        --use_original_model \
        --output_dir $OUTPUT_DIR \
        --learning_rate $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --num_epochs $NUM_EPOCHS \
        --hidden_dim 256 \
        --dropout 0.1 \
        --use_attention \
        --use_focal_loss \
        --label_smoothing 0.1 \
        --max_length 128 \
        --validation_split 0.2 \
        --log_level INFO \
        --seed 42

# Training with UIE data
elif [ "$DATA_TYPE" = "uie" ]; then
    echo "Training with UIE data..."
    if [ -z "$DATA_PATH" ] || [ -z "$TASK_MAPPING_FILE" ]; then
        echo "Error: DATA_PATH and TASK_MAPPING_FILE must be set for UIE data"
        exit 1
    fi
    
    python train_task_id_predictor.py \
        --data_type uie \
        --data_path $DATA_PATH \
        --task_mapping_file $TASK_MAPPING_FILE \
        --model_name $MODEL_NAME \
        --predictor_type $PREDICTOR_TYPE \
        --use_original_model \
        --output_dir $OUTPUT_DIR \
        --learning_rate $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --num_epochs $NUM_EPOCHS \
        --hidden_dim 256 \
        --dropout 0.1 \
        --use_attention \
        --max_length 128 \
        --validation_split 0.2 \
        --log_level INFO \
        --seed 42

# Training with benchmark data
elif [ "$DATA_TYPE" = "benchmark" ]; then
    echo "Training with benchmark data..."
    if [ -z "$DATA_PATH" ]; then
        echo "Error: DATA_PATH must be set for benchmark data"
        exit 1
    fi
    
    python train_task_id_predictor.py \
        --data_type benchmark \
        --data_path $DATA_PATH \
        --model_name $MODEL_NAME \
        --predictor_type $PREDICTOR_TYPE \
        --use_original_model \
        --output_dir $OUTPUT_DIR \
        --learning_rate $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --num_epochs $NUM_EPOCHS \
        --hidden_dim 256 \
        --dropout 0.1 \
        --use_attention \
        --max_length 128 \
        --validation_split 0.2 \
        --log_level INFO \
        --seed 42

else
    echo "Error: Unknown data type: $DATA_TYPE"
    echo "Supported data types: synthetic, uie, benchmark"
    exit 1
fi

echo "Training completed successfully!"
echo "Results saved to: $OUTPUT_DIR"

# Optional: Test the trained predictor
if [ "$TEST_PREDICTOR" = "true" ]; then
    echo "Testing trained predictor..."
    python -c "
import sys
sys.path.append('src')
from peft.tuners.task_id_predictor_trainer import TaskIDPredictorTrainer
from peft.tuners.task_id_predictor import TaskIDPredictor
import torch

# Load best model
checkpoint_path = '$OUTPUT_DIR/best_model.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Create model
model = TaskIDPredictor(input_dim=768, num_tasks=$NUM_TASKS, hidden_dim=256)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f'Loaded model with validation accuracy: {checkpoint[\"best_val_acc\"]:.4f}')
print('Model ready for inference!')
"
fi
