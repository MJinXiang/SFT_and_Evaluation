# ===================== 参数读取 ============================
MODEL_PATH="/mnt/usercache/huggingface/Qwen2.5-3B-Instruct"   # 模型路径，比如 /mnt/usercache/huggingface/Qwen2.5-3B-Instruct
TASK_NAME="fetaqa"  # 任务名，比如 tatqa
TRAIN_TYPE="base"  # 训练方式，比如 grpo、ppo、sft
MODEL_SIZE="3b"    # 模型大小，比如 3b、7b
TENSOR_PARALLEL_SIZE=2  # 张量并行大小，为了与注意力头数量匹配
BATCH_SIZE=128      # 批处理大小
MAX_TOKENS=4096     # 模型生成的最大token数

# ===================== 路径配置 ============================
BASE_PATH="$(pwd)"  # 使用当前目录作为基础路径，如：/mnt/usercache/mengjinxiang/Project/SFT_and_Evaluation
INFER_SCRIPT="${BASE_PATH}/tests/${TASK_NAME}.py"
EVAL_SCRIPT="${BASE_PATH}/tests/eval/${TASK_NAME}_eval.py"

# 自动生成输出文件路径
PRED_OUTPUT="${BASE_PATH}/results/${TASK_NAME}/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}.json"
INFER_LOG="${BASE_PATH}/results/${TASK_NAME}/logs/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}_infer.log"
EVAL_OUTPUT="${BASE_PATH}/results/${TASK_NAME}/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}_eval_results.json"

# 确保输出目录存在
mkdir -p "$(dirname "$PRED_OUTPUT")"
mkdir -p "$(dirname "$INFER_LOG")"

# ===================== 运行推理 ============================
echo "Running inference with VLLM..."
python "$INFER_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --output_file "$PRED_OUTPUT" \
    --log_file "$INFER_LOG" \
    --base_path "$BASE_PATH" \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --batch_size $BATCH_SIZE \
    --max_tokens $MAX_TOKENS \
    --temperature 0.0

if [ $? -ne 0 ]; then
  echo "Inference failed"
  exit 1
fi

# ===================== 运行评估 ============================
echo "Running evaluation..."
python "$EVAL_SCRIPT" \
    --results_file "$PRED_OUTPUT" \
    --output_file "$EVAL_OUTPUT" \
    --base_path "$BASE_PATH" 

if [ $? -ne 0 ]; then
  echo "Evaluation failed"
  exit 1
fi

echo "Testing and evaluation completed successfully!"