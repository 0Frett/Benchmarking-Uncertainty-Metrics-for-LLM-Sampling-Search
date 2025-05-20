models=(
    'google/gemma-2-9b-it'
    'meta-llama/Meta-Llama-3-8B-Instruct'
    # 'meta-llama/Llama-3.2-1B-Instruct'
    'meta-llama/Llama-3.2-3B-Instruct'
    'google/gemma-2-2b-it'
    'mistralai/Mistral-7B-Instruct-v0.3'
)

datasets=(
    'truthfulQA'
    'commonsenseQA'
    'gsm8k'
    'triviaQA'
)

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        model_name=$(basename "$model")
        data_path="data/${dataset}.json"

        PYTHONPATH=lib python3 bin/gen_estimator_tree.py \
            --model "$model" \
            --data_path "$data_path" \
            --dataset "$dataset" \
            --branch_num 8 \
            --sampling_num 16 \
            --max_tokens 2048 \
            --temperature 1.0 \
            --tensor_parallel_size 1 \
            --gpu_memory_utilization 0.9
    done
done
