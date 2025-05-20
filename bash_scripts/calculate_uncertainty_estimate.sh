

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

models=(
    'meta-llama/Meta-Llama-3-8B-Instruct'
    'google/gemma-2-9b-it'
    'google/gemma-2-2b-it'
    'meta-llama/Llama-3.2-1B-Instruct'
    'meta-llama/Llama-3.2-3B-Instruct'
    'mistralai/Mistral-7B-Instruct-v0.3'
)

datasets=(
    'commonsenseQA'
    'math'
    'triviaQA'
    'truthfulQA'
)

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        model_name=$(basename "$model")

        # Set the paths for input and output
        et_dir="labeled_estimator_trees/$dataset/$model_name/b1q32"
        save_dir="labeled_estimator_trees/$dataset/$model_name/b1q32"

        PYTHONPATH=lib python3 bin/calculate_estimator_value.py \
            --etree_dir "$et_dir" \
            --save_dir "$save_dir" \
            --model "$model" \
            --dataset "$dataset" \
            --tensor_parallel_size 1 \
            --gpu_memory_utilization 0.95 \
            --max_tokens 2048 \
            --temperature 1.0
    done
done
