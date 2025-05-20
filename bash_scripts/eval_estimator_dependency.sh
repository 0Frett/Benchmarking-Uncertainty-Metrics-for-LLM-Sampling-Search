
models=(
    'meta-llama/Meta-Llama-3-8B-Instruct'
    'google/gemma-2-9b-it'
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

        etree_dir="labeled_estimator_trees/$dataset/$model_name/b1q32"
        save_dir="eval_result/estimator_dependency/$dataset/$model_name"

        PYTHONPATH=lib python3 bin/eval_estimator_dependency.py \
            --etree_dir "$etree_dir" \
            --save_dir "$save_dir" \
            --n_bootstrap 500
    done
done
