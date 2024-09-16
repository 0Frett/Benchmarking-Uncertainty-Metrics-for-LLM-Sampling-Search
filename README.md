# Benchmarking Large Language Model Uncertainty for Prompt Optimization
Prompt optimization algorithms for Large Language Models (LLMs) excel in multi-step reasoning but still lack effective uncertainty estimation. This paper introduces a benchmark dataset to evaluate uncertainty metrics, focusing on Answer, Correctness, Aleatoric, and Epistemic Uncertainty. Through analysis of models like GPT-3.5-Turbo and Meta-Llama-3.1-8B-Instruct, we show that current metrics align more with Answer Uncertainty,
which reflects output confidence and diversity,rather than Correctness Uncertainty, highlighting the need for improved metrics that are optimization-objective-aware to better guide prompt optimization.
![Framework](./display_imgs/workflow.png)
*Figure 1: A reliable uncertainty quantification metric targeting correctness of binary classification problem exhibits 50\% accuracy when uncertainty is at its highest. On the contrary, most existing uncertainty quntification metrics are designed to capture the confidence (diversity) of responses fails to be sufficient for prompt optimization tasks.*
![I](./display_imgs/gpt-3.5-turbo_au-cu_scatter.png)
*Figure 2: The benchmark dataset aims at evaluate the uncertainty of prompt optimization for multi-step rea-
soning. The construction workflow is consist of three steps in every level. 
1. Random perturb the input question 
2. Random sample model output with temperature 
3. Calculate uncertainty based on different measurements*

## version info
#### |-> v0(single-step + question perturbation + answer sampling / metric: NPE,LNPE,Top-DISP,Intra)
#### |-> v1(multi-step + answer sampling / metric: NPE,LNPE,Top-DISP)
#### |-> v2(multi-step + question perturbation + answer sampling / metric: NPE,LNPE,Top-DISP)
### for every task folder
#### |-> run inference.py to generate whole tree
#### |-> run calculate_uncertainty.py to calculate uncertainty metrics of each tree nodes
#### |-> run eval_uncertainty.py to get dataframe of uncertainty metrics and ground truth uncertainty
### Additional 
#### |-> get hard questions using hard_qs.py (select by reasoning/decomposition steps)

## References
The structure and certain functions in this project were inspired by [llm-reasoning](https://github.com/maitrix-org/llm-reasoners), created by [Ber666].
