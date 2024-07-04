from llms import *
from truth import *
from model import *
from world import *
import json
from datasets import load_dataset
import copy
import pickle
from collections import Counter


def select_difficult_question(dataset, topk_hard):
    """select question needing more reasoning step"""
    answer = dataset['answer']
    steps = []
    for a in answer:
        step = len(a.split("\n"))
        steps.append(step)
    count = Counter(steps)
    select_difficulties = list(dict(sorted(count.items())).keys())[-1*topk_hard:]
    indices = [i for i, x in enumerate(steps) if x in select_difficulties]

    return indices

class gsm8k_Inference():
    def __init__(self, tree_algo, prompt_pool=None):
        self.input_processor = lambda x: x["question"]
        self.get_true_answer = lambda x: x["answer"]
        self.full_dataset = load_dataset("openai/gsm8k", 'main', split='test')
        self.tree_algo = tree_algo
        self.prompt_pool = prompt_pool

    def sample_prompt(self, num_shot=4):
        ret = copy.deepcopy(self.prompt_pool)
        ret['interactive_examples'], ret['useful_examples'] = zip(*random.sample(
                                                                        list(zip(ret['interactive_examples'],ret['useful_examples'])),
                                                                        k=num_shot
                                                                    )
                                                                )
        return ret
    
    def inference(self, num_shot=4, start_idx=0, end_idx=20,):

        self.dataset = list(self.full_dataset)[start_idx:end_idx]

        for i, example in enumerate(tqdm(self.dataset,
                                         total=start_idx + len(self.dataset),
                                         initial=start_idx,
                                         desc=f"gsm8k_tree{start_idx}~{end_idx}")):

            ex = self.input_processor(example)
            true_answer = utils.retrieve_true_answer(self.get_true_answer(example))
            pr_ex = self.sample_prompt(num_shot=num_shot)
            self.tree_algo.model.update_example(example=ex, prompt=pr_ex)
            self.tree_algo.world.update_example(example=ex, prompt=pr_ex)
            tree_root = self.tree_algo(true_answer=true_answer)
            
            # save tree
            os.makedirs('./output_trees', exist_ok=True)
            save_dir = f'./output_trees/Q{start_idx+i}'
            os.makedirs(save_dir, exist_ok=True)
            save_dir += f'/tree_{len(os.listdir(save_dir))+1}-temp2.0'
            os.makedirs(save_dir, exist_ok=True)
            filename = f'{save_dir}/tree.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(tree_root, f)


if __name__ == '__main__':
    # args
    llm_name = 'gpt-3.5-turbo'
    node_width = 10
    tree_depth = 7
    temperature = 2.0
    print(f'node_width:{node_width}\ntree_depth:{tree_depth}\ntemperature:{temperature}')
    # start_idx = 0
    # end_idx = 1
    q_idx = [8] #, 177, 211, 214, 218  8, 39, 74, 107, 119, 141, 144, 154, 161, 177, 211, 214, 218
    with open('./prompts/prompt_pool.json') as f:
        prompt_pool = json.load(f)

    llm = OpenAIModel(model=llm_name)
    model = GSM8kModel(language_model=llm,
                    n_actions=node_width,
                    temperature=temperature,
                    depth_limit=tree_depth,
                    terminate_on_depth_limit=True)
    world = GSM8kWorld(language_model=llm, temperature=temperature)
    tree_algo = TruthTree(tree_depth=tree_depth, temperature=temperature, model=model, world=world)
    for idx in q_idx:
        start_idx = idx
        end_idx = idx + 1
        gsm8k_Inference(tree_algo=tree_algo, prompt_pool=prompt_pool).inference(start_idx=start_idx, end_idx=end_idx)



