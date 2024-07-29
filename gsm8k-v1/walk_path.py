import pickle
import random

def largest_walk(node):
    if node is None:
        return []
    if not node.children:
        return [node.correct]
    largest_child = max(
        node.children, 
        key=lambda x: (x.top2disparity + x.npe + x.lnpe) / 3
    )
    
    return [node.correct] + largest_walk(largest_child)

def smallest_walk(node):
    if node is None:
        return []
    if not node.children:
        return [node.correct]
    smallest_child = min(
        node.children, 
        key=lambda x: (x.top2disparity + x.npe + x.lnpe) / 3
    )
    
    return [node.correct] + smallest_walk(smallest_child)

def random_walk(node):
    if node is None:
        return []
    if not node.children:
        return [node.correct]
    random_child = random.choice(node.children)
    
    return [node.correct] + random_walk(random_child)


def small_large_walk(node, select_largest_first=True):
    if node is None:
        return []
    if not node.children:
        return [node.correct]
    if select_largest_first:
        selected_child = max(node.children, key=lambda x: (x.top2disparity + x.npe + x.lnpe) / 3)
    else:
        selected_child = min(node.children, key=lambda x: (x.top2disparity + x.npe + x.lnpe) / 3)

    return [node.correct] + small_large_walk(selected_child, select_largest_first=False)

def load_tree(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    # exp_stats = {}
    smallwalk_success = 0
    randomwalk_success = 0
    largewalk_success = 0
    small_large_success = 0
    total_run = 0
    for idx in [8, 39, 74, 107, 119, 141, 144, 154, 161, 177, 211, 214, 218]:#,
        print(f"============ Q{idx} ===============")
        #exp_stats[idx] = {"smallest_walk_success":}
        for tree in [1, 2]:
            filename = f'./output_trees/Q{idx}/tree_{tree}/tree.pkl'
            root = load_tree(filename=filename)
            smallest_path = smallest_walk(root)
            print("smallest_path :", smallest_path)
            if smallest_path[-1]:
                smallwalk_success += 1
    
            random_path = random_walk(root)
            print("random_path :", random_path)
            if random_path[-1]:
                randomwalk_success += 1

            largest_path = largest_walk(root)
            print("largest_path :", largest_path)
            if largest_path[-1]:
                largewalk_success += 1
            
            small_large_path = small_large_walk(root)
            print("small_large_path :", small_large_path)
            if small_large_path[-1]:
                small_large_success += 1

            total_run += 1

    print("small", smallwalk_success/total_run)
    print("large", largewalk_success/total_run)
    print("small-large", small_large_success/total_run)
    print("random", randomwalk_success/total_run)
