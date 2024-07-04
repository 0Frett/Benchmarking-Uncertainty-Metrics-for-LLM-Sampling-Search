import graphviz
import pickle
import numpy as np
from collections import deque

def visualize_metrics(root_node, filename):
    def draw_tree(node, dot=None):
        if dot is None:
            dot = graphviz.Digraph()
        # Define condition for coloring nodes
        color = 'black'
        if len(node.children) == 0:  # leaf nodes
            if node.correct:
                color = 'red' # color correct leaf
            label = f"""<
                <table border="0" cellborder="1" cellspacing="0">
                    <tr><td><font color="black">npe:</font></td><td><font color="black">{node.npe:.2f}</font></td></tr>
                    <tr><td><font color="black">lnpe:</font></td><td><font color="black">{node.lnpe:.2f}</font></td></tr>
                    <tr><td><font color="black">disparity:</font></td><td><font color="black">{node.top2disparity:.2f}</font></td></tr>
                    <tr><td><font color="black">answer:</font></td><td><font color="black">{node.answer}</font></td></tr>
                </table>
            >"""
        else:
            label = f"""<
                <table border="0" cellborder="1" cellspacing="0">
                    <tr><td><font color="black">npe:</font></td><td><font color="black">{node.npe:.2f}</font></td></tr>
                    <tr><td><font color="black">lnpe:</font></td><td><font color="black">{node.lnpe:.2f}</font></td></tr>
                    <tr><td><font color="black">disparity:</font></td><td><font color="black">{node.top2disparity:.2f}</font></td></tr>
                    <tr><td><font color="black">entropy:</font></td><td><font color="black">{node.entropy:.2f}</font></td></tr>
                </table>
            >"""
        dot.node(str(id(node)), label, color=color)
        for child in node.children:
            dot.edge(str(id(node)), str(id(child)))
            draw_tree(child, dot)

        return dot
    
    def get_levels(root):
        if not root:
            return []

        levels = []
        queue = deque([(root, 0)])  # (node, level)

        while queue:
            node, level = queue.popleft()
            
            if level == len(levels):
                levels.append([])
            
            levels[level].append(node)
            
            for child in node.children:
                queue.append((child, level + 1))

        return levels
    
    def get_largest_two_indices(lst):
        if len(lst) < 2:
            raise ValueError("List must contain at least two elements.")
        arr = np.array(lst)
        first_max_idx = np.argmax(arr)
        arr[first_max_idx] = -np.inf
        second_max_idx = np.argmax(arr)

        return [first_max_idx, second_max_idx]

    def get_smallest_two_indices(lst):
        if len(lst) < 2:
            raise ValueError("List must contain at least two elements.")
        arr = np.array(lst)
        first_min_idx = np.argmin(arr)
        arr[first_min_idx] = np.inf
        second_min_idx = np.argmin(arr)

        return [first_min_idx, second_min_idx]

    def find_common_elements_with_occurrences(list1, list2, list3):
        elements_occurrences = {}

        for idx, lst in enumerate([list1, list2, list3]):
            for element in lst:
                if element not in elements_occurrences:
                    elements_occurrences[element] = set()
                elements_occurrences[element].add(idx)

        # Filter elements that appear in more than one list
        common_elements = {element for element, indices in elements_occurrences.items() if len(indices) > 1}

        return common_elements, elements_occurrences

    dot = draw_tree(root_node)
    levels = get_levels(root_node)
    for i, levell in enumerate(levels):
        if i == 0 or i == len(levels) - 1:
            continue

        level = [node for node in levell if node.children != []]
        npeidxs = get_largest_two_indices([node.npe for node in level])
        lnpeidxs = get_largest_two_indices([node.lnpe for node in level])
        disparityidxs = get_smallest_two_indices([node.top2disparity for node in level])

        common_elements, occurrences = find_common_elements_with_occurrences(npeidxs, lnpeidxs, disparityidxs)

        for idx in occurrences.keys():
            node = level[idx]
            npecolor, lnpecolor, disparitycolor = 'black', 'black', 'black'
            for occ in occurrences[idx]:
                if occ == 0:
                    npecolor = "orange"
                if occ == 1:
                    lnpecolor = "blue"
                if occ == 2:
                    disparitycolor = "green"
            label = f"""<
                <table border="0" cellborder="1" cellspacing="0">
                    <tr><td><font color="{npecolor}">npe:</font></td><td><font color="{npecolor}">{node.npe:.2f}</font></td></tr>
                    <tr><td><font color="{lnpecolor}">lnpe:</font></td><td><font color="{lnpecolor}">{node.lnpe:.2f}</font></td></tr>
                    <tr><td><font color="{disparitycolor}">disparity:</font></td><td><font color="{disparitycolor}">{node.top2disparity:.2f}</font></td></tr>
                    <tr><td><font color="black">entropy:</font></td><td><font color="black">{node.entropy:.2f}</font></td></tr>
                </table>
            >"""
            node_id = str(id(node))
            dot.node(node_id, label, color="blue")

    dot.render(filename)

def visualize_state(root_node, filename):
    def draw_tree(node, dot=None):
        if dot is None:
            dot = graphviz.Digraph()
        # Define condition for coloring nodes
        color = 'black'
        if len(node.children) == 0 and node.correct:  # correct leaf nodes
            color = 'red'
        if node.state != []:
            label = f"""{node.state[-1]}"""
        else:
            label = " "
        dot.node(str(id(node)), label, color=color)
        for child in node.children:
            dot.edge(str(id(node)), str(id(child)))
            draw_tree(child, dot)

        return dot

    dot = draw_tree(root_node)
    dot.render(filename)

# Load the tree
def load_tree(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    for idx in [0,1,2,8, 39, 74, 107, 119, 141, 144, 154, 161, 177, 211, 214, 218]: # 
        question = f'Q{idx}'
        tree_num = 1
        pklpath = f'./output_trees/{question}/tree_{tree_num}/tree.pkl'
        savepath = f'./output_trees/{question}/tree_{tree_num}/uncertainty_metrics'
        loaded_root = load_tree(filename=pklpath)
        visualize_metrics(root_node=loaded_root, filename=savepath)
        #visualize_state(root_node=loaded_root, filename=savepath)
