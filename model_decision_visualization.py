from graphviz import Digraph
"""
    This python file is dedicated for visualizing the decision tree algorithm's of how does this operate.
    how to compute the decision of nodes based on Entropy and Information gain.
    this can assist us for better understanding of decision tree classification operation.
    @ Mohammad Wasil 
"""
class TreeVisualizer:
    def __init__(self, tree):
        self.tree = tree # trained tree
        self.graph = Digraph()

    def add_nodes(self, node, node_id=0):
        if node is None:
            return node_id

        # storing the feature index, entropy of the feature and information gain
        label = f"Feature {node.feature_index}\nEntropy: {node.entropy:.3f}\nIG: {node.information_gain:.3f}" if node.feature_index is not None else f"Leaf: {node.label}"
        self.graph.node(str(node_id), label, shape="ellipse" if node.feature_index is not None else "box")

        # recall the function into left child
        left_id = node_id + 1
        right_id = self.add_nodes(node.left, left_id)

        # considering whether to go left child
        if node.left is not None:
            self.graph.edge(str(node_id), str(left_id), label=f" <= {node.threshold}")

        # or going to right child
        if node.right is not None:
            right_id = self.add_nodes(node.right, right_id + 1)
            self.graph.edge(str(node_id), str(right_id), label=f" > {node.threshold}")

        return right_id

    def visualize(self):
        self.add_nodes(self.tree)
        self.graph.render("decision_tree", format="png", view=True)


