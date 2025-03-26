import numpy as np
from impurity import data_split, maximum_ig, entropy

"""
    This python file is going to build the tree.
    to build the tree we are going to use class.
    and the class consist of building_tree, total nodes used, depth,
    predict, accuracy and prune (Reduced Error Pruning) methods are used.
    @ Mohammad Wasil
"""

class Tree:
    def __init__(self):
        self.threshold = None
        self.feature_index = None
        self.left = None
        self.right = None
        self.label = None
        self.information_gain = None
        self.entropy = None
        self.before_prune_accuracy = []
        self.after_prune_accuracy = []

    # Building the tree or fit methods
    def build_tree(self, X, y):
        unique = np.unique(y) # identifying uniqueness of the classes
        if len(unique) == 1: # if the all values belongs to one class
            self.label = unique[0] # storing the unique class
            return self

        if len(X) <= 2: # if the data is so small
            self.label = np.bincount(y).argmax() # storing the maximum index of value
            return self

        feature, ig, threshold = maximum_ig(X, y) # calling maximum information gain
        x_left, x_right, y_left, y_right = data_split(X, y) # calling data split function

        if len(y_left) == 0 or len(y_right) == 0: # checking if the left and right target value is zero
            self.label = np.bincount(y).argmax() # store the maximum index of target value
            return self

        self.entropy = entropy(y) # storing entropy
        self.information_gain = ig # storing information gain
        self.feature_index = feature # storing feature
        self.threshold = threshold  # storing threshold

        self.left = Tree().build_tree(x_left, y_left) # creating left nodes
        self.right = Tree().build_tree(x_right, y_right)  # creating right nodes

        return self # return the object

    # Count total nodes used in the model
    def count_nodes(self):
        if self.left is None and self.right is None:
            return 1  # Leaf node
        left_count = self.left.count_nodes() if self.left else 0
        right_count = self.right.count_nodes() if self.right else 0
        return 1 + left_count + right_count

    # depth or count of nodes from root to leaf node
    def depth(self):
        if self.left is None and self.right is None:
            return 1  # Leaf node
        left_depth = self.left.depth() if self.left else 0
        right_depth = self.right.depth() if self.right else 0
        return 1 + max(left_depth, right_depth)

    # Predict the data
    def predict(self, X_sample):
        if self.left is None and self.right is None: # checking if the left and right nodes are not empty
            return self.label # return the leaf node

        if X_sample[self.feature_index] <= self.threshold: # check if selected feature values are less than threshold
            return self.left.predict(X_sample)  # recurse the predict on left child

        else:
            return self.right.predict(X_sample) # recurse the predict on right child


    # the batch predict method
    def predict_batch(self, X_test):
        result = [self.predict(sample) for sample in X_test] # calling the predict method and store the result
        return result

    # Accuracy method for checking the model prediction
    def accuracy(self, y_true, y_pred):
        true_pred = 0 # storing the amount of correct model prediction
        i = 0
        while i < len(y_true): # go through every single value the true data
            if y_pred[i] == y_true[i]: # checking if the ture and predict values match
                true_pred += 1 # increase the true_pred
            i += 1

        return round(true_pred/len(y_pred), 4) # printing the accuracy of model predicted on the data and round them into 4 decimal digits

    # Prune method to reduce the chance of overfiting
    def prune(self, X_val, y_val):
        # If it's a leaf node, return (nothing to prune)
        if self.left is None and self.right is None:
            return

        # Recursively prune left and right children
        if self.left is not None:
            self.left.prune(X_val, y_val)
        if self.right is not None:
            self.right.prune(X_val, y_val)

        # If both children are leaf nodes, attempt pruning
        if isinstance(self.left, Tree) and isinstance(self.right, Tree):
            if self.left.left is None and self.left.right is None and self.right.left is None and self.right.right is None:

                # Get the subset of validation data reaching this node
                X_subset, y_subset = [], []
                for i, x in enumerate(X_val):
                    if self.predict(x) in [self.left.label, self.right.label]:
                        X_subset.append(x)
                        y_subset.append(y_val[i])

                if len(X_subset) < 5:  # Avoid pruning with too few samples
                    return

                    # Calculate accuracy before pruning
                y_pred_before = self.predict_batch(X_subset)
                acc_before = self.accuracy(y_subset, y_pred_before)
                self.before_prune_accuracy.append(acc_before) # storing the accuracy before pruning

                # Backup current state
                self.left_backup, self.right_backup = self.left, self.right
                self.left = None
                self.right = None
                self.label = max(set(y_subset), key=y_subset.count)  # Majority class

                # Calculate accuracy after pruning
                y_pred_after = self.predict_batch(X_subset)
                acc_after = self.accuracy(y_subset, y_pred_after)
                self.after_prune_accuracy.append(acc_after) # storing the accuracy after prune

                # Prune if accuracy does not decrease
                if acc_after >= acc_before:
                    # Keep pruning accuracy is same or better
                    self.left_backup, self.right_backup = None, None
                else:
                    # Restore original structure accuracy dropped
                    self.left, self.right = self.left_backup, self.right_backup
                    self.label = None  # Restore to internal node


