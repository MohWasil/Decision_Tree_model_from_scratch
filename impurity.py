import math
import numpy as np

"""
    This python file is the core source of decision tree algorithm.
    here we are going to implement the calculation of Entropy(disorder) , Information Gain
    finding Maximum information gain of every feature and threshold of every values of feature.
    then split the data into left or right child.
    @ Mohammad Wasil
"""

def entropy(targets):
    visited = [] # checking once we are calculating values
    entro = 0 # storing total entropy
    length = len(targets) # length of target variable(y)
    for target in targets: # checking every value of target
        if target not in visited: # checking there is no duplicate value
            visited.append(target)
            count = np.count_nonzero(targets==target) # counting total amount of a value in the target
            prob = count/length # finding the probability of a value
            entro += -(prob * math.log2(prob)) # finding entropy

    return entro



"""
    This function is responsible to find information gain of every nodes
"""
def information_gain(X, y, feature_index, threshold):
    # Compute parent entropy
    H_parent = entropy(y)

    # Split data based on threshold
    left_mask = X[:, feature_index] <= threshold
    right_mask = X[:, feature_index] > threshold
    # finding the left and right values from target feature
    y_left = y[left_mask]
    y_right = y[right_mask]

    # Compute child entropies
    H_left = entropy(y_left) if len(y_left) > 0 else 0
    H_right = entropy(y_right) if len(y_right) > 0 else 0

    # Compute weighted entropy
    weight_left = len(y_left) / len(y)
    weight_right = len(y_right) / len(y)

    # calculating the total left and right children weights
    weighted_entropy = (weight_left * H_left) + (weight_right * H_right)

    # Compute Information Gain
    IG = H_parent - weighted_entropy
    return IG



"""
    This function is in charge of checking every features and finding its
    maximum information gain for selecting the best feature.
    To do this we need to find the best threshold also and that is why you can check it in the function
    for checking the best threshold of every feature values. 
    at the end of the function it will return (feature, threshold, IG) in such a way
    that IG is the best or maximum among every features checked.
"""

def maximum_ig(X, y):
    best_feature = None
    best_threshold = None
    best_ig = -1
    best_entropy = None  # Store entropy of the node

    features_ig = []  # Store max information gain per feature

    num_features = X.shape[1]  # Number of features (columns)
    for feature_index in range(num_features):  # Iterate over features
        feature = X[:, feature_index]  # Select feature column
        unique_feature_values = np.unique(feature)  # Get unique values
        sorted_feature_values = np.sort(unique_feature_values)  # Sort them

        threshold_temp_ig = []  # Reset for each feature

        # Compute all possible thresholds
        for ind in range(len(sorted_feature_values) - 1):
            threshold = (sorted_feature_values[ind] + sorted_feature_values[ind + 1]) / 2  # Midpoint
            ig= information_gain(X, y, feature_index, threshold)  # Compute IG
            threshold_temp_ig.append((ig, threshold))  # Store (IG, threshold)

        if threshold_temp_ig:  # Check if list is not empty
            best_threshold = max(threshold_temp_ig, key=lambda x: x[0])  # Max IG threshold
            features_ig.append((feature_index, best_threshold[0], best_threshold[1]))  # Store (feature, IG, threshold)

    if features_ig:  # Checking the feature information gain list is not empty
        return max(features_ig, key=lambda x: x[1])  # Best feature with the highest Information Gain
    else:
        return None  # Handle edge case where no valid splits exist


"""
    This function is going to split the data into left and right.
    This process assist us on creating the tree. 
"""
def data_split(X, y):
    y_left = None # initializing the target(y) left
    y_right = None # initializing the dependent(y) right

    feature, ig, threshold = maximum_ig(X, y) # calling the maximum information gain function
    # checking feature values based on threshold
    x_left_split = X[:, feature] <= threshold
    x_right_split = X[:, feature] > threshold

    # Selecting True values of dependent and independent features and split them into left and right
    x_left = X[x_left_split]
    x_right = X[x_right_split]
    y_left = y[x_left_split]
    y_right = y[x_right_split]

    if x_right.size == 0 and x_left.size == 0: # making sure both left and right splits are not zero
        return None

    # Returning the left, right of both X and y
    return x_left, x_right, y_left, y_right

