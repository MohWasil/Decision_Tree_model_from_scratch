# Import the libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import model_decision_visualization

# Load the dataset
data = load_breast_cancer()

# Split the dataset into features (X) and target (y)
X = data.data
y = data.target

# Split the dataset into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Print the sizes of the subsets
print("Training set size:", len(X_train))
print("Validation set size:", len(X_val))
print("Test set size:", len(X_test))



train = Tree()
training = train.build_tree(X_train, y_train)
score = training.predict_batch(X_test)
acc = accuracy_score(y_test, score)
nodes = training.count_nodes()
depth = training.depth()
print(f"Accuracy before pruning: {acc}")
print(f"Nodes before pruning: {nodes}")
print(f"Depth before pruning: {depth}")

prune_train = train.prune(X_val, y_val)
prune_pred = train.predict_batch(X_test)
prune_acc = accuracy_score(y_test, prune_pred)
nodes = train.count_nodes()
depth = train.depth()
print(f"After prune prediction: {prune_acc}")
print(f"Nodes after pruning: {nodes}")
print(f"Depth after pruning: {depth}")

vis = model_decision_visualization.TreeVisualizer(train)
vis.visualize()
