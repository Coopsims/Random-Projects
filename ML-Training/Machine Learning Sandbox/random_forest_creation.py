import numpy as np
from collections import Counter
import random

class Node:
    """
    Node class for decision tree
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """
        Initialize a node in the decision tree
        
        Parameters:
        -----------
        feature : int
            Index of the feature to split on
        threshold : float
            Threshold value for the feature
        left : Node
            Left child node
        right : Node
            Right child node
        value : any
            Value for leaf node (prediction)
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf(self):
        """Check if the node is a leaf node"""
        return self.value is not None


class DecisionTree:
    """
    Decision Tree implementation that can be used for classification or regression
    """
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None, criterion="gini"):
        """
        Initialize a decision tree
        
        Parameters:
        -----------
        min_samples_split : int
            Minimum number of samples required to split a node
        max_depth : int
            Maximum depth of the tree
        n_features : int
            Number of features to consider when looking for the best split
        criterion : str
            Function to measure the quality of a split ("gini" or "entropy" for classification, "mse" for regression)
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.criterion = criterion
        self.root = None
        
    def fit(self, X, y):
        """
        Build the decision tree
        
        Parameters:
        -----------
        X : numpy array
            Training data
        y : numpy array
            Target values
        """
        # If n_features is None, consider all features
        self.n_features = X.shape[1] if self.n_features is None else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)
        
    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow the decision tree
        
        Parameters:
        -----------
        X : numpy array
            Training data
        y : numpy array
            Target values
        depth : int
            Current depth of the tree
        
        Returns:
        --------
        Node
            Root node of the tree
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Check stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)
        
        # Randomly select features to consider
        feature_indices = np.random.choice(n_features, self.n_features, replace=False)
        
        # Find the best split
        best_feature, best_threshold = self._best_split(X, y, feature_indices)
        
        # Create child nodes
        left_indices, right_indices = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(best_feature, best_threshold, left, right)
    
    def _best_split(self, X, y, feature_indices):
        """
        Find the best feature and threshold for splitting
        
        Parameters:
        -----------
        X : numpy array
            Training data
        y : numpy array
            Target values
        feature_indices : numpy array
            Indices of features to consider
            
        Returns:
        --------
        tuple
            (best_feature, best_threshold)
        """
        best_gain = -float("inf")
        split_feature, split_threshold = None, None
        
        for feature_idx in feature_indices:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                # Calculate information gain
                gain = self._information_gain(y, X_column, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    split_feature = feature_idx
                    split_threshold = threshold
                    
        return split_feature, split_threshold
    
    def _information_gain(self, y, X_column, threshold):
        """
        Calculate information gain for a split
        
        Parameters:
        -----------
        y : numpy array
            Target values
        X_column : numpy array
            Feature values
        threshold : float
            Threshold value for the feature
            
        Returns:
        --------
        float
            Information gain
        """
        # Parent entropy/impurity
        parent_impurity = self._calculate_impurity(y)
        
        # Create children
        left_indices, right_indices = self._split(X_column, threshold)
        
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0
        
        # Calculate weighted average of children's impurity
        n = len(y)
        n_left, n_right = len(left_indices), len(right_indices)
        left_impurity = self._calculate_impurity(y[left_indices])
        right_impurity = self._calculate_impurity(y[right_indices])
        child_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        
        # Calculate information gain
        information_gain = parent_impurity - child_impurity
        return information_gain
    
    def _split(self, X_column, threshold):
        """
        Split the data based on a feature and threshold
        
        Parameters:
        -----------
        X_column : numpy array
            Feature values
        threshold : float
            Threshold value for the feature
            
        Returns:
        --------
        tuple
            (left_indices, right_indices)
        """
        left_indices = np.argwhere(X_column <= threshold).flatten()
        right_indices = np.argwhere(X_column > threshold).flatten()
        return left_indices, right_indices
    
    def _calculate_impurity(self, y):
        """
        Calculate impurity of a node
        
        Parameters:
        -----------
        y : numpy array
            Target values
            
        Returns:
        --------
        float
            Impurity value
        """
        if self.criterion == "gini":
            return self._gini(y)
        elif self.criterion == "entropy":
            return self._entropy(y)
        elif self.criterion == "mse":
            return self._mse(y)
        else:
            raise ValueError(f"Criterion '{self.criterion}' not supported")
    
    def _gini(self, y):
        """Calculate Gini impurity"""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities**2)
    
    def _entropy(self, y):
        """Calculate entropy"""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def _mse(self, y):
        """Calculate mean squared error (for regression)"""
        return np.mean((y - np.mean(y))**2)
    
    def _calculate_leaf_value(self, y):
        """
        Calculate the value for a leaf node
        
        Parameters:
        -----------
        y : numpy array
            Target values
            
        Returns:
        --------
        any
            Leaf value (most common class for classification, mean for regression)
        """
        if self.criterion in ["gini", "entropy"]:
            # Classification: return most common class
            return Counter(y).most_common(1)[0][0]
        else:
            # Regression: return mean value
            return np.mean(y)
    
    def predict(self, X):
        """
        Predict target values for samples in X
        
        Parameters:
        -----------
        X : numpy array
            Samples to predict
            
        Returns:
        --------
        numpy array
            Predicted values
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """
        Traverse the tree to make a prediction
        
        Parameters:
        -----------
        x : numpy array
            Single sample
        node : Node
            Current node
            
        Returns:
        --------
        any
            Predicted value
        """
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForest:
    """
    Random Forest implementation that can be used for classification or regression
    """
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_features=None, criterion="gini", bootstrap=True):
        """
        Initialize a random forest
        
        Parameters:
        -----------
        n_trees : int
            Number of trees in the forest
        min_samples_split : int
            Minimum number of samples required to split a node
        max_depth : int
            Maximum depth of the trees
        n_features : int
            Number of features to consider when looking for the best split
        criterion : str
            Function to measure the quality of a split
        bootstrap : bool
            Whether to use bootstrap samples
        """
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.trees = []
        
    def fit(self, X, y):
        """
        Build the random forest
        
        Parameters:
        -----------
        X : numpy array
            Training data
        y : numpy array
            Target values
        """
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=self.n_features,
                criterion=self.criterion
            )
            
            # Bootstrap sampling
            if self.bootstrap:
                n_samples = X.shape[0]
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_bootstrap, y_bootstrap = X[indices], y[indices]
                tree.fit(X_bootstrap, y_bootstrap)
            else:
                tree.fit(X, y)
                
            self.trees.append(tree)
            
    def predict(self, X):
        """
        Predict target values for samples in X
        
        Parameters:
        -----------
        X : numpy array
            Samples to predict
            
        Returns:
        --------
        numpy array
            Predicted values
        """
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Transpose to get predictions for each sample
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)
        
        # Aggregate predictions (majority vote for classification, mean for regression)
        if self.criterion in ["gini", "entropy"]:
            # Classification: majority vote
            predictions = np.array([self._most_common_label(pred) for pred in tree_predictions])
        else:
            # Regression: mean
            predictions = np.mean(tree_predictions, axis=1)
            
        return predictions
    
    def _most_common_label(self, y):
        """
        Find the most common label in an array
        
        Parameters:
        -----------
        y : numpy array
            Array of labels
            
        Returns:
        --------
        any
            Most common label
        """
        return Counter(y).most_common(1)[0][0]


# Example of how to use the random forest (not executed)
def example_usage():
    """
    Example of how to use the random forest
    """
    # Create a random dataset (not executed)
    # X = np.random.rand(100, 5)
    # y = np.random.choice([0, 1], size=100)
    
    # Create and train a random forest classifier
    rf = RandomForest(
        n_trees=10,
        min_samples_split=2,
        max_depth=5,
        criterion="gini"
    )
    
    # rf.fit(X, y)
    
    # Make predictions
    # predictions = rf.predict(X)
    
    return rf


if __name__ == "__main__":
    # This code will not be executed as per the requirements
    # Just showing how to create a random forest instance
    rf = example_usage()
    print("Random Forest created successfully!")