import numpy as np


def gini_impurity(y):
    """
    Computes the Gini impurity for a set of class labels y.

    Gini impurity: I_G = 1 - \sum_{i=1}^{K} p_i^2,
    where p_i is the proportion of class i.
    """
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1.0 - np.sum(probs ** 2)


def best_split(feature, y):
    """
    Finds the best split for a given continuous feature by evaluating
    the decrease in impurity.

    For each candidate threshold, the impurity decrease is:
    \Delta I = I_{parent} - \left(\frac{n_{left}}{n} I_{left} + \frac{n_{right}}{n} I_{right}\right)

    Returns the best threshold, its impurity decrease, and boolean indices for left/right splits.
    """
    unique_vals = np.unique(feature)
    best_threshold = None
    best_impurity_decrease = 0
    best_left_idx, best_right_idx = None, None

    impurity_parent = gini_impurity(y)
    n = len(y)

    for threshold in unique_vals:
        left_idx = feature <= threshold
        right_idx = feature > threshold

        # Ensure both splits are non-empty
        if left_idx.sum() == 0 or right_idx.sum() == 0:
            continue

        impurity_left = gini_impurity(y[left_idx])
        impurity_right = gini_impurity(y[right_idx])

        weighted_impurity = (left_idx.sum() / n) * impurity_left + (right_idx.sum() / n) * impurity_right
        impurity_decrease = impurity_parent - weighted_impurity

        if impurity_decrease > best_impurity_decrease:
            best_impurity_decrease = impurity_decrease
            best_threshold = threshold
            best_left_idx, best_right_idx = left_idx, right_idx

    return best_threshold, best_impurity_decrease, best_left_idx, best_right_idx


# Example data for a node:
# Consider a feature with candidate values and binary class labels y.
feature = np.array([2, 3, 7, 9, 10, 12, 15])
y = np.array([0, 0, 1, 1, 1, 0, 1])

# Find the best split for this feature
best_threshold, best_impurity_decrease, left_idx, right_idx = best_split(feature, y)

print("Best threshold:", best_threshold)
print("Impurity decrease:", best_impurity_decrease)
print("Indices for left split:", np.where(left_idx)[0])
print("Indices for right split:", np.where(right_idx)[0])


def gini_impurity(probs):
    """
    Computes the Gini impurity.

    The Gini impurity is defined as:

        I_G(p) = 1 - \sum_{i=1}^{K} p_i^2

    where p_i is the proportion of samples of class i.
    """
    return 1.0 - np.sum(np.square(probs))


def entropy(probs):
    """
    Computes the entropy using a logarithm with base 2.

    The entropy is given by:

        I_E(p) = -\sum_{i=1}^{K} p_i \log_2(p_i)

    Note: by convention, 0 * log2(0) is defined as 0.
    """
    probs = np.array(probs)
    # Replace zeros to avoid log2(0); these terms contribute 0.
    probs = np.where(probs == 0, 1, probs)
    return -np.sum(probs * np.log2(probs))


def log_loss_impurity(probs):
    """
    Computes the log loss (cross-entropy) impurity using the natural logarithm.

    It is defined as:

        L(p) = -\sum_{i=1}^{K} p_i \ln(p_i)

    Again, 0 * ln(0) is taken as 0 by convention.
    """
    probs = np.array(probs)
    probs = np.where(probs == 0, 1, probs)
    return -np.sum(probs * np.log(probs))


# Example: consider a node in a binary classification with class probabilities 0.7 and 0.3.
probs = [0.7, 0.3]

gini = gini_impurity(probs)
ent = entropy(probs)
logloss = log_loss_impurity(probs)

print(f"Gini Impurity: {gini:.4f}")
print(f"Entropy (base 2): {ent:.4f}")
print(f"Log Loss Impurity (natural log): {logloss:.4f}")