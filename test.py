import numpy as np

def find_best_split(X, y):
    # Example data:
    X = np.array([
        [2, 3],  # Sample 0
        [5, 4],  # Sample 1
        [9, 6],  # Sample 2
        [4, 7],  # Sample 3
        [8, 1],  # Sample 4
        [7, 2]   # Sample 5
    ])
    y = np.array([0, 0, 1, 0, 1, 1])

    n_samples, n_features = X.shape  # 6 samples, 2 features
    best_gini = float('inf')
    
    # Loop through each feature (0 and 1)
    for feature in range(n_features):
        # For feature 0:
        # Original: [2, 5, 9, 4, 8, 7]
        # Labels:   [0, 0, 1, 0, 1, 1]
        
        # After sorting:
        sort_idx = np.argsort(X[:, feature])
        feature_sorted = [2, 4, 5, 7, 8, 9]  # Sorted values
        y_sorted = [0, 0, 0, 1, 1, 1]       # Corresponding labels
        
        # Initialize counts
        left_count = [0, 0]    # [count of 0s, count of 1s] in left group
        right_count = [3, 3]   # All samples start in right group
        
        # Try each possible split point
        for i in range(1, n_samples):
            # Example: First iteration
            current_class = y_sorted[i-1]  # = 0
            left_count[0] += 1   # Add class 0 to left
            right_count[0] -= 1  # Remove class 0 from right
            
            # Skip if values are same
            if feature_sorted[i] == feature_sorted[i-1]:
                continue
                
            # Calculate Gini impurity
            # Gini = 1 - (p0² + p1²)
            # where p0, p1 are proportions of each class
            
            # For left group:
            left_gini = 1.0 - sum((count/i)**2 for count in left_count)
            
            # For right group:
            right_gini = 1.0 - sum((count/(n_samples-i))**2 
                                 for count in right_count)
            
            # Weighted average
            weighted_gini = (i * left_gini + (n_samples-i) * right_gini) / n_samples
            
            # Update if this split is better
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature
                best_threshold = (feature_sorted[i] + feature_sorted[i-1]) / 2

    return best_feature, best_threshold

# Let's see it in action
def show_split_details(X, y):
    feature = 0  # Look at first feature
    sort_idx = np.argsort(X[:, feature])
    feature_sorted = X[sort_idx, feature]
    y_sorted = y[sort_idx]
    
    print("Sorted feature values:", feature_sorted)
    print("Sorted labels:", y_sorted)
    
    # Try each split
    for i in range(1, len(X)):
        left_values = feature_sorted[:i]
        right_values = feature_sorted[i:]
        left_labels = y_sorted[:i]
        right_labels = y_sorted[i:]
        
        print(f"\nSplit point {i}:")
        print(f"Left group: values={left_values}, labels={left_labels}")
        print(f"Right group: values={right_values}, labels={right_labels}")
        
        # Calculate Gini
        left_count = np.bincount(left_labels, minlength=2)
        right_count = np.bincount(right_labels, minlength=2)
        
        left_gini = 1.0 - sum((count/len(left_labels))**2 for count in left_count)
        right_gini = 1.0 - sum((count/len(right_labels))**2 for count in right_count)
        weighted_gini = (len(left_labels) * left_gini + len(right_labels) * right_gini) / len(X)
        
        print(f"Weighted Gini: {weighted_gini:.4f}")

# Example usage
y = np.array([0, 0, 1, 0, 1])
left_mask = np.array([True, True, False, False, False])
print(y[left_mask])