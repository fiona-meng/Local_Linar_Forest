import numpy as np
import pandas as pd
import os
from urllib.error import URLError
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier


class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None

    def fit_transform(self, X):
        # standardize the data
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X = (X - X_mean) / X_std
        # Compute covariance matrix
        cov_matrix = np.cov(X, rowvar=False)
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        # Select top n_components
        if self.n_components is not None:
            eigenvectors = eigenvectors[:, :self.n_components]
        self.components_ = eigenvectors.T
        # Transform data
        return np.dot(X, self.components_.T)

    def transform(self, X):
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X = (X - X_mean) / X_std
        return np.dot(X, self.components_.T)


class LinearRegression:
    def __init__(self, lambda_reg=0.1, learning_rate=0.01, max_iter=100, tol=1e-4):
        self.coef_ = None
        self.intercept_ = None
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def _loss_function(self, X, y, beta):
        """Compute loss function L(Y, X, G, PGS) + λP(β)"""
        '''Loss: L = (1/2n) * sum((y - Xβ)²) + (λ/2n) * sum(β²)'''
        n_samples = X.shape[0]
        y_pred = X @ beta
        mse = 0.5 * np.mean((y - y_pred) ** 2)
        l2_penalty = 0.5 * (self.lambda_reg / n_samples) * np.sum(beta[1:] ** 2) 
        return mse + l2_penalty

    def _gradient(self, X, y, beta):
        """Compute gradient of loss function
        ∂L/∂β = (1/n) * X^T(y - Xβ) + (λ/n) * β
        """
        n_samples = X.shape[0]
        error = y - X @ beta
        grad = (1/n_samples) * X.T @ error
        # Regularization gradient (excluding intercept)
        grad[1:] += (self.lambda_reg/n_samples) * beta[1:]
        return grad

    def fit(self, X, y):
        # Add bias term
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        n_features = X_b.shape[1]
        
        # Initialize parameters
        beta = np.zeros(n_features)
        prev_loss = float('inf')
        
        # Gradient descent
        for iter in range(self.max_iter):
            # Compute gradient
            grad = self._gradient(X_b, y, beta)
            
            # Update parameters
            beta -= self.learning_rate * grad
            
            # Check convergence
            current_loss = self._loss_function(X_b, y, beta)
            if abs(prev_loss - current_loss) < self.tol:
                break
            prev_loss = current_loss
        
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, criterion="entropy"):
        # Tree structure arrays
        self.children_left = []    # indices of left children
        self.children_right = []   # indices of right children
        self.feature = []          # feature indices for splits
        self.threshold = []        # threshold values for splits
        self.value = []           # store majority class for leaf nodes
        self.node_count = 0        # total number of nodes
        
        # Tree parameters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.feature_names = None

    def _entropy(self, y):
        """Calculate entropy of a node"""
        if len(y) == 0:
            return 0
        p1 = np.sum(y) / len(y)
        if p1 == 0 or p1 == 1:
            return 0
        return -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
    
    def _gini(self, y):
        """Calculate gini impurity of a node"""
        p1 = np.sum(y) / len(y)
        return 1 - p1**2 - (1 - p1)**2

    def _information_gain(self, y, y_left, y_right):
        """Calculate information gain from a split using gini impurity
        """
        parent_gini = self._gini(y)
        n = len(y)
        n_l, n_r = len(y_left), len(y_right)
        if n_l == 0 or n_r == 0:
            return 0
        child_entropy = (n_l/n) * self._gini(y_left) + (n_r/n) * self._gini(y_right)
        return parent_gini - child_entropy

    def _add_node(self, feature_idx=None, threshold=None, value=None):
        """Add a new node to the tree"""
        node_id = self.node_count
        self.children_left.append(-1)  # -1 indicates no child
        self.children_right.append(-1)
        self.feature.append(feature_idx if feature_idx is not None else -1)
        self.threshold.append(threshold if threshold is not None else 0.0)
        self.value.append(value)
        self.node_count += 1
        return node_id

    def _best_split(self, X, y):
        """Find best feature and threshold for splitting"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if (np.sum(left_mask) < self.min_samples_leaf or 
                    np.sum(right_mask) < self.min_samples_leaf):
                    continue

                gain = self._information_gain(
                    y, 
                    y[left_mask], 
                    y[right_mask]
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    
    def _build_tree(self, X, y, depth=0, parent=None):
        """Recursively build the tree"""
        n_samples = len(y)

        # Create a leaf node if stopping criteria are met
        if (n_samples < self.min_samples_split or 
            (self.max_depth is not None and depth >= self.max_depth) or
            len(np.unique(y)) == 1):
            # Create leaf node with majority class
            majority_class = np.bincount(y).argmax()
            return self._add_node(value=majority_class)

        # Find best split
        feature, threshold, gain = self._best_split(X, y)

        # If no valid split found, make leaf
        if feature is None:
            majority_class = np.bincount(y).argmax()
            return self._add_node(value=majority_class)

        # Create current node
        current_node = self._add_node(feature_idx=feature, threshold=threshold)

        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # Build children
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1, current_node)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1, current_node)

        # Link children to current node
        self.children_left[current_node] = left_child
        self.children_right[current_node] = right_child

        return current_node

    def fit(self, X, y, feature_names=None):
        """Fit decision tree to training data"""
        self.feature_names = feature_names
        self._build_tree(X, y)
        return self

    def predict_single(self, x, node_id=0):
        """Predict single sample"""
        # If we're at a leaf node
        if self.feature[node_id] == -1:
            return self.value[node_id]

        # If not at leaf, decide which child to go to
        if x[self.feature[node_id]] <= self.threshold[node_id]:
            return self.predict_single(x, self.children_left[node_id])
        else:
            return self.predict_single(x, self.children_right[node_id])


    def predict(self, X):
        """Predict multiple samples"""
        return np.array([self.predict_single(x) for x in X])

    def apply_single(self, x, node_id=0):
        """Return leaf node ID for a single sample"""
        if self.feature[node_id] == -1:  # Leaf node
            return node_id
            
        if x[self.feature[node_id]] <= self.threshold[node_id]:
            return self.apply_single(x, self.children_left[node_id])
        else:
            return self.apply_single(x, self.children_right[node_id])

    def apply(self, X):
        """Return leaf node IDs for multiple samples"""
        return np.array([self.apply_single(x) for x in X])

    def predict_with_nodes(self, X, sample_ids):
        """Predict and return both predictions and node IDs
        
        Parameters:
        -----------
        X : array-like
            Features to predict
        sample_ids : array-like, optional
            Sample IDs to include in output
            
        Returns:
        --------
        DataFrame with columns [id, node_id, prediction]
        """
            
        # Get predictions and node IDs
        predictions = self.predict(X)
        node_ids = self.apply(X)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'id': sample_ids,
            'node_id': node_ids,
            'prediction': predictions
        })
        
        return results

    def plot_tree(self, figsize=(20,10), fontsize=10):
        """Visualize the decision tree using matplotlib"""
        from sklearn.tree import DecisionTreeClassifier, plot_tree
        import matplotlib.pyplot as plt
        
        # Create a sklearn tree for visualization
        sklearn_tree = DecisionTreeClassifier()
        n_nodes = len(self.feature)
        
        # Basic tree structure
        sklearn_tree.tree_ = type('TreeStruct', (), {
            'children_left': np.array(self.children_left, dtype=np.int32),
            'children_right': np.array(self.children_right, dtype=np.int32),
            'feature': np.array(self.feature, dtype=np.int32),
            'threshold': np.array(self.threshold, dtype=np.float32),
            'n_classes': np.array([2], dtype=np.int32),
            'n_outputs': np.int32(1),
            'n_node_samples': np.ones(n_nodes, dtype=np.int32),
            'weighted_n_node_samples': np.ones(n_nodes, dtype=np.float32),
            'impurity': np.zeros(n_nodes, dtype=np.float32),
            'value': np.zeros((n_nodes, 1, 2), dtype=np.float32)
        })
        
        # Set basic class distribution
        for i in range(n_nodes):
            if self.feature[i] == -1:  # Leaf node
                if self.value[i] == 0:  # Class 0
                    sklearn_tree.tree_.value[i] = [[1, 0]]
                else:  # Class 1
                    sklearn_tree.tree_.value[i] = [[0, 1]]
            else:  # Internal node
                sklearn_tree.tree_.value[i] = [[0.5, 0.5]]
        
        # Create the plot
        plt.figure(figsize=figsize)
        tree.plot_tree(sklearn_tree, 
                    feature_names=self.feature_names,
                    class_names=['Non-obese', 'Obese'],
                    filled=False,  # Disable color filling to avoid color issues
                    rounded=True,
                    fontsize=fontsize)
        
        return plt
 
class LocalLinearForest:
    def __init__(self, 
                 n_estimators=100,
                 criterion="entropy",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features=None,
                 n_components=None,
                 random_state=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.n_components = n_components
        self.random_state = random_state
        self.trees_ = []
        self.pca_ = None

    def _bootstrap_sample(self, X, y, random_state):
        """Generate bootstrap sample"""
        rng = np.random.RandomState(random_state)
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, G, y):
        """Fit Local Linear Forest"""
        # Reduce dimensionality of G using PCA
        if self.n_components is not None:
            self.pca_ = PCA(n_components=self.n_components)
            G_reduced = self.pca_.fit_transform(G)
        else:
            G_reduced = G

        # Combine features
        X_combined = np.hstack((X, G_reduced))
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Train individual trees
        for i in range(self.n_estimators):
            # Bootstrap sample
            X_boot, y_boot = self._bootstrap_sample(
                X_combined, y, 
                random_state=rng.randint(np.iinfo(np.int32).max)
            )
            
            # Create and train tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion
            )
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)
        
        return self

    def predict(self, X, G):
        """Make predictions with Local Linear Forest"""
        # Transform G if PCA was used
        if self.n_components is not None:
            G_reduced = self.pca_.transform(G)
        else:
            G_reduced = G
            
        # Combine features
        X_combined = np.hstack((X, G_reduced))
        
        # Get predictions from all trees
        predictions = np.zeros((X_combined.shape[0], len(self.trees_)))
        for i, tree in enumerate(self.trees_):
            predictions[:, i] = tree.predict(X_combined)
            
        # Return average prediction
        return np.mean(predictions, axis=1)

    def apply(self, X, G):
        """Return the leaf indices for each sample"""
        # Transform G if PCA was used
        if self.n_components is not None:
            G_reduced = self.pca_.transform(G)
        else:
            G_reduced = G
            
        # Combine features
        X_combined = np.hstack((X, G_reduced))
        
        # Get leaf indices from all trees
        leaf_indices = []
        for tree in self.trees_:
            leaf_indices.append(tree.apply(X_combined))
            
        return np.array(leaf_indices).T

    
    def cross_validate(self, X, G, y, n_splits=5, random_state=None):
        """Perform k-fold cross-validation
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        G : array-like of shape (n_samples, n_genetic_features)
            Genetic features
        y : array-like of shape (n_samples,)
            Target values
        n_splits : int, default=5
            Number of folds for cross-validation
        random_state : int, default=None
            Random state for reproducibility
            
        Returns:
        --------
        cv_scores : dict
            Dictionary containing arrays of train and validation scores
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        train_scores = []
        val_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            G_train, G_val = G[train_idx], G[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Fit model
            self.fit(X_train, G_train, y_train)
            
            # Get predictions
            train_pred = self.predict(X_train, G_train)
            val_pred = self.predict(X_val, G_val)
            
            # Calculate MSE
            train_mse = mean_squared_error(y_train, train_pred)
            val_mse = mean_squared_error(y_val, val_pred)
            
            train_scores.append(train_mse)
            val_scores.append(val_mse)
            
            print(f"Fold {fold+1}/{n_splits}:")
            print(f"Train MSE: {train_mse:.4f}")
            print(f"Validation MSE: {val_mse:.4f}\n")
        
        return {
            'train_scores': np.array(train_scores),
            'val_scores': np.array(val_scores),
            'train_mean': np.mean(train_scores),
            'train_std': np.std(train_scores),
            'val_mean': np.mean(val_scores),
            'val_std': np.std(val_scores)
        }
    

class DataPreprocessor:
    def __init__(self):
        self.data = None
        self.demographic = None
        self.genetic = None
        self.genetic_pca = None
        self.target = None
        
    def load_raw_data(self, file_name, chunk_size=100):
        """Load raw data from CSV file"""
        chunks = []
        for chunk in pd.read_csv(file_name, chunksize=chunk_size):
            chunks.append(chunk)
            print(f"Loaded {len(chunks) * chunk_size} rows...", end='\r')
        
        self.data = pd.concat(chunks, ignore_index=True)
        print(f"\nSuccessfully loaded {len(self.data)} rows")
        
    def preprocess_data(self, n_components=20):
        """Preprocess data and apply PCA to genetic features"""
        # Define feature groups
        demographic_features = ['id', 'SEX', 'DECADE_BIRTH', 'ETHNICITY', 'RACE']
        genetic_features = [f'X{i}' for i in range(1, 2048)]
        target_feature = 'CASE_CONTROL_EXTREMEOBESITY'
        
        # Split data into groups
        self.demographic = self.data[demographic_features]
        self.genetic = self.data[genetic_features]
        self.target = self.data[[target_feature]]
        
        # Apply PCA to genetic data
        pca = PCA(n_components=n_components)
        self.genetic_pca = pd.DataFrame(
            pca.fit_transform(self.genetic),
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        print("Data preprocessing completed")
        
    def save_processed_data(self, output_file='processed_data.csv'):
        """Save processed data to CSV"""
        # Combine all processed data
        processed_data = pd.concat([
            self.demographic,
            self.genetic_pca,
            self.target,
            self.genetic
        ], axis=1)
        
        # Save to CSV
        processed_data.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
    
class DataLoader:
    def __init__(self):
        self.id = None
        self.X = None
        self.y = None
        self.G = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.G_train = None
        self.G_test = None
        self.test_ids = None
        self.train_ids = None

    def load_data(self, file_name):
        """Load data from CSV file"""
        self.data = pd.read_csv(file_name)
        print(f"Data loaded from {file_name}")
        demographic_features = ['SEX', 'DECADE_BIRTH', 'ETHNICITY', 'RACE']
        genetic_features = [f'X{i}' for i in range(1, 2048)]
        target_feature = 'CASE_CONTROL_EXTREMEOBESITY'
        genetic_pca_features = [f'PC{i}' for i in range(1, 21)]
        self.X = self.data[demographic_features + genetic_pca_features].to_numpy()
        self.y = self.data[target_feature].to_numpy()
        self.G = self.data[genetic_features].to_numpy()
        self.id = self.data['id'].to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test, self.G_train, self.G_test, self.train_ids_ids, self.test_ids = train_test_split(
            self.X, self.y, self.G, self.id, test_size=0.2, random_state=42
        )
        print("Data loaded and split")

    def plot_tree_results(self, predictions_df, min_samples=50, save_path=None):
        """Plot samples colored by leaf nodes (only nodes with > 50 samples)"""
        fig = plt.figure(figsize=(30, 8))
        ax = fig.add_subplot(111)
        
        # Get nodes with > 50 samples
        node_counts = predictions_df['node_id'].value_counts()
        large_nodes = node_counts[node_counts > min_samples].index
        
        # Generate colors only for large nodes
        colors = plt.cm.rainbow(np.linspace(0, 1, len(large_nodes)))
        color_dict = dict(zip(large_nodes, colors))
        
        # Transform PC1 values using Box-Muller transform
        pc1_values = self.data['PC1'].values
        # Normalize to [0,1] first
        pc1_norm = (pc1_values - np.min(pc1_values)) / (np.max(pc1_values) - np.min(pc1_values))
        # Apply Box-Muller transform
        R = np.sqrt(-2 * np.log(pc1_norm + 1e-100))  
        theta = 2 * np.pi * pc1_norm
        self.data['transformed_pc1'] = R * np.cos(theta) * 10  
        
        # Plot only large nodes
        for node_id in large_nodes:
            node_mask = predictions_df['node_id'] == node_id
            node_sample_ids = predictions_df[node_mask]['id']
            node_data = self.data[self.data['id'].isin(node_sample_ids)]
            
            ax.scatter(
                node_data['transformed_pc1'],
                node_data['PC2'],
                c=[color_dict[node_id]],
                label=f'Node {node_id} (n={len(node_sample_ids)})',
                alpha=0.6
            )
        
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('Box-Muller Transformed PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'Sample Distribution (Nodes with >{min_samples} samples)')
        
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_demographic_distribution(self, predictions_df, save_path=None):
        """Plot demographic distribution (Race x Ethnicity) for each node"""
        # Create figure
        plt.figure(figsize=(100, 8))
        
        # Get unique nodes
        unique_nodes = predictions_df['node_id'].unique()
        
        # Define demographic combinations
        demographics = {
            'White Non-Hispanic': (1, 1),
            'White Hispanic': (1, 2),
            'Black Non-Hispanic': (2, 1),
            'Black Hispanic': (2, 2)
        }
        
        # Calculate counts for each node
        node_demographics = []
        for node_id in unique_nodes:
            # Get samples in this node
            node_mask = predictions_df['node_id'] == node_id
            node_sample_ids = predictions_df[node_mask]['id']
            node_data = self.data[self.data['id'].isin(node_sample_ids)]
            
            # Count each demographic combination
            counts = {}
            for label, (race, ethnicity) in demographics.items():
                count = len(node_data[(node_data['RACE'] == race) & 
                                    (node_data['ETHNICITY'] == ethnicity)])
                counts[label] = count
            
            node_demographics.append({
                'node_id': node_id,
                **counts
            })
        
        # Convert to DataFrame for plotting
        df_plot = pd.DataFrame(node_demographics)
        df_plot.set_index('node_id', inplace=True)
        
        # Create bar plot
        ax = df_plot.plot(kind='bar', width=0.8)
        
        plt.title('Demographic Distribution Across Nodes')
        plt.xlabel('Node ID')
        plt.ylabel('Number of Samples')
        plt.legend(title='Demographics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Demographic distribution plot saved to {save_path}")
        else:
            plt.show()
        plt.close()

# Usage example:
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # Load and process data
    preprocessor.load_raw_data("obesity_data.csv")
    preprocessor.preprocess_data(n_components=20)
    preprocessor.save_processed_data("processed_obesity_data.csv")

    data = DataLoader()
    data.load_data("processed_obesity_data.csv")

    X_train, y_train, G_train, X_test, y_test, G_test, train_ids, test_ids = data.X_train, data.y_train, data.G_train, data.X_test, data.y_test, data.G_test, data.train_ids, data.test_ids
    # preprocessor.save_processed_data("processed_obesity_data.csv")
    print("X_train.shape:", X_train.shape)
    print("y_train.shape:", X_test.shape)
    print("total samples:", data.data.shape[0])

    tree = DecisionTree(max_depth=5)
    sklearn_tree = DecisionTreeClassifier(max_depth=5, random_state=42)


    sklearn_tree.fit(X_train, y_train)
    tree.fit(X_train, y_train)

    
    # Get predictions with node IDs
    results_custom = tree.predict_with_nodes(X_test, sample_ids=test_ids)


    sklearn_leaf_ids = sklearn_tree.apply(X_test)  # Get leaf node IDs
    sklearn_predictions = sklearn_tree.predict(X_test)  # Get predictions
    sklearn_results = pd.DataFrame({
        'id': test_ids,
        'node_id': sklearn_leaf_ids,
        'prediction': sklearn_predictions
    })
    
    # Save to CSV
    sklearn_results.to_csv('sklearn_predictions_with_nodes.csv', index=False)
    
    # Save results
    results_custom.to_csv('custom_tree_results.csv', index=False)
    
    # Plot results
    data.plot_tree_results(results_custom, min_samples=50, save_path='tree_nodes_visualization.png')
    data.plot_tree_results(sklearn_results, min_samples=50, save_path='sklearn_tree_nodes_visualization.png')

    # Plot demographic distribution
    data.plot_demographic_distribution(results_custom, 'demographic_distribution.png')
    data.plot_demographic_distribution(sklearn_results, 'sklearn_demographic_distribution.png')


    # Calculate metrics
    # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    # # Get predictions
    # custom_results = tree.predict_with_nodes(X_test, sample_ids=test_ids)
    # custom_pred = custom_results['prediction'].values
    # sklearn_pred = sklearn_tree.predict(X_test)
    
    # # Calculate metrics for both models
    # print("\nCustom Tree Metrics:")
    # print(f"Accuracy: {accuracy_score(y_test, custom_pred):.4f}")
    # print(f"Precision: {precision_score(y_test, custom_pred):.4f}")
    # print(f"Recall: {recall_score(y_test, custom_pred):.4f}")
    # print(f"F1 Score: {f1_score(y_test, custom_pred):.4f}")
    # print("\nConfusion Matrix:")
    # print(confusion_matrix(y_test, custom_pred))
    
    # print("\nSklearn Tree Metrics:")
    # print(f"Accuracy: {accuracy_score(y_test, sklearn_pred):.4f}")
    # print(f"Precision: {precision_score(y_test, sklearn_pred):.4f}")
    # print(f"Recall: {recall_score(y_test, sklearn_pred):.4f}")
    # print(f"F1 Score: {f1_score(y_test, sklearn_pred):.4f}")
    # print("\nConfusion Matrix:")
    # print(confusion_matrix(y_test, sklearn_pred))
    

    

    

