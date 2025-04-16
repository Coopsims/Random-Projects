# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


class SpotifyModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=None, num_classes=2, dropout_rate=0.3):
        """
        A deeper feed-forward network for binary classification on tabular data.

        Args:
            input_size (int): Number of input features.
            hidden_sizes (list): List specifying the number of neurons in each hidden layer.
            num_classes (int): Number of output classes (2 for binary classification).
            dropout_rate (float): Dropout probability for regularization.
        """
        super(SpotifyModel, self).__init__()

        # Provide defaults if hidden_sizes is not supplied
        # e.g. three hidden layers with 128, 64, and 32 neurons each
        if hidden_sizes is None:
            hidden_sizes = [64, 128, 64, 32]

        self.num_layers = len(hidden_sizes)
        self.linears = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # 1) Input layer -> first hidden layer
        self.linears.append(nn.Linear(input_size, hidden_sizes[0]))
        self.batch_norms.append(nn.BatchNorm1d(hidden_sizes[0]))
        self.dropouts.append(nn.Dropout(dropout_rate))

        # 2) Hidden layers
        for i in range(self.num_layers - 1):
            self.linears.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            self.dropouts.append(nn.Dropout(dropout_rate))

        # 3) Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input features of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Raw output logits of shape (batch_size, num_classes).
        """
        # Pass through each hidden layer in sequence
        for i in range(self.num_layers):
            x = self.linears[i](x)  # Linear transform
            x = self.batch_norms[i](x)  # Batch normalization
            x = F.relu(x)  # Non-linear activation
            x = self.dropouts[i](x)  # Dropout for regularization

        # Finally, pass through the output layer
        x = self.output_layer(x)

        return x


class BuildModel:
    def __init__(self):
        self.threshold = 35
        # Check for CUDA first, then MPS (Apple Silicon), then fall back to CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.scaler = None
        self.input_size = None

        self.load_data()
        self.build_model()

    def load_data(self):
        # Load and preprocess data
        df = pd.read_csv('../dataset.csv')
        df["popularity_bin"] = df["popularity"].apply(lambda x: 1 if x >= self.threshold else 0)
        df.drop(columns=["popularity"], inplace=True)

        df["explicit"] = df["explicit"].astype(int)

        # Convert 'track_genre' (categorical) to integer codes
        df["track_genre"] = df["track_genre"].astype("category").cat.codes

        df["artists"] = df["artists"].astype("category").cat.codes

        # Extract features and target
        y = df['popularity_bin'].values

        df.drop(columns=[
            "popularity_bin",  # label
            "track_id",  # originally object/string
            "album_name",
            "track_name"
        ], inplace=True)

        x = df.values

        # Standardize features
        self.scaler = StandardScaler().fit(x)
        X = self.scaler.transform(x)

        self.input_size = X.shape[1]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)

        # Create datasets
        train_dataset = Data(X_train_tensor, y_train_tensor)
        test_dataset = Data(X_test_tensor, y_test_tensor)

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=1024, shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=1024, shuffle=False
        )

        print(f"Training samples: {len(train_dataset)}")
        print(f"Testing samples: {len(test_dataset)}")

    def build_model(self):
        # Initialize the model
        self.model = SpotifyModel(input_size=self.input_size).to(self.device)
        print(self.model)

    def train_model(self, epochs=30, learning_rate=0.001):
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Print epoch statistics
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = 100 * correct / total
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

            # Evaluate on test set every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.evaluate_model(dataset='test', verbose=False)

        # Save the trained model
        torch.save(self.model.state_dict(), '../best_model.pth')
        print("Model saved to '../best_model.pth'")

    def evaluate_model(self, dataset='test', verbose=True):
        """
        Evaluate the model on the specified dataset.

        Args:
            dataset (str): 'train' or 'test' dataset to evaluate on
            verbose (bool): Whether to print detailed metrics and plot confusion matrix

        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Set model to evaluation mode
        self.model.eval()

        # Select the appropriate data loader
        data_loader = self.train_loader if dataset == 'train' else self.test_loader

        # Lists to store predictions and true labels
        all_preds = []
        all_labels = []

        # No gradient computation needed for evaluation
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                # Store predictions and labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Convert lists to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        conf_matrix = confusion_matrix(all_labels, all_preds)

        # Store metrics in a dictionary
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix
        }

        # Print metrics if verbose
        if verbose:
            print(f"\nEvaluation on {dataset} dataset:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("\nConfusion Matrix:")
            print(conf_matrix)

            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Not Popular', 'Popular'],
                        yticklabels=['Not Popular', 'Popular'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - {dataset.capitalize()} Set')
            plt.tight_layout()
            plt.savefig(f'confusion_matrix_{dataset}.png')
            plt.show()

        # Set model back to training mode if we're still training
        if self.model.training:
            self.model.train()

        return metrics

class Data(Dataset):
    def __init__(self, features, labels):
        """
        Store the features and labels internally.
        :param features: A numpy array or PyTorch tensor of feature data.
        :param labels: A numpy array or PyTorch tensor of corresponding labels.
        """
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        """
        Return a single sample (features, label) by index.
        """
        # Retrieve features and label at position 'index'
        x = self.features[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.features)


if __name__ == "__main__":
    # Create an instance of BuildModel
    model_builder = BuildModel()

    # Train the model
    print("\n=== Starting Model Training ===")
    model_builder.train_model(epochs=50, learning_rate=0.001)

    # Evaluate the model on both training and test sets
    print("\n=== Evaluating Model on Training Set ===")
    train_metrics = model_builder.evaluate_model(dataset='train', verbose=True)

    print("\n=== Evaluating Model on Test Set ===")
    test_metrics = model_builder.evaluate_model(dataset='test', verbose=True)

    # Print comparison of metrics
    print("\n=== Performance Comparison ===")
    print(f"Training Accuracy: {train_metrics['accuracy']:.4f} | Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Training F1 Score: {train_metrics['f1_score']:.4f} | Test F1 Score: {test_metrics['f1_score']:.4f}")

    print("\nModel training and evaluation completed successfully!")

    """
    50 EPOCH
    === Performance Comparison ===
    Training Accuracy: 0.8177 | Test Accuracy: 0.7097
    Training F1 Score: 0.8177 | Test F1 Score: 0.7097
    
    100 EPOCH
    === Performance Comparison ===
    Training Accuracy: 0.8588 | Test Accuracy: 0.7123
    Training F1 Score: 0.8588 | Test F1 Score: 0.7122
    
    """
