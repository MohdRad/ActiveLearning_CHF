import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def get_dynamic_batch_size(train_size, min_size=4, max_size=64, factor=4):
    """Calculate batch size based on current training dataset size."""
    return max(min_size, min(max_size, train_size // factor))
    
# Data Preparation
data = pd.read_csv("../data/chfAll.csv", header="infer")

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1, 1)  # Ensure Y is a column vector

# Experiment parameters
num_experiments = 100 #########
num_iterations = 300
initial_train_size = 10
sampling_size = 1

r2_scores_active_all = np.zeros((num_experiments, num_iterations))
r2_scores_random_all = np.zeros((num_experiments, num_iterations))

for exp in range(num_experiments):
    print(f"Running experiment {exp+1}/{num_experiments}")
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Scaling the data
    Xscaler = MinMaxScaler()
    Yscaler = MinMaxScaler()
    X_train = Xscaler.fit_transform(X_train)
    y_train = Yscaler.fit_transform(y_train)
    X_test = Xscaler.transform(X_test)
    y_test = Yscaler.transform(y_test)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    # Define the Feedforward Neural Network class
    class FNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(FNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=0.1)  # Dropout layer
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
    
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)  # Apply dropout after the first hidden layer
            x = self.fc2(x)
            x = self.relu(x)
            x = self.dropout(x)  # Apply dropout after the second hidden layer
            x = self.fc3(x)
            return x
    
    
    # Model parameters
    input_dim = X_train.shape[1]
    hidden_dim = 64
    output_dim = 1
    num_epochs = 300
    
    # Separate models for Active Learning and Random Sampling
    model_active = FNN(input_dim, hidden_dim, output_dim)
    model_random = FNN(input_dim, hidden_dim, output_dim)
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer_active = optim.Adam(model_active.parameters(), lr=0.0005, weight_decay=1e-5)
    optimizer_random = optim.Adam(model_random.parameters(), lr=0.0005, weight_decay=1e-5)

    
    # Split the initial training set and sampling pool
    initial_indices = np.random.choice(len(X_train), initial_train_size, replace=False)
    sampling_indices = np.setdiff1d(np.arange(len(X_train)), initial_indices)
    
    X_initial_active = X_train[initial_indices]
    y_initial_active = y_train[initial_indices]
    X_pool_active = X_train[sampling_indices]
    y_pool_active = y_train[sampling_indices]
    
    X_initial_random = X_train[initial_indices]
    y_initial_random = y_train[initial_indices]
    X_pool_random = X_train[sampling_indices]
    y_pool_random = y_train[sampling_indices]
    
    # Function to train the model
    def train_model(model, train_loader, criterion, optimizer, num_epochs):
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    # Function to calculate MAPE
    def calculate_mape(model, X, y, Yscaler):
        model.eval()
        with torch.no_grad():
            predictions = model(X)
            predictions_np = Yscaler.inverse_transform(predictions.numpy())
            y_np = Yscaler.inverse_transform(y.numpy())
            mape = mean_absolute_percentage_error(y_np, predictions_np)
        return mape, predictions_np, y_np
    
    # Random Sampling Loop
    r2_scores_random = []
    #batch_size = get_dynamic_batch_size(len(X_initial_active))
    
    batch_size = 8
    
    train_dataset_random = torch.utils.data.TensorDataset(X_initial_random, y_initial_random)
    train_loader_random = torch.utils.data.DataLoader(train_dataset_random, batch_size=batch_size, shuffle=True)
    train_model(model_random, train_loader_random, criterion, optimizer_random, num_epochs)
    
    for iteration in range(num_iterations):
        
        print(f"Random Sampling Iteration {iteration+1}")
    
        # Evaluate on the test set for random sampling
        _, predictions_np_random, y_test_np_random = calculate_mape(model_random, X_test, y_test, Yscaler)
        r2_random = r2_score(y_test_np_random, predictions_np_random)
        r2_scores_random_all[exp, iteration] = r2_random
    
        # Randomly select samples
        random_indices = np.random.choice(len(X_pool_random), sampling_size, replace=False)
        X_new_random = X_pool_random[random_indices]
        y_new_random = y_pool_random[random_indices]
    
        # Update training data
        X_initial_random = torch.cat((X_initial_random, X_new_random), dim=0)
        y_initial_random = torch.cat((y_initial_random, y_new_random), dim=0)
    
        # Remove selected samples from the pool
        mask_random = np.ones(len(X_pool_random), dtype=bool)
        mask_random[random_indices] = False
        X_pool_random = X_pool_random[mask_random]
        y_pool_random = y_pool_random[mask_random]
        
        batch_size = get_dynamic_batch_size(len(X_initial_random))
    
        # Retrain the random sampling model
        train_dataset_random = torch.utils.data.TensorDataset(X_initial_random, y_initial_random)
        train_loader_random = torch.utils.data.DataLoader(train_dataset_random, batch_size=batch_size, shuffle=True)
        train_model(model_random, train_loader_random, criterion, optimizer_random, num_epochs)
        
    # Active Learning Loop with Efficient Pool Sampling
    r2_scores_active = []
    
    train_dataset_active = torch.utils.data.TensorDataset(X_initial_active, y_initial_active)
    train_loader_active = torch.utils.data.DataLoader(train_dataset_active, batch_size=batch_size, shuffle=True)
    train_model(model_active, train_loader_active, criterion, optimizer_active, num_epochs)
    
    for iteration in range(num_iterations):
        print(f"Active Learning Iteration {iteration+1}")
    
        _, predictions_np, y_test_np = calculate_mape(model_active, X_test, y_test, Yscaler)
        r2_active = r2_score(y_test_np, predictions_np)
        r2_scores_active_all[exp, iteration] = r2_active
    
        # **Batch-Based Pool Evaluation (Efficient)**
        pool_sample_size = min(500, len(X_pool_active))  # Limit to 500 samples for speed
        subset_indices = np.random.choice(len(X_pool_active), pool_sample_size, replace=False)
        subset_X_pool = X_pool_active[subset_indices]
        subset_y_pool = y_pool_active[subset_indices]
    
        # Compute MAPE for the subset
        mape_values = [
            calculate_mape(model_active, subset_X_pool[i:i+1], subset_y_pool[i:i+1], Yscaler)[0]
            for i in range(len(subset_X_pool))
        ]
    
        # Select the sample with the **highest MAPE**
        top_index = subset_indices[np.argmax(mape_values)]
        X_new = X_pool_active[top_index].unsqueeze(0)
        y_new = y_pool_active[top_index].unsqueeze(0)
    
        # Update training data
        X_initial_active = torch.cat((X_initial_active, X_new), dim=0)
        y_initial_active = torch.cat((y_initial_active, y_new), dim=0)
        
        batch_size = get_dynamic_batch_size(len(X_initial_active))
    
        # Remove selected sample from pool
        mask = np.ones(len(X_pool_active), dtype=bool)
        mask[top_index] = False
        X_pool_active = X_pool_active[mask]
        y_pool_active = y_pool_active[mask]
    
        # Retrain the model
        train_dataset_active = torch.utils.data.TensorDataset(X_initial_active, y_initial_active)
        train_loader_active = torch.utils.data.DataLoader(train_dataset_active, batch_size=batch_size, shuffle=True)
        train_model(model_active, train_loader_active, criterion, optimizer_active, num_epochs)

# Compute mean and standard deviation of R2 scores across experiments
r2_mean_active = np.mean(r2_scores_active_all, axis=0)
r2_std_active = np.std(r2_scores_active_all, axis=0)

r2_mean_random = np.mean(r2_scores_random_all, axis=0)
r2_std_random = np.std(r2_scores_random_all, axis=0)

# Plot R2 scores with standard deviation
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_iterations + 1), r2_mean_active, marker='o', linestyle='-', label='Active Learning', color='blue')
plt.fill_between(range(1, num_iterations + 1), r2_mean_active - r2_std_active, r2_mean_active + r2_std_active, color='blue', alpha=0.2)

plt.plot(range(1, num_iterations + 1), r2_mean_random, marker='s', linestyle='--', label='Random Sampling', color='red')
plt.fill_between(range(1, num_iterations + 1), r2_mean_random - r2_std_random, r2_mean_random + r2_std_random, color='red', alpha=0.2)

plt.title('Mean R2 Score with Standard Deviation: Active Learning vs. Random Sampling')
plt.xlabel('Iteration')
plt.ylabel('Mean R2 Score')
plt.legend()
plt.grid(True)
plt.savefig("FNN_active_learning_vs_random_r2_mean_std.png")
