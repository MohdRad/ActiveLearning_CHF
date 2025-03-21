import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
import random

from multiprocessing import Pool, cpu_count
import time
import multiprocessing

# Set multiprocessing start method
multiprocessing.set_start_method('spawn', force=True)

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Base class for Variational Layers
class BaseVariationalLayer_(nn.Module):
    def __init__(self):
        super().__init__()
        self._dnn_to_bnn_flag = False

    @property
    def dnn_to_bnn_flag(self):
        return self._dnn_to_bnn_flag

    @dnn_to_bnn_flag.setter
    def dnn_to_bnn_flag(self, value):
        self._dnn_to_bnn_flag = value

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):

        kl = torch.log(sigma_p) - torch.log(
            sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 *
                                                          (sigma_p**2)) - 0.5
        return kl.mean()

# Linear layer with reparameterization trick
class LinearReparameterization(BaseVariationalLayer_):
    def __init__(self,
                 in_features,
                 out_features,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True):

        super(LinearReparameterization, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init,  # mean of weight
        self.posterior_rho_init = posterior_rho_init, # variance of weight --> sigma = log (1 + exp(rho))
        self.bias = bias

        self.mu_weight = Parameter(torch.Tensor(out_features, in_features))
        self.rho_weight = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('eps_weight',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        self.register_buffer('prior_weight_mu',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        self.register_buffer('prior_weight_sigma',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_features))
            self.rho_bias = Parameter(torch.Tensor(out_features))
            self.register_buffer(
                'eps_bias',
                torch.Tensor(out_features),
                persistent=False)
            self.register_buffer(
                'prior_bias_mu',
                torch.Tensor(out_features),
                persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_features),
                                 persistent=False)
        else:
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)

        self.init_parameters()

    def init_parameters(self):
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        self.mu_weight.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
        self.rho_weight.data.normal_(mean=self.posterior_rho_init[0], std=0.1)
        if self.mu_bias is not None:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)
            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0],
                                       std=0.1)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        kl = self.kl_div(
            self.mu_weight,
            sigma_weight,
            self.prior_weight_mu,
            self.prior_weight_sigma)
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(self.mu_bias, sigma_bias,
                              self.prior_bias_mu, self.prior_bias_sigma)
        return kl

    def forward(self, input):
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        eps_weight = self.eps_weight.data.normal_()
        tmp_result = sigma_weight * eps_weight
        weight = self.mu_weight + tmp_result

        kl_weight = self.kl_div(self.mu_weight, sigma_weight,
                                    self.prior_weight_mu, self.prior_weight_sigma)

        bias = None
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = self.mu_bias + (sigma_bias * self.eps_bias.data.normal_())
            kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        out = F.linear(input, weight, bias)

        if self.mu_bias is not None:
            kl = kl_weight + kl_bias
        else:
            kl = kl_weight

        return out, kl

# Variational Feedforward Neural Network (last layer)
class vFNN(nn.Module):
    def __init__(self, in_features, hidden_size1, hidden_size2, out_features, 
                 prior_mean=0, prior_variance=0.5, posterior_rho_init=-4.0, bias=True):
        super(vFNN, self).__init__()

        # First fully connected layer (FNN)
        self.fc1 = nn.Linear(in_features, hidden_size1, bias=bias)
        
        # Second fully connected layer (FNN)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2, bias=bias)

        # Third (Final) Bayesian layer with reparameterization
        self.fc3 = LinearReparameterization(
            in_features=hidden_size2,
            out_features=out_features,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_rho_init=posterior_rho_init,
            bias=bias
        )

    def forward(self, x):
        # First layer with ReLU activation
        x = F.relu(self.fc1(x))

        # Second layer with ReLU activation
        x = F.relu(self.fc2(x))

        # Final Bayesian layer with uncertainty
        output, kl3 = self.fc3(x)

        # Only one KL divergence term from Bayesian layer
        return output, kl3

# Training process (with KL divergence handling)
def train_model(model, train_loader, num_epochs, reconstruction_loss_fn, optimizer, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), kl_schedule=None):
    model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()

        if kl_schedule == 'linear':
            kl_weight = 0.05 * (epoch / num_epochs)
        elif kl_schedule == 'sigmoid_growth':
            kl_weight = 0.05 / (1 + np.exp(-2 * (epoch - 0.7 * num_epochs))) + 0.005 # max / (1 + e^[-rate * (epoch - frac_training_w/o_KL*num_epochs)]) + min
        elif kl_schedule == 'sigmoid_decay':
            kl_weight = 0.05 / (1 + np.exp(2 * (epoch - 0.15 * num_epochs))) + 0.0005
        elif kl_schedule == 'step-up':
            if epoch/num_epochs <= 0.8:
                kl_weight = 0.005
            else:
                kl_weight = 0.05
        else:
            kl_weight = 0.05

        running_kl_train_loss = 0.0
        running_mse_train_loss = 0.0
        running_train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs, kl_loss = model(inputs)

            # Compute the reconstruction loss
            reconstruction_loss = reconstruction_loss_fn(outputs, targets)

            # Total loss (reconstruction + KL divergence)
            total_loss = reconstruction_loss + kl_weight * kl_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            running_kl_train_loss += kl_loss
            running_mse_train_loss += reconstruction_loss
            running_train_loss += total_loss.item()

        avg_kl_train_loss = running_kl_train_loss / len(train_loader)
        avg_mse_train_loss = running_mse_train_loss / len(train_loader)
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)


        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.5f}, Train MSE: {avg_mse_train_loss:.5f}')

    return train_losses

def get_dynamic_batch_size(train_size, min_size=1, max_size=64, factor=4):
    """Calculate batch size based on current training dataset size."""
    return max(min_size, min(max_size, train_size // factor))

# Prediction with uncertainty
def predict_with_uncertainty(model, test_loader, n_samples=100, scaler_y=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.eval()

    all_predictions = []
    true_values = []

    # Disable gradient computation for inference
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Store true values (used for comparison)
            true_values.append(targets.cpu().numpy())

            # Generate multiple predictions for each input
            predictions = []
            for _ in range(n_samples):
                outputs, _ = model(inputs)
                predictions.append(outputs.cpu().numpy())

            # Stack predictions across samples
            predictions = np.stack(predictions, axis=0)  # Shape: (n_samples, batch_size, num_outputs)
            all_predictions.append(predictions)

    # Concatenate predictions across all batches
    all_predictions = np.concatenate(all_predictions, axis=1)  # Shape: (n_samples, total_samples, num_outputs)
    true_values = np.concatenate(true_values, axis=0)  # Shape: (total_samples, num_outputs)

    # Apply inverse scaling if scaler_y is provided
    if scaler_y is not None:
        true_values = scaler_y.inverse_transform(np.array(true_values).reshape(-1, 1))
        all_predictions = np.array([scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten() for pred in all_predictions])

    return all_predictions, true_values

# Function to calculate mean and std
def calculate_mean_and_ci(predictions, confidence= 0.95):
    # Get num samples - need to divide by num samples when computing confidence intervals -> CI = y_hat +/- z * (std / sqrt(n))
    n_samples = predictions.shape[0]

    # Mean of the predictions
    mean_predictions = np.mean(predictions, axis=0)  # Shape: (total_samples, num_outputs)

    # Std. deviation of predictions
    std_predictions = np.std(predictions, axis=0)    # Shape: (total_samples, num_outputs)

    ci = std_predictions
    return mean_predictions, ci

# Metrics function
def evaluate_model(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2

##################################

           ###########

#################################

# Function to run a single experiment
def run_experiment(exp_id):
    print(f"Starting Experiment {exp_id}")
    
    # Data Preparation
    data = pd.read_csv("../data/chfAll.csv", header="infer")

    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values.reshape(-1, 1)  # Ensure Y is a column vector

    # Experiment parameters
    num_experiments = 50
    num_iterations = 500

    # r2 score
    r2_scores_active_all = np.zeros((num_experiments, num_iterations))
    r2_scores_random_all = np.zeros((num_experiments, num_iterations))
    
    # Split into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(X_train.shape)

    # Scaling the data
    Xscaler = MinMaxScaler()
    Yscaler = MinMaxScaler()
    X_train = Xscaler.fit_transform(X_train)
    y_train = Yscaler.fit_transform(Y_train)
    X_test = Xscaler.transform(X_test)
    y_test = Yscaler.transform(Y_test)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

    batch_size = 64  
    test = TensorDataset(X_test, y_test)  
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    # Set model architecture
    hidden1 = 100 
    hidden2 = 100 
    input_features = X_train.shape[1]
    output_features = 1  

    # Separate models for Active Learning and Random Sampling
    model_active = vFNN(input_features, hidden1, hidden2, output_features).to(device)
    model_random = vFNN(input_features, hidden1, hidden2, output_features).to(device)


    print("Device for model_active:", next(model_active.parameters()).device)
    print("Device for model_random:", next(model_random.parameters()).device)
    print("Device for X_train:", X_train.device)
    print("Device for y_train:", y_train.device)


    # define initial training and sampling parameters
    initial_train_size = 10
    sampling_size = 20

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


    epochs = 500
    lr = 0.001
    batch_size = get_dynamic_batch_size(len(X_initial_active))
    #batch_size = 128
    criterion = nn.MSELoss()
    # Define separate optimizers for both models
    optimizer_active = optim.Adam(model_active.parameters(), lr= lr, weight_decay=1e-5)
    optimizer_random = optim.Adam(model_random.parameters(), lr= lr, weight_decay=1e-5)
    
    # Results storage
    r2_random_scores = []
    r2_active_scores = []

    # Random Sampling Loop

    train_dataset_random = torch.utils.data.TensorDataset(X_initial_random, y_initial_random)
    train_loader_random = torch.utils.data.DataLoader(train_dataset_random, batch_size=batch_size, shuffle=True)
    train_model(model_random, train_loader_random, epochs, criterion, optimizer_random, kl_schedule="sigmoid_decay")

    for iteration in range(num_iterations):

        # Evaluate the model on the test data
        predictions, true_values = predict_with_uncertainty(model_random, test_loader, n_samples=100, scaler_y=Yscaler)
        mean_predictions, _ = calculate_mean_and_ci(predictions)
        rmse, r2 = evaluate_model(true_values, mean_predictions)
        r2_random_scores.append(r2)

        # Retrain the model with the updated training set
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
        #batch_size = 128
        # Retrain the random sampling model
        train_dataset_random = torch.utils.data.TensorDataset(X_initial_random, y_initial_random)
        train_loader_random = torch.utils.data.DataLoader(train_dataset_random, batch_size=batch_size, shuffle=True)
        train_model(model_random, train_loader_random, epochs, criterion, optimizer_random, kl_schedule="sigmoid_decay")
        print(f"Random Sampling - R2 Score after iteration {iteration + 1}: {r2:.4f}")


    ####### Active learning


    # Active learning loop


    r2_scores_active = []
    active_sample_sizes = []
    
    batch_size = get_dynamic_batch_size(len(X_initial_active))
    #batch_size = 128
    train_dataset_active = torch.utils.data.TensorDataset(X_initial_active, y_initial_active)
    train_loader_active = torch.utils.data.DataLoader(train_dataset_active, batch_size=batch_size, shuffle=True)
    train_model(model_active, train_loader_active, epochs, criterion, optimizer_active, kl_schedule="sigmoid_decay")


    for iteration in range(num_iterations):

        predictions, true_values = predict_with_uncertainty(model_active, test_loader, n_samples=100, scaler_y=Yscaler)
        mean_predictions, _ = calculate_mean_and_ci(predictions)
        rmse, r2 = evaluate_model(true_values, mean_predictions)
        r2_active_scores.append(r2)

        # **Batch-Based Pool Evaluation (Efficient)**
        pool_sample_size = min(500, len(X_pool_active))  # Limit to 500 samples for speed
        subset_indices = np.random.choice(len(X_pool_active), pool_sample_size, replace=False)
        subset_X_pool = X_pool_active[subset_indices]
        subset_y_pool = y_pool_active[subset_indices]

        val_loader = DataLoader(
            TensorDataset(torch.tensor(subset_X_pool, dtype=torch.float32), torch.tensor(subset_y_pool , dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=False,
        )

        predictions, true_values = predict_with_uncertainty(model_active, val_loader, n_samples=100, scaler_y=Yscaler)
        mean_predictions, ci = calculate_mean_and_ci(predictions)


        # Select the sample with the uncertainty
        top_index = subset_indices[np.argsort(ci)[-sampling_size:]]
        
        print(f"top_index shape: {top_index.shape}, values: {top_index}")
        X_new = X_pool_active[top_index]
        y_new = y_pool_active[top_index]
        
        # Debugging: Verify tensor shapes before concatenation
        print(f"X_initial_active shape: {X_initial_active.shape}")  # Expected: [num_samples, num_features]
        print(f"X_new shape: {X_new.shape}")  # Expected: [num_selected, num_features]
        print(f"y_initial_active shape: {y_initial_active.shape}")  # Expected: [num_samples, 1]
        print(f"y_new shape: {y_new.shape}")  # Expected: [num_selected, 1]
        

        # Update training data
        X_initial_active = torch.cat((X_initial_active, X_new), dim=0)
        y_initial_active = torch.cat((y_initial_active, y_new), dim=0)
        
        # Store the size of X_initial_active
        active_sample_sizes.append(len(X_initial_active))

        # Remove selected sample from pool
        mask = np.ones(len(X_pool_active), dtype=bool)
        mask[top_index] = False
        X_pool_active = X_pool_active[mask]
        y_pool_active = y_pool_active[mask]

        # Retrain the model
        batch_size = get_dynamic_batch_size(len(X_initial_active))
        #batch_size = 128
        train_dataset_active = torch.utils.data.TensorDataset(X_initial_active, y_initial_active)
        train_loader_active = torch.utils.data.DataLoader(train_dataset_active, batch_size=batch_size, shuffle=True)
        train_model(model_active, train_loader_active, epochs, criterion, optimizer_active, kl_schedule="sigmoid_decay")

        print(f"R2 Score after iteration {iteration + 1}: {r2:.4f}")
        
    return {
        "r2_random_scores": r2_random_scores,
        "r2_active_scores": r2_active_scores,
        "active_sample_sizes": active_sample_sizes
    }


if __name__ == "__main__":
    start_time = time.time()
    num_experiments = 50 ###############
        
    # Parallel execution
    with Pool(processes=min(cpu_count(), num_experiments)) as pool:
        results = pool.map(run_experiment, range(num_experiments))

    # Combine results
    r2_random_all = np.array([res["r2_random_scores"] for res in results])
    r2_active_all = np.array([res["r2_active_scores"] for res in results])

    # Compute mean and std for plotting
    r2_random_mean = r2_random_all.mean(axis=0)
    r2_random_std = r2_random_all.std(axis=0)
    r2_active_mean = r2_active_all.mean(axis=0)
    r2_active_std = r2_active_all.std(axis=0)

    # Save results
    results_df = pd.DataFrame({
        "Iteration": range(1, r2_random_mean.shape[0] + 1),
        "Random Mean R2": r2_random_mean,
        "Random Std R2": r2_random_std,
        "Active Mean R2": r2_active_mean,
        "Active Std R2": r2_active_std
    })
    results_df.to_csv("vi_last_50_10_20s_500it_500e_results.csv", index=False)

    print(f"Total Time Taken: {time.time() - start_time:.2f} seconds")

    active_sample_growth_all = np.array([res["active_sample_sizes"] for res in results])
    
    # Compute mean growth over all experiments
    active_sample_mean = active_sample_growth_all.mean(axis=0)
    
    # Save sample growth to CSV
    df_samples = pd.DataFrame({
        "Iteration": range(1, len(active_sample_mean) + 1),  # Ensure same length
        "X_initial_active_size": active_sample_mean
    })
    
    #df_samples.to_csv("active_sample_growth_viLAst.csv", index=False)
    # Plot the results
    iterations = range(1, r2_random_mean.shape[0] + 1)

    plt.figure(figsize=(10, 6))

    # Plot random sampling results
    plt.plot(iterations, r2_random_mean, label="Random Sampling Mean R2", color="red", linestyle="--", marker="s")
    plt.fill_between(iterations, r2_random_mean - r2_random_std, r2_random_mean + r2_random_std, color="red", alpha=0.2)

    # Plot active learning results
    plt.plot(iterations, r2_active_mean, label="Active Learning Mean R2", color="blue", linestyle="-", marker="o")
    plt.fill_between(iterations, r2_active_mean - r2_active_std, r2_active_mean + r2_active_std, color="blue", alpha=0.2)

    # Add titles and labels
    plt.title("R2 Score Comparison: Active Learning vs Random Sampling")
    plt.xlabel("Iteration")
    plt.ylabel("R2 Score")
    plt.legend()
    plt.grid(True)

    # Show and save the plot
    plt.savefig("../results/VI_last_50_10_20s_500it_500e_comparison_plot.png")
