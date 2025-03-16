import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool, cpu_count

# Handy function to determine different validation metrics
def calc_metric(Y, Yhat):
    MSE = np.mean((Y - Yhat) ** 2, 0)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(Y - Yhat), 0)
    Ybar = np.mean(Y, 0)
    R2 = 1 - np.sum((Y - Yhat) ** 2, 0) / np.sum((Y - Ybar) ** 2, 0)
    return pd.DataFrame([MSE, RMSE, MAE, R2], index=['MSE', 'RMSE', 'MAE', 'R2'])

# Load data
data = pd.read_csv("../data/chfAll.csv")
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1, 1)  

# Experiment parameters
num_experiments = 5
num_iterations = 500
initial_train_size = 100
sampling_size = 20

# Function to run a single experiment in parallel
def run_experiment(exp):

    print(f"Running experiment {exp+1}/{num_experiments}")

    # Split into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Scale Data
    Xscaler = MinMaxScaler()
    Yscaler = MinMaxScaler()
    X_train = Xscaler.fit_transform(X_train)
    X_test = Xscaler.transform(X_test)
    Y_train = Yscaler.fit_transform(Y_train).ravel()  # Flatten for compatibility
    Y_test = Yscaler.transform(Y_test).ravel()
    
    # Initialize Random Forest models
    model_active = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model_random = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    
    # Initial Training Pool for Active Learning
    initial_indices = np.random.choice(len(X_train), initial_train_size, replace=False)
    sampling_indices = np.setdiff1d(np.arange(len(X_train)), initial_indices)
    
    X_initial_active = X_train[initial_indices]
    Y_initial_active = Y_train[initial_indices]
    X_pool_active = X_train[sampling_indices]
    Y_pool_active = Y_train[sampling_indices]
    
    X_initial_random = X_train[initial_indices]
    Y_initial_random = Y_train[initial_indices]
    X_pool_random = X_train[sampling_indices]
    Y_pool_random = Y_train[sampling_indices]
    
    # Function to evaluate the model using calc_metric
    def evaluate_model(model, X, Y):
        predictions = model.predict(X)
        return calc_metric(Y, predictions), predictions
    
    # Active Learning Loop
    r2_scores_active = []
    
    for iteration in range(num_iterations):  
        print(f"Active Learning Iteration {iteration+1}")
        model_active.fit(X_initial_active, Y_initial_active)
    
        train_metrics, _ = evaluate_model(model_active, X_initial_active, Y_initial_active)
        test_metrics, predictions_active = evaluate_model(model_active, X_test, Y_test)
        r2_scores_active.append(test_metrics.loc['R2'][0])
    
        # Evaluate pool samples for selection (batch-based for efficiency)
        pool_sample_size = min(500, len(X_pool_active))  # Limit to 500 samples for performance
        subset_indices = np.random.choice(len(X_pool_active), pool_sample_size, replace=False)
        subset_X_pool = X_pool_active[subset_indices]
        subset_Y_pool = Y_pool_active[subset_indices]
    
        mape_values = [
            calc_metric(np.array([subset_Y_pool[i]]), np.array([model_active.predict(subset_X_pool[i:i+1])[0]])).loc['MAE'][0]
            for i in range(len(subset_X_pool))
        ]
    
        top_indices = subset_indices[np.argsort(mape_values)[-sampling_size:]]  # Select worst-performing sample
        X_new = X_pool_active[top_indices]
        Y_new = Y_pool_active[top_indices]
    
        # Update the training pool
        X_initial_active = np.vstack((X_initial_active, X_new))
        Y_initial_active = np.hstack((Y_initial_active, Y_new))
    
        # Remove the selected samples from the pool
        mask = np.ones(len(X_pool_active), dtype=bool)
        mask[top_indices] = False
        X_pool_active = X_pool_active[mask]
        Y_pool_active = Y_pool_active[mask]
    
    # Random Sampling Loop
    r2_scores_random = []
    
    for iteration in range(num_iterations):
        print(f"Random Sampling Iteration {iteration+1}")
        model_random.fit(X_initial_random, Y_initial_random)
    
        train_metrics, _ = evaluate_model(model_random, X_initial_random, Y_initial_random)
        test_metrics, predictions_random = evaluate_model(model_random, X_test, Y_test)
    
        r2_scores_random.append(test_metrics.loc['R2'][0])
    
        # Randomly select samples
        random_indices = np.random.choice(len(X_pool_random), sampling_size, replace=False)
        X_new_random = X_pool_random[random_indices]
        Y_new_random = Y_pool_random[random_indices]
    
        # Update the training pool
        X_initial_random = np.vstack((X_initial_random, X_new_random))
        Y_initial_random = np.hstack((Y_initial_random, Y_new_random))
    
        # Remove selected samples from the pool
        mask_random = np.ones(len(X_pool_random), dtype=bool)
        mask_random[random_indices] = False
        X_pool_random = X_pool_random[mask_random]
        Y_pool_random = Y_pool_random[mask_random]


    return r2_scores_active, r2_scores_random

# Parallel processing using multiprocessing
if __name__ == '__main__':
    num_cores = min(cpu_count(), num_experiments)
    with Pool(num_cores) as p:
        results = p.map(run_experiment, range(num_experiments))

    # Collect results
    r2_scores_active_all, r2_scores_random_all = zip(*results)
    r2_scores_active_all = np.array(r2_scores_active_all)
    r2_scores_random_all = np.array(r2_scores_random_all)

    # Compute mean and standard deviation of R2 scores across experiments
    r2_mean_active = np.mean(r2_scores_active_all, axis=0)
    r2_std_active = np.std(r2_scores_active_all, axis=0)

    r2_mean_random = np.mean(r2_scores_random_all, axis=0)
    r2_std_random = np.std(r2_scores_random_all, axis=0)

    # Save R2 scores and standard deviation per iteration
    r2_df = pd.DataFrame({
        "Iteration": range(1, num_iterations + 1),
        "Mean R2 Active Learning": r2_mean_active,
        "Std R2 Active Learning": r2_std_active,
        "Mean R2 Random Sampling": r2_mean_random,
        "Std R2 Random Sampling": r2_std_random
    })
    r2_df.to_csv("../results/RF_r2_scores_per_iteration.csv", index=False)

    # Plot R2 scores with standard deviation
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_iterations + 1), r2_mean_active, marker='o', linestyle='-', label='Active Learning', color='blue')
    plt.fill_between(range(1, num_iterations + 1), r2_mean_active - r2_std_active, r2_mean_active + r2_std_active, color='blue', alpha=0.2)

    plt.plot(range(1, num_iterations + 1), r2_mean_random, marker='s', linestyle='--', label='Random Sampling', color='red')
    plt.fill_between(range(1, num_iterations + 1), r2_mean_random - r2_std_random, r2_mean_random + r2_std_random, color='red', alpha=0.2)

    #plt.title('RF Active Learning vs Random Sampling')
    plt.xlabel('Iteration')
    plt.ylabel('Mean R2 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig("../results/RF_active_learning_vs_random_r2_mean_std.png")
