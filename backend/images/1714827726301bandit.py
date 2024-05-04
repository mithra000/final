import sys
import pandas as pd
import numpy as np

def parse_arguments():
    # Parse command line arguments
    if len(sys.argv) != 5:
        print("ERROR: Not enough or too many input arguments.")
        sys.exit(1)
    
    file_name = sys.argv[1]
    epsilon = float(sys.argv[2]) if 0 <= float(sys.argv[2]) <= 1 else 0.3
    train_percentage = int(sys.argv[3]) if 0 <= int(sys.argv[3]) <= 50 else 50
    threshold = int(sys.argv[4])
    
    return file_name, epsilon, train_percentage, threshold

def load_data(file_name):
    # Load and process input data
    data = pd.read_csv(file_name)
    return data

def epsilon_greedy(data, epsilon, train_percentage, threshold):
    # Training phase
    train_size = int(len(data) * (train_percentage / 100))
    train_data = data.iloc[:train_size]
    success_counts = np.zeros(len(data.columns))
    total_counts = np.zeros(len(data.columns))
    
    for i, row in train_data.iterrows():
        if np.random.rand() < epsilon:
            arm = np.random.randint(len(data.columns))
        else:
            arm = np.argmax(success_counts / total_counts)
        
        if row[arm] < threshold:
            success = 1
        else:
            success = 0
        
        total_counts[arm] += 1
        success_counts[arm] += success
    
    # Testing phase
    test_data = data.iloc[train_size:]
    chosen_arm = np.argmax(success_counts / total_counts)
    success_percentage = sum(test_data[chosen_arm] < threshold) / len(test_data) * 100
    
    # Report results
    print(f"epsilon: {epsilon}")
    print(f"Training data percentage: {train_percentage} %")
    print(f"Success threshold: {threshold}")
    print("Success probabilities:")
    for i, col in enumerate(data.columns):
        print(f"P({col}) = {success_counts[i] / total_counts[i]}")
    print(f"Bandit [{data.columns[chosen_arm]}] was chosen to be played for the rest of data set.")
    print(f"{data.columns[chosen_arm]} Success percentage: {success_percentage}")

def main():
    # Parse command line arguments
    file_name, epsilon, train_percentage, threshold = parse_arguments()

    # Load and process input data
    data = load_data(file_name)

    # Implement epsilon-greedy algorithm
    epsilon_greedy(data, epsilon, train_percentage, threshold)

if __name__ == "__main__":
    main()
