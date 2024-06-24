import pandas as pd
import numpy as np

# Load the CSV data
file_path = 'prob_arrays/tubing_tubing_55s.csv'  # Update this path
data = pd.read_csv(file_path)

# Function to generate a single random probability within a range
def random_prob(low, high):
    return round(np.random.uniform(low, high), 8)

# Introduce variability by changing some low to high and vice versa at random indices
def add_variability(data, col, low_range, high_range, low_indices, high_indices, variability_count=5):
    for _ in range(variability_count):
        low_to_high_idx = np.random.choice(low_indices)
        high_to_low_idx = np.random.choice(high_indices)
        data.iloc[low_to_high_idx, col] = random_prob(*high_range)
        data.iloc[high_to_low_idx, col] = random_prob(*low_range)

# Modify the probabilities for column 1
low_indices_col1 = list(range(25, 107)) + list(range(274, len(data)))
high_indices_col1 = list(range(107, 263))
for i in range(len(data)):
    if i in high_indices_col1:
        data.iloc[i, 0] = random_prob(0.6, 0.9)
    elif i in low_indices_col1:
        data.iloc[i, 0] = random_prob(0.3, 0.55)
add_variability(data, 0, (0.3, 0.55), (0.6, 0.9), low_indices_col1, high_indices_col1)

# Modify the probabilities for column 2
low_indices_col2 = list(range(107, 263))
high_indices_col2 = list(range(25, 108)) + list(range(275, len(data)))
for i in range(len(data)):
    if i in high_indices_col2:
        data.iloc[i, 1] = random_prob(0.6, 0.9)
    elif i in low_indices_col2:
        data.iloc[i, 1] = random_prob(0.3, 0.55)
add_variability(data, 1, (0.3, 0.55), (0.6, 0.9), low_indices_col2, high_indices_col2)

# Save the modified data to a new CSV file
output_file_path = 'tubing_tubing_55s_modified_with_variability.csv'  # Update this path if needed
data.to_csv(output_file_path, index=False)

print(f"Modified CSV file saved as {output_file_path}")

# Display the modified rows for verification
print(data.loc[20:280])
