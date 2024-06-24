import random

# Path to your data file
data_file_path = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/KITTI_MOTS/instances_txt/0020.txt'

# Read the data from the file
with open(data_file_path, 'r') as file:
    lines = file.readlines()

# Shuffle the data to ensure randomness
# random.shuffle(lines)

# Calculate the split index for 80% training data
split_index = int(0.8 * len(lines))

# Split the data
training_data = lines[:split_index]
evaluation_data = lines[split_index:]

# Paths for output files
train_output_path = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/KITTI_MOTS/training_instances/0020.txt'
eval_output_path = '/home/zahra/Documents/Projects/prototype/MOTR-codes/test_bbox/MOTR-main/data/Dataset/KITTI_MOTS/eval_instances/0020.txt'
# Write the training data to a file
with open(train_output_path, 'w') as file:
    file.writelines(training_data)

# Write the evaluation data to a file
with open(eval_output_path, 'w') as file:
    file.writelines(evaluation_data)

print("Data split into training (80%) and evaluation (20%) files.")
