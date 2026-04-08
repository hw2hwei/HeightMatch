import os
import random

# Set the root folder path
full_folder_path = '/home/Datasets/RSSeg/INRIA'

# Output txt file that stores all file paths
full_file = 'full.txt'

# Open the txt file for writing
with open(full_file, 'w') as f:
    # Recursively traverse the folder
    for root, dirs, files in os.walk(full_folder_path):
        for file in files:
            # Save paths only if the directory contains 'image'
            if 'image' in root.lower() and 'fourier' not in root.lower() and 'org' not in root.lower():
                # Get the full image path
                image_path = os.path.join(root, file)

                # Convert to a path relative to the root folder
                image_path = os.path.relpath(image_path, start=full_folder_path)

                # Build the corresponding label and height paths
                mask_path = image_path.replace('image', 'label')
                height_path = image_path.replace('image', 'height').replace('tif', 'png')

                # Write the triplet to the txt file
                f.write(image_path + ' ' + mask_path + ' ' + height_path + '\n')

print(f'Image paths have been saved to {full_file}')

# Split the dataset with a 7:1:2 ratio
with open(full_file, 'r') as f:
    lines = f.readlines()

# Shuffle the data
random.shuffle(lines)

# Compute split indices
total_lines = len(lines)
train_end = int(0.7 * total_lines)
val_end = train_end + int(0.1 * total_lines)

# Split into train, val, and test sets
train_lines = lines[:train_end]
val_lines = lines[train_end:val_end]
test_lines = lines[val_end:]

# Save train.txt, val.txt, and test.txt
with open('train.txt', 'w') as train_file:
    train_file.writelines(train_lines)

with open('val.txt', 'w') as val_file:
    val_file.writelines(val_lines)

with open('test.txt', 'w') as test_file:
    test_file.writelines(test_lines)

print("The dataset has been split into train.txt, val.txt, and test.txt with a 7:1:2 ratio")

# Define the percentage list
list_of_percentages = [0.1, 0.2, 0.5, 1, 2, 50]

# Split the training set according to each percentage
for percentage in list_of_percentages:
    # Compute the number of labeled samples to keep
    labeled_count = int((percentage / 100) * len(train_lines))

    # Randomly select labeled samples
    random.shuffle(train_lines)
    labeled_lines = train_lines[:labeled_count]
    unlabeled_lines = train_lines[labeled_count:]

    # Create the subfolder for the current percentage
    subfolder = f'{percentage}'
    os.makedirs(subfolder, exist_ok=True)

    # Save labeled.txt
    with open(os.path.join(subfolder, 'labeled.txt'), 'w') as labeled_file:
        labeled_file.writelines(labeled_lines)

    # Save unlabeled.txt
    with open(os.path.join(subfolder, 'unlabeled.txt'), 'w') as unlabeled_file:
        unlabeled_file.writelines(unlabeled_lines)

    print(f'Data for {percentage}% has been saved to the subfolder "{subfolder}"')