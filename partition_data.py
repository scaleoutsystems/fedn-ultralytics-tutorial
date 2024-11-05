import os
import shutil
import random
import sys

def find_file_with_extension(directory, base_name, extensions):
    """Helper function to find a file with one of the specified extensions."""
    for ext in extensions:
        file_path = os.path.join(directory, base_name + ext)
        if os.path.exists(file_path):
            return file_path
    raise FileNotFoundError(f"No file with base name {base_name} and extensions {extensions} in {directory}")

def create_splits(data_dir, dataset_name, num_splits, seed=42):
    # Seed for reproducibility
    random.seed(seed)
    
    subdirs = ['train', 'valid', 'test']
    image_ext = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG')
    label_ext = ('.txt', '.TXT')

    for subdir in subdirs:
        # Get a list of all image files and ensure corresponding labels exist
        image_files = [f for f in os.listdir(os.path.join(data_dir, subdir, 'images')) if f.endswith(image_ext)]
        label_files = [f for f in os.listdir(os.path.join(data_dir, subdir, 'labels')) if f.endswith(label_ext)]
        
        # Create a set of base filenames (without extensions)
        image_bases = {os.path.splitext(f)[0] for f in image_files}
        label_bases = {os.path.splitext(f)[0] for f in label_files}
        
        # Check for any mismatches between image and label files
        if image_bases != label_bases:
            raise ValueError(f"Mismatch between image and label files in {subdir}. Ensure each image has a corresponding label with the same base filename. Missing labels: {image_bases - label_bases}, missing images: {label_bases - image_bases}")
        
        # Convert the base filenames back to a list and shuffle
        base_filenames = list(image_bases)
        random.shuffle(base_filenames)
        
        # Split the list of filenames into approximately equal parts
        split_files = [base_filenames[i::num_splits] for i in range(num_splits)]
        
        # Create directories and copy files into each partition
        for i, files in enumerate(split_files):
            split_dir = os.path.join('datasets', f"{dataset_name}_split_{i+1}")
            split_dir_images = os.path.join(split_dir, subdir, 'images')
            split_dir_labels = os.path.join(split_dir, subdir, 'labels')
            os.makedirs(split_dir_images, exist_ok=True)
            os.makedirs(split_dir_labels, exist_ok=True)

            for base_name in files:
                src_image_file = find_file_with_extension(os.path.join(data_dir, subdir, 'images'), base_name, image_ext)
                src_label_file = find_file_with_extension(os.path.join(data_dir, subdir, 'labels'), base_name, label_ext)

                dst_image_file = os.path.join(split_dir_images, os.path.basename(src_image_file))
                dst_label_file = os.path.join(split_dir_labels, os.path.basename(src_label_file))

                shutil.copy(src_image_file, dst_image_file)
                shutil.copy(src_label_file, dst_label_file)

    # Print path of each split directory after all subdirs have been processed
    for i in range(num_splits):
        print(f"Split directory created for {dataset_name} in datasets/{dataset_name}_split_{i+1}.")

if __name__ == "__main__":
    # Check if the number of arguments is correct
    if len(sys.argv) != 3:
        print("Usage: python partition_data.py <dataset_name> <num_splits>")
        sys.exit(1)
    
    # Parse the arguments
    dataset_name = sys.argv[1]  # The dataset name provided by the user
    data_directory = os.path.join('datasets', dataset_name)  # Construct the dataset path
    
    try:
        num_splits = int(sys.argv[2])  # The number of splits (convert to int)
    except ValueError:
        print("The number of splits must be an integer.")
        sys.exit(1)
    
    # Call the function with the user-supplied arguments
    create_splits(data_directory, dataset_name, num_splits)