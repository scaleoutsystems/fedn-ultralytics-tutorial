import os
import shutil
import random

def create_splits(data_dir, output_dir, num_splits):
    subdirs = ['train', 'valid','test']
    image_ext = '.jpg'
    image_ext2 = '.JPG' 
    label_ext = '.txt'

    for subdir in subdirs:
        # Get a list of all image files and ensure corresponding labels exist
        image_files = [f for f in os.listdir(os.path.join(data_dir, subdir, 'images')) if (f.endswith(image_ext)) or (f.endswith(image_ext2))]
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
        
        # Calculate the size of each split
        split_size = len(base_filenames) // num_splits
        
        # Create the necessary directories for each split in the new output directory
        for i in range(num_splits):
            split_dir_images = os.path.join(output_dir, f'split_{i+1}', subdir, 'images')
            split_dir_labels = os.path.join(output_dir, f'split_{i+1}', subdir, 'labels')
            os.makedirs(split_dir_images, exist_ok=True)
            os.makedirs(split_dir_labels, exist_ok=True)

            start = i * split_size
            if i == num_splits - 1:  # last split gets the remainder
                end = len(base_filenames)
            else:
                end = start + split_size

            split_files = base_filenames[start:end]
            
            for base_name in split_files:
                src_image_file = os.path.join(data_dir, subdir, 'images', base_name + image_ext)
                src_label_file = os.path.join(data_dir, subdir, 'labels', base_name + label_ext)

                dst_image_file = os.path.join(split_dir_images, base_name + image_ext)
                dst_label_file = os.path.join(split_dir_labels, base_name + label_ext)

                shutil.copy(src_image_file, dst_image_file)
                shutil.copy(src_label_file, dst_label_file)

        # print path of the split directories
        print(f"Split directories created in {output_dir} for {subdir}.")

if __name__ == "__main__":
    data_directory = 'datasets/african-wildlife'  # The dataset path
    output_directory = 'datasets/african_wildlife_partitions'  # Directory to store the new splits
    num_splits = 2  # Replace with the desired number of splits
    
    create_splits(data_directory, output_directory, num_splits)