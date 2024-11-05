import yaml
import os

# Load server configuration
with open('client/global_config.yaml', 'r') as config_file:
    server_config = yaml.safe_load(config_file)

# Extract configuration values
model_size = server_config.get("model_size", "nano").lower()
num_classes = server_config.get("num_classes", 1)
class_names = server_config.get("class_names", [f"Class {i}" for i in range(num_classes)])

# Dataset paths from config
train_path = server_config.get("train", "fed_dataset/train/images")
val_path = server_config.get("val", "fed_dataset/valid/images")
test_path = server_config.get("test", "fed_dataset/test/images")

# Paths for model files
model_folder = "client/yolov8models"
model_file = os.path.join(model_folder, f"{model_size}.yaml")
output_model_file = "client/model.yaml"
output_data_file = "client/data.yaml"

# Generate model.yaml with 'nc' on line 4
if not os.path.exists(model_file):
    print(f"Error: Model file '{model_file}' does not exist in '{model_folder}'.")
else:
    # Read the model file content as a list of lines
    with open(model_file, 'r') as file:
        model_lines = file.readlines()

    # Insert the nc line at line 4 (index 3)
    model_lines.insert(3, f"nc: {num_classes} # Number of classes from global_config\n")

    # Write the modified content to the model.yaml file
    with open(output_model_file, 'w') as output_file:
        output_file.writelines(model_lines)

    print(f"'{output_model_file}' created successfully with nc: {num_classes} based on '{model_file}'")

# Generate data.yaml with paths, nc, and names
data_content = {
    "train": train_path,
    "val": val_path,
    "test": test_path,
    "nc": num_classes,
    "names": {i: class_name for i, class_name in enumerate(class_names)}
}

# Write the data.yaml content
with open(output_data_file, 'w') as data_file:
    yaml.dump(data_content, data_file, sort_keys=False)

print(f"'{output_data_file}' created successfully with paths from global_config.yaml, nc: {num_classes}, and indexed class names.")