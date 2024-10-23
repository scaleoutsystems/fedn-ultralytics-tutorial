import os
import sys
from ultralytics import YOLO
import torch
from fedn.utils.helpers.helpers import save_metadata
import tempfile

from model import load_parameters, save_parameters
from data import load_data
import yaml

def train(in_model_path, out_model_path, data_yaml_path=None, epochs=10,batch_size=16):
    """Complete a model update using YOLOv8.

    Load model parameters from in_model_path (managed by the FEDn client),
    perform a model update, and write updated parameters
    to out_model_path (picked up by the FEDn server).

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_model_path: The path to save the output model to.
    :type out_model_path: str
    :param data_path: The path to the data file (YOLO dataset YAML file).
    :type data_path: str
    :param batch_size: The batch size to use.
    :type batch_size: int
    :param epochs: The number of epochs to train.
    :type epochs: int
    :param lr: The learning rate to use.
    :type lr: float
    """

    # Load YOLOv8 model
    model = load_parameters(in_model_path)

    # Load the client configuration 
    config_path = os.path.join(os.path.dirname(__file__), '../../client_config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        epochs = config.get('local_epochs', epochs)
        batch_size = config.get('batch_size', batch_size)
    else:
        print(f"Config file not found at {config_path}. Using default epochs ({epochs}) and batch size ({batch_size}).")

    data, length = load_data(data_yaml_path, step="train")
    # Train the model and remove the unnecessary files
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.train(data=data, epochs=epochs,batch=batch_size,verbose=False,exist_ok=True, project=tmp_dir)

    # Save the updated model to the output path
    save_parameters(model, out_model_path)

    # Metadata needed for aggregation server side
    metadata = {
        "num_examples": length,  # Get number of examples
    }

    # Save JSON metadata file (mandatory for FEDn)
    save_metadata(metadata, out_model_path)

if __name__ == "__main__":

    in_model_path = sys.argv[1]
    out_model_path = sys.argv[2]

    train(in_model_path, out_model_path)