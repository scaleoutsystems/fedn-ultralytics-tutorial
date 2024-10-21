import os
import sys
from ultralytics import YOLO
import tempfile

from fedn.utils.helpers.helpers import save_metrics

from model import load_parameters
from data import load_data

def validate(in_model_path, out_json_path, data_yaml_path=None):
    """Validate YOLO model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_yaml_path: The path to the data file (YOLO dataset YAML file).
    :type data_yaml_path: str
    """

    test_data_yaml, test_data_length = load_data(None, step="test")
    train_data_yaml, train_data_length = load_data(None, step="train")
    # Load YOLOv8 model
    model = load_parameters(in_model_path)

    # Evaluate the model on both train and test datasets using YOLO's val() method
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_results = model.val(data=train_data_yaml, verbose=False, exist_ok=True,project=tmp_dir)
        test_results = model.val(data=test_data_yaml, verbose=False, exist_ok=True,project=tmp_dir)

    # Extract metrics from the results
    report = {
        "training_recall": train_results.results_dict['metrics/recall(B)'],
        "training_precision": train_results.results_dict['metrics/precision(B)'],
        "training_mAP50": train_results.results_dict['metrics/mAP50(B)'],  # mAP for training data
        "training_mAP50-95": train_results.results_dict['metrics/mAP50-95(B)'],  # mAP for training data
        "test_recall": test_results.results_dict['metrics/recall(B)'],
        "test_precision": test_results.results_dict['metrics/precision(B)'],
        "test_mAP50": test_results.results_dict['metrics/mAP50(B)'],  # mAP for testing data
        "test_mAP50-95": test_results.results_dict['metrics/mAP50-95(B)'],  # mAP for testing data
    }

    # Save JSON report (mandatory for FEDn)
    save_metrics(report, out_json_path)


if __name__ == "__main__":

    in_model_path = sys.argv[1]
    out_json_path = sys.argv[2]

    validate(in_model_path, out_json_path)