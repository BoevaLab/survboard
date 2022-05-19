import os
import sys
import argparse
import logging
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
from skorch.net import NeuralNet
from survival_benchmark.python.utils.hyperparameters import MODEL_FACTORY, LOSS_FACTORY

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=str, help="Path to the folder containing data.")
parser.add_argument("training_params", type=str, help="Path to the parameters needed for training in JSON format.")
parser.add_argument("results_path", type=str, help="Path where results should be saved")
parser.add_argument("model_path", type=str, help="Path where best model is saved")
parser.add_argument("model_name", type=str, help="Name of model being trained.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(data_dir, training_params, results_path, model_path, model_name):
    # setup logging
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(results_path, f"{model_name}.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(f"{model_name}")
    logger.setLevel(logging.DEBUG)

    clrs = sns.color_palette("Set2", 2)
    fig, ax = plt.subplots(constrained_layout=True)
    fig.suptitle(f"Train and Validation Loss of {model_name}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")

    save_here = os.path.join(results_path, "results")
    model = None
    if model is not None:
        del model

    with open(training_params, "r") as readjson:
        train_params = json.load(readjson)

    # init params
    neural_model = MODEL_FACTORY[train_params.get("neural_model", "cox")]
    loss_fn = LOSS_FACTORY[train_params.get("loss_fn", "cox")]
    optimiser = torch.optim.Adam
    skorch_args = train_params["skorch_args"]
    # other params like time intervals, etc

    # init dataset
    train_dataset = 0
    test_dataset = 0
    # init skorch model
    model_skorch = NeuralNet(module=neural_model, criterion=loss_fn, optimizer=optimiser, **skorch_args)

    # fit and predict
    model_skorch.fit(train_dataset)
    model_skorch.predict(test_dataset)

    # save and plot


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.data_dir, args.training_params, args.results_path, args.model_path, args.model_name)
