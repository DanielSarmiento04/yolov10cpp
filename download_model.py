# Import libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve
import argparse


# Define the function to download the model
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '--model', 
    choices=['yolov10n', 'yolov10s', 'yolov10m', 'yolov10b', 'yolov10l', 'yolov10x'],
    default='yolov10n', 
    help='Model to download'
)

args = parser.parse_args()

def download_model(model):

    url = "https://github.com/THU-MIG/yolov10/releases/download/v1.1/" + model + ".pt"
    # Downloading zip file using urllib package.
    print("Downloading the model...")
    urlretrieve(url, model + ".pt")
    print("Model downloaded successfully!")


# Call the function to download the model
download_model(args.model)