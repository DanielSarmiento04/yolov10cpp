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
parser.add_argument('--model', choices=['yolov9-c', 'yolov9-e'], default='yolov9-c', help='Model to download')

args = parser.parse_args()

def download_model(model):

    url = "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/" + model + ".pt"
    # Downloading zip file using urllib package.
    print("Downloading the model...")
    urlretrieve(url, model + ".pt")
    print("Model downloaded successfully!")


# Call the function to download the model
download_model(args.model)