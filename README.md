# PyTorch Image Classifier

## Overview
This project shows how to build a basic image classifier using Python. The classifier is trained on the CIFAR-10 dataset, which is a large set of images from 10 different classification categories.

## Virtual environment
This project assumes a virtual environment is being used. In Python, this environment can be created by typing from the command line: `python3 -m venv venv`. The virtual environment, venv, can then be activated from the command line by using `source bin/venv/activate`.

## Requirements
After installing requirements, try running from the command line:
`python -c "import torch; print(torch.backen.mps.is_available())"`. The output result should be `True`.

## Files
Here are files associated with this repository
1. main.py - Python script for imports, loading data, defining the model, and doing training and evaluation.
2. predict.py - Script used for getting model image classification predictions for a set of local images.
3. requirements.txt - Set of required packages for running this code.

## Dataset
This image classifier is set to be trained on the CIFAR 10 dataset. This dataset can be obtained from:

[CIFAR 10 Dataset](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

Citation: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

## Training the Classifier
To run the training for the classifier, after activating the virtual enviroment and installed required packages, from the command line use: `python main.py`

## Running predictions
To run predictions on your own images, save them to a local destination, and reference this desintation within the script `predictions.py`
