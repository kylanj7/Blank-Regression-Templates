# This content is directly from OpenCV University 

import os
import random
import math
import time
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim


import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt  # one of the best graphics library for python
import matplotlib.animation as animation

plt.rcParams["figure.figsize"] = (15, 6)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

block_plot = False
bold = f"\033[1m"
reset = f"\033[0m"

def system_config(SEED_VALUE=42, package_list=None):
    """
    Configures the system environment for PyTorch-based operations.

    Args:
        SEED_VALUE (int): Seed value for random number generation. Default is 42.
        package_list (str): String containing a list of additional packages to install
        for Google Colab or Kaggle. Default is None.

    Returns:
        tuple: A tuple containing the device name as a string and a boolean indicating GPU availability.
    """

    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)

    def is_running_in_colab():
        return 'COLAB_GPU' in os.environ

    def is_running_in_kaggle():
        return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

    #--------------------------------
    # Check for the availability GPUs.
    #--------------------------------
    if torch.cuda.is_available():
        print('Using CUDA GPU')

        # This section for installing packages required by Colab.
        if is_running_in_colab() or is_running_in_kaggle():
            print('Installing required packages...')
            !pip install {package_list}

        # Set the device to the first CUDA device.
        DEVICE = torch.device('cuda')
        print("Device: ", DEVICE)
        GPU_AVAILABLE = True

        torch.cuda.manual_seed(SEED_VALUE)
        torch.cuda.manual_seed_all(SEED_VALUE)

        # Performance and deterministic behavior.
        torch.backends.cudnn.enabled = True       # Provides highly optimized primitives for DL operations.
        torch.backends.cudnn.deterministic = True # Insures deterministic even when above cudnn is enabled.
        torch.backends.cudnn.benchmark = False    # Setting to True can cause non-deterministic behavior.

    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print('Using Apple Silicon GPU')

        # Set the device to the Apple Silicon GPU Metal Performance Shader (MPS).
        DEVICE = torch.device("mps")
        print("Device: ", DEVICE)
        # Environment variable that allows PyTorch to fall back to CPU execution
        # when encountering operations that are not currently supported by MPS.
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        GPU_AVAILABLE = True

        torch.mps.manual_seed(SEED_VALUE)
        torch.use_deterministic_algorithms(True)

    else:
        print('Using CPU')
        DEVICE = torch.device('cpu')
        print("Device: ", DEVICE)
        GPU_AVAILABLE = False

        if is_running_in_colab() or is_running_in_kaggle():
            print('Installing required packages...')
            !pip install {package_list}
            print('Note: Change runtime type to GPU for better performance.')

        torch.use_deterministic_algorithms(True)

    return str(DEVICE), GPU_AVAILABLE
    
    # Additional packages required for Google Colab or Kaggle.
package_list = "torchinfo"

DEVICE, GPU_AVAILABLE = system_config(package_list=package_list)

from torchinfo import summary

def generate_data(mean_0=[4.0, 20.0], stddev_0=[1.0, 1.0],
                  mean_1=[5.5, 23.0], stddev_1=[0.6, 0.8],
                  num_points_0=200, num_points_1=200):

    class_0_dist = dist.Normal(loc=torch.tensor(mean_0), scale=torch.tensor(stddev_0))
    class_1_dist = dist.Normal(loc=torch.tensor(mean_1), scale=torch.tensor(stddev_1))

    class_0_points = class_0_dist.sample((num_points_0,))
    class_1_points = class_1_dist.sample((num_points_1,))

    return class_0_points, class_1_points
    
class_0_points, class_1_points = generate_data()

plt.figure(figsize=(20, 10))
plt.scatter(class_0_points[:, 0], class_0_points[:, 1], color="b", alpha=0.5, label="Class:0")
plt.scatter(class_1_points[:, 0], class_1_points[:, 1], color="r", alpha=0.5, label="Class:1")
plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.xlim([0, 10])
plt.ylim([16, 28])
plt.grid(True)
plt.show(block=block_plot)
plt.close()

def prepare_data(class_0_points, class_1_points):

    label_zero = torch.zeros(class_0_points.shape[0], dtype=torch.float)
    label_one  = torch.ones(class_1_points.shape[0], dtype=torch.float)

    labels = torch.cat([label_zero, label_one], dim=0).unsqueeze(dim=1)
    data_points = torch.cat([class_0_points, class_1_points], dim=0)

    print(f"Data points size: {data_points.shape}")
    print(f"Label size: {labels.shape}")
    return data_points, labels
    
    X_train, y_train = prepare_data(class_0_points, class_1_points)
    
def normalize_data(data, mean, std):
    data_norm = (data - mean)/std
    return data_norm
    
    print(f"Train Data shape: {X_train.shape}")

mean = X_train.mean(0)
std = X_train.std(0)

print(f"Mean:   {mean}")
print(f"Std:    {std}")

X_train = normalize_data(X_train, mean=mean, std=std)


model = nn.Linear(in_features=2, out_features=1)

print(summary(model, input_size=(1,2)))

def train(model, data_points, labels, epochs, optimizer, batch_size=10):

    num_batches = math.ceil(len(labels)/ batch_size)

    loss_history = []
    acc_history = []

    model = model.to(DEVICE)
    model.train()

    # Trainig time measurement.
    t_begin = time.time()
    for epoch_idx in range(epochs):

        # Clear cell outputs at the start of each epoch.
        clear_output()

        # Shuffle data at the start of each epochs.
        shuffled_indices = torch.randperm(len(labels))
        shuffled_data = data_points[shuffled_indices]
        shuffled_labels = labels[shuffled_indices]

        step_loss = 0
        step_accuracy = 0

        for batch_idx in range(num_batches):

            start_ind = batch_idx*batch_size
            end_ind   = (batch_idx + 1)*batch_size

            batched_data = shuffled_data[start_ind:end_ind].to(DEVICE)
            batched_targets = shuffled_labels[start_ind:end_ind].to(DEVICE)

            # Set the weight gradients to zero for every min-batch to avoid
            # gradient accumulation.
            optimizer.zero_grad()

            # Forward pass through the model.
            logits = model(batched_data)

            # Compute Loss.
            loss = F.binary_cross_entropy_with_logits(logits, batched_targets)

            # Compute gradients using backpropagation.
            loss.backward()

            # Update model weights.
            optimizer.step()

            # Convert the output logits to probabilities.
            predictions = logits.sigmoid()

            # Batch Loss.
            step_loss+= loss.item()* batched_data.shape[0]

            # Batch Acuuracy.
            step_accuracy+= ((predictions > 0.5).int().cpu() == batched_targets.cpu()).sum()

        epoch_loss = float(step_loss / len(labels))
        epoch_acc = float(step_accuracy/ len(labels))

        print(f"{f'{bold}[ Epoch: {epoch_idx+1} ]{reset}':=^80}")

        train_loss_stat = f"{bold}Loss: {epoch_loss:.4f}{reset}"
        train_acc_stat = f"{bold}Accuracy: {epoch_acc:.4f}{reset}"

        print(f"\n{train_loss_stat:<30}{train_acc_stat}")

        print(f"{'='*72}\n")

        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)


    print(f"Total time taken: {(time.time() - t_begin):.2f}s")
    return model, loss_history, acc_history
    

