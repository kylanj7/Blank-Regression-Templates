# Content from OpenCV University bootcamp

import os
import numpy as np
import random
import requests
import subprocess
from zipfile import ZipFile, BadZipFile

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch import optim

plt.rcParams["figure.figsize"] = (15, 6)
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 12

BOLD = "\033[1m"
END = "\033[0m"

def system_config(SEED_VALUE=42, package_list=None):
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    
    def is_running_in_colab():
        return 'COLAB_GPU' in os.environ
        
    def is_running_in_kaggle():
        return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
        
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        
        if is_running_in_colab() or is_running_in_kaggle():
            if package_list:
                print("Installing required packages...")
                subprocess.run(['pip', 'install'] + package_list.split(), check=True)
            
        DEVICE = torch.device('cuda')
        print("Device:", DEVICE)
        GPU_AVAILABLE = True
        
        torch.cuda.manual_seed(SEED_VALUE)
        torch.cuda.manual_seed_all(SEED_VALUE)
        
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("Using Apple Silicon GPU")
        DEVICE = torch.device("mps")
        print("Device:", DEVICE)
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        GPU_AVAILABLE = True
        
        torch.mps.manual_seed(SEED_VALUE)
        torch.use_deterministic_algorithms(True)
    
    else:
        print('Using CPU')
        DEVICE = torch.device('cpu')
        print("Device:", DEVICE)
        GPU_AVAILABLE = False
        
        if is_running_in_colab() or is_running_in_kaggle():
            if package_list:
                print('Installing required packages...')
                subprocess.run(['pip', 'install'] + package_list.split(), check=True)
            print("Note: Change runtime type to GPU for better performance.")
    
        torch.use_deterministic_algorithms(True)
        
    return str(DEVICE), GPU_AVAILABLE

package_list = "torchinfo"
DEVICE, GPU_AVAILABLE = system_config(package_list=package_list)

from torchinfo import summary

def download_file(url, save_name):
    response = requests.get(url, stream=True)
    with open(save_name, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
    print(f"Downloaded: {save_name}")
    
def unzip(zip_file_path):
    try:
        with ZipFile(zip_file_path, 'r') as z:
            z.extractall("./")
            print(f"Extracted: {os.path.splitext(zip_file_path)[0]}\n")
    except FileNotFoundError:
        print("File not found")
    except BadZipFile:
        print("Invalid or corrupt zip file")
    except Exception as e:
        print(f"Error occurred: {e}")

URL = ""
archive_name = ""
zip_name = f"./{archive_name}.zip"

if not os.path.exists(archive_name):
    download_file(URL, zip_name)
    unzip(zip_name)

column_names = []

raw_dataset = pd.read_csv(
    os.path.join(archive_name, ""),
    names=column_names, 
    na_values="?",
    comment="\t",
    sep=" ",
    skipinitialspace=True,
)

dataset = raw_dataset.copy()
print(dataset.head())

dataset.isna().sum()
dataset = dataset.dropna()
dataset.shape

train_dataset = dataset.sample(frac=0.8, random_state=42)
test_dataset = dataset.drop(train_dataset.index)

print(train_dataset.shape)
print(test_dataset.shape)

X_train = train_dataset.copy()
X_test = test_dataset.copy()

y_train = X_train.pop('')
y_test = X_test.pop('')

mean_feature = np.mean(X_train[''])
std_feature = np.std(X_train[''])

print("\nFor feature column:")
print("Mean:", mean_feature)
print("Std:", std_feature)
print("Count:", X_train.shape[0])

X_train["feature_scaled"] = (X_train[""] - mean_feature) / std_feature
X_test["feature_scaled"] = (X_test[""] - mean_feature) / std_feature

class Regressor(nn.Module):
    def __init__(self, in_features=1, out_features=1):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        
    def forward(self, x):
        return self.linear(x)

model = Regressor(in_features=1, out_features=1)

batch_size = 1
summary(model, input_size=(batch_size, 1,), device="cpu", col_names=("input_size", "output_size", "num_params"))

optimizer = optim.Adam(model.parameters(), lr=0.1)
criterion = torch.nn.L1Loss()

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

loss_curve_train = []
loss_curve_eval = []

X_train_tensor = torch.from_numpy(X_train_split["feature_scaled"].values).reshape(-1, 1).to(torch.float32)
y_train_tensor = torch.from_numpy(y_train_split.values).reshape(-1, 1).to(torch.float32)

X_val_tensor = torch.from_numpy(X_val_split["feature_scaled"].values).reshape(-1, 1).to(torch.float32)
y_val_tensor = torch.from_numpy(y_val_split.values).reshape(-1, 1).to(torch.float32)

def plot_loss(loss_curve_train, loss_curve_eval):
    plt.figure(figsize=(15, 5))
    plt.plot(loss_curve_train, label="Loss")
    plt.plot(loss_curve_eval, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.show()
    return

for epoch in range(500):
    model.train()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_curve_train.append(loss.detach().item())
    model.eval()
    with torch.no_grad():
        output = model(X_val_tensor)
        
    loss = criterion(output, y_val_tensor)
    loss_curve_eval.append(loss.item())

plot_loss(loss_curve_train, loss_curve_eval)

x = torch.linspace(X_train[""].min(), X_train[""].max(), len(X_train[""]))

model.eval()
with torch.no_grad():
    y = model((x.view(-1, 1) - mean_feature) / std_feature)
    
def plot_predictions(x, y):
    plt.figure(figsize=(15, 5))
    plt.scatter(
        (X_train["feature_scaled"] * std_feature) + mean_feature,
        y_train,
        label="Data",
        color="green",
        alpha=0.5,
    )
    plt.plot(x, y, color="k", label="Predictions")
    plt.xlabel("")
    plt.ylabel("")
    plt.legend()
    plt.grid(True)
    plt.show()
    return

plot_predictions(x, y)
