# macbook-machine-learning-setup
An easy [setup guide for macbook](https://www.imranabdullah.com/2024-03-25/Deep-Learning-environment-set-up-MacBook-M3-with-Miniconda-TensorFlow-and-PyTorch) as of 2025.

This one is for Macbooks with M3 chipset to use MPS from PyTorch.

You should *not install both tensorflow and pytorch in the same virtual environment* or you will summon a trully dark evil spirit.

## Variables
* Virtual environment: deeplearning-env
* Python version: 3.10.13
* Folder: Machine Learning

## Setup
### Requirements
```bash
  mkdir "Machine Learning" && cd "Machine Learning"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  brew install miniconda
  conda create -n deeplearning-env python=3.10.13
  conda activate deeplearning-env
  pip3 install notebook
```

### Option: TensorFlow
```bash
  conda install -c apple tensorflow-deps
  pip install tensorflow-macos tensorflow-metal
```

**Verify installation**
1. Launch `jupyter notebook` and create a new Python 3 file.
2. Paste the below in there and run.

```python
  import sys
  import keras
  import tensorflow as tf
  import platform
  
  print(f"Python Platform: {platform.platform()}")
  print(f"Tensor Flow Version: {tf.__version__}")
  print(f"Keras Version: {keras.__version__}")
  gpu = len(tf.config.list_physical_devices("GPU")) > 0
  print("GPU is", "available" if gpu else "NOT AVAILABLE")
```

### Option: PyTorch
```bash
  pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

**Verify installation**
1. Launch `jupyter notebook` and create a new Python 3 file.
2. Paste the below in there and run.

```python
  import torch
  print(f"PyTorch version: {torch.__version__}")
  print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
  print(f"Is MPS available? {torch.backends.mps.is_available()}")
  device = "mps" if torch.backends.mps.is_available() else "cpu"
  print(f"Using device: {device}")
```

**Enable MPS if it's available**
Add the below as a second codeblock in your notebook.
```python
  device = "mps" if torch.backends.mps.is_available() else "cpu"
  x = torch.rand(size=(3, 4)).to(device)
```
