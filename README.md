# Hybrid ML Pipeline for Weak Supervision + Joint Learning

## Features
- Feature extraction from vibration/image data
- Weak supervision using labeling functions
- Joint learning with MLP
- Subset selection for efficient labeling

## Tech Stack
- Python
- PyTorch
- Flask (for inference app)

## How to Run

### Install dependencies
pip install -r requirements.txt

### Generate Labeled Data
python datagen.py

### Generate Unlabeled Data
python unlabeled_datagen.py

### Train baseline
python train_baseline.py

### Train joint model
python train_joint_learning.py

### Run app
python main.py
