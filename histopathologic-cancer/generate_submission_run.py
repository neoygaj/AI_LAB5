# generate_submission_run.py
import sys, os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from predict import generate_submission
# from model import HistopathologicCNN
from model import DenseNet201Classifier

import torch

# ADDED FOR GPU
# ✅ Detect device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# If you saved a model previously, load it here
# model = HistopathologicCNN()
# model.load_state_dict(torch.load("outputs/model.pth"))

# ADDED for DenseNet201
model = DenseNet201Classifier(use_pretrained=False)



# ADDED FOR GPU
model.load_state_dict(torch.load("outputs/model.pth", map_location=device))
model = model.to(device)


# Optional: if you saved model weights to file
# model.load_state_dict(torch.load("outputs/model.pth"))

generate_submission(
    model=model,
    test_dir="data/test",
    sample_csv="data/sample_submission.csv",
    output_csv="outputs/submission.csv"
)
