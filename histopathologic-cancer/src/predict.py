# src/predict.py

import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import os
# from model import HistopathologicCNN
from model import DenseNet201Classifier


# def generate_submission(model, test_dir, sample_csv, output_csv='outputs/submission.csv'):
#     model.eval()

#     # ✅ Resize to match model's expected input shape
#     transform = transforms.Compose([
#         transforms.Resize((96, 96)),
#         transforms.ToTensor()
#     ])

#     df = pd.read_csv(sample_csv)
#     predictions = []

#     with torch.no_grad():
#         for img_id in df['id']:
#             img_path = os.path.join(test_dir, f"{img_id}.tif")
#             try:
#                 img = Image.open(img_path)
#                 img = transform(img).unsqueeze(0)  # Shape: (1, 3, 96, 96)
#                 output = model(img)
#                 prob = float(output.squeeze().item())
#             except Exception as e:
#                 print(f"❌ Failed to process {img_id}: {e}")
#                 prob = 0.0
#             predictions.append(prob)

#     df['label'] = predictions
#     df.to_csv(output_csv, index=False)
#     print(f"✅ Submission file saved to: {output_csv}")

# ADDED FOR GPU
def generate_submission(model, test_dir, sample_csv, output_csv='outputs/submission.csv'):
    import torchvision.transforms as transforms
    import pandas as pd
    import torch
    from PIL import Image
    import os

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device for prediction: {device}")

    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])

    df = pd.read_csv(sample_csv)
    predictions = []

    with torch.no_grad():
        for img_id in df['id']:
            img_path = os.path.join(test_dir, f"{img_id}.tif")
            try:
                img = Image.open(img_path)
                img = transform(img).unsqueeze(0).to(device)  # ✅ Move image to GPU
                output = model(img)
                prob = float(output.squeeze().item())
            except Exception as e:
                print(f"❌ Failed to process {img_id}: {e}")
                prob = 0.0
            predictions.append(prob)

    df['label'] = predictions
    df.to_csv(output_csv, index=False)
    print(f"✅ Submission file saved to: {output_csv}")



