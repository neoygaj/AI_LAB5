# src/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from model import HistopathologicCNN
from sklearn.model_selection import train_test_split
from tqdm import tqdm

## ADDED FOR GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")


# def get_dataloaders(train_csv, train_dir, batch_size=64):
#     df = pd.read_csv(train_csv)
#     train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'])

#     class HistopathologicDataset(torch.utils.data.Dataset):
#         def __init__(self, dataframe, dir_path, transform=None):
#             self.data = dataframe
#             self.dir = dir_path
#             self.transform = transform

#         def __len__(self):
#             return len(self.data)

#         def __getitem__(self, idx):
#             img_id = self.data.iloc[idx]['id']
#             label = self.data.iloc[idx]['label']
#             img = Image.open(os.path.join(self.dir, img_id + '.tif'))
#             if self.transform:
#                 img = self.transform(img)
#             return img, torch.tensor(label, dtype=torch.float32)

#     transform = transforms.Compose([
#         transforms.Resize((96, 96)),  # ✅ ensure size matches model input
#         transforms.ToTensor()
#     ])

#     train_dataset = HistopathologicDataset(train_df, train_dir, transform)
#     val_dataset = HistopathologicDataset(val_df, train_dir, transform)

#     return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), \
#            DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ADDED FOR AUGMENTATION
def get_dataloaders(train_csv, train_dir, batch_size=64):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from PIL import Image
    import os
    from torchvision import transforms
    import torch

    df = pd.read_csv(train_csv)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'])

    class HistopathologicDataset(torch.utils.data.Dataset):
        def __init__(self, dataframe, dir_path, transform=None):
            self.data = dataframe
            self.dir = dir_path
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_id = self.data.iloc[idx]['id']
            label = self.data.iloc[idx]['label']
            img = Image.open(os.path.join(self.dir, img_id + '.tif'))
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.float32)

    # ✅ Apply data augmentation only to training set
    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    # ✅ Clean validation transform
    val_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])

    train_dataset = HistopathologicDataset(train_df, train_dir, train_transform)
    val_dataset = HistopathologicDataset(val_df, train_dir, val_transform)

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), \
           DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


def train_model(epochs=5, save_path='outputs/model.pth'):
    # model = HistopathologicCNN()

    # ADDED FOR GPU
    model = HistopathologicCNN().to(device)


    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader, val_loader = get_dataloaders('data/train_labels.csv', 'data/train')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True, dynamic_ncols=True)


    # for imgs, labels in progress_bar:
    #     optimizer.zero_grad()
    #     outputs = model(imgs)
    #     loss = criterion(outputs.squeeze(), labels)
    #     loss.backward()
    #     optimizer.step()
    #     epoch_loss += loss.item()
    #     progress_bar.set_postfix(loss=loss.item())

    # ADDED FOR GPU
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())


    # print(f"📈 Epoch {epoch+1} - Avg Loss: {epoch_loss/len(train_loader):.4f}")

    # ADDED MEAN LOSS
    average_loss = epoch_loss / len(train_loader)
    tqdm.write(f"📈 Epoch {epoch+1}/{epochs} - Avg Loss: {average_loss:.4f}")

    import time
    start = time.time()
    # training loop
    end = time.time()
    tqdm.write(f"⏱️ Epoch {epoch+1} took {end - start:.2f} seconds")





    # ✅ Save model weights
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to: {save_path}")
    return model
