import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class ChestImage64(Dataset):
    def __init__(self, csv_path, root_path, class_name,  transform=None):
        df = pd.read_csv(csv_path)
        self.annotation = df[df['Frontal/Lateral'] == class_name]
        self.root_path = root_path
        self.transform = transform

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_path, self.annotation.iloc[index]['Image_Path'])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.annotation.iloc[index]["Enlarged Cardiomediastinum"], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

