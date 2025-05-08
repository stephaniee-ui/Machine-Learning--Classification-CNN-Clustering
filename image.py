import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import logging
import matplotlib.pyplot as plt
from PIL import Image

class ImageDataset:
    def __init__(self, datapath: str, use_augmentation=True, model="CNN", fft=False):
        self.datapath = datapath
        self.data = pd.read_csv(datapath)
        self.fft_flag = fft
        self.preprocess()

        if model == "CNN":
            self.transform = transforms.Compose([
                transforms.Resize((512, 768)),  # Resize images to the most commonly occurring sizes in the dataset
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip(),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.6210, 0.5891, 0.5232],  # Normalize images using the dataset's mean and standard deviation
                                    std=[0.2602, 0.2560, 0.2772])
            ]) if use_augmentation else None
            self.test_transform = transforms.Compose([
                transforms.Resize((512, 768)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.6210, 0.5891, 0.5232],
                                    std=[0.2602, 0.2560, 0.2772])
            ])
        elif model == "ViT":
            self.transform = transforms.Compose([
                transforms.Resize((100, 100)),  # Resize images to the most commonly occurring sizes in the dataset
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip(),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.6210, 0.5891, 0.5232],  # Normalize images using the dataset's mean and standard deviation
                                    std=[0.2602, 0.2560, 0.2772])
            ]) if use_augmentation else None
            self.test_transform = transforms.Compose([
                transforms.Resize((100, 100)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.6210, 0.5891, 0.5232],
                                    std=[0.2602, 0.2560, 0.2772])
            ])

        if self.transform is None:
            self.transform = self.test_transform

        if use_augmentation:
            self.data = pd.concat([self.data, self.data], ignore_index=True)
            self.aug_flags = [False] * (len(self.data) // 2) + [True] * (len(self.data) // 2)
        else:
            self.aug_flags = [False] * len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        img_path = self.data.loc[idx]['file_name']
        img_label = self.data.loc[idx]['label']
        try:
            img = Image.open(img_path).convert("RGB")  # Ensure RGB format
            use_aug = self.aug_flags[idx]
            if use_aug:
                img = self.transform(img)
            else:
                img = self.test_transform(img)
            if self.fft_flag:
                img = np.transpose(img, (1, 2, 0))
                img = img.numpy()
                r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                gray_img = 0.299 * r + 0.587 * g + 0.114 * b
                _fft = np.fft.fft2(gray_img)
                _fft_shifted = np.fft.fftshift(_fft)
                magnitude_spectrum = 20 * np.log(np.abs(_fft_shifted) + 1e-5)
                return magnitude_spectrum.flatten(), img_label
            else:
                return img, img_label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

    def preprocess(self):
        self.data["file_name"] = self.data["file_name"].apply(
            lambda x:
                os.path.join(os.path.dirname(self.datapath), "train_data", os.path.basename(x))
        )

    def check(self):
        # Check missing values in each column
        missing_values = self.data.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        print(f'There are {len(missing_values)} missing values')

        # Check the occurances of label 0 and 1
        label_counts = self.data['label'].value_counts()
        label_0 = label_counts[0]
        label_1 = label_counts[1]
        print(f"{label_0} images with label 0, and {label_1} with label 1.")

        # Check if all filenames end with ".jpg"
        all_jpg = self.data["file_name"].str.endswith(".jpg").all()
        if all_jpg:
            print("All filenames end with .jpg")

    def show_image(self, transform=False, fft=False):
        neg_images = self.data[self.data['label'] == 0]['file_name'].tolist()
        pos_images = self.data[self.data['label'] == 1]['file_name'].tolist()
        img_paths = {
            "0": neg_images[np.random.randint(0, len(neg_images))],
            "1": pos_images[np.random.randint(0, len(pos_images))]
        }

        for lb, img_path in img_paths.items():
            img = Image.open(img_path).convert("RGB")  # Ensure RGB format
            if self.transform and transform:
                img = self.transform(img)  # Resize + Normalize
            else:
                img = transforms.functional.to_tensor(img)
            print(f"Label: {lb}, shape: {img.shape}")

            if fft:
                img = np.transpose(img, (1, 2, 0))
                img = img.numpy()
                r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                gray_img = 0.299 * r + 0.587 * g + 0.114 * b
                _fft = np.fft.fft2(gray_img)
                _fft_shifted = np.fft.fftshift(_fft)
                magnitude_spectrum = 20 * np.log(np.abs(_fft_shifted) + 1e-5)

                plt.axis('off')
                plt.subplot(1, 2, 1)
                plt.title('Original Image (gray scale)')
                plt.imshow(gray_img, cmap='gray')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.title('FFT Magnitude Spectrum')
                plt.imshow(magnitude_spectrum, cmap='jet')
                plt.axis('off')

                plt.tight_layout()
                plt.show()
            else:
                plt.imshow(img.permute(1, 2, 0))
                plt.axis('off')
                plt.show()

    # Deprecated
    def data_split(self, train_size: float, test_size: float, validation_size: float, state: int=42):
        raise RuntimeError("Deprecated function. You need to load data from separate csv files")
        assert(1 == train_size + test_size + validation_size)
        self.train_df, self.test_df = train_test_split(self.data, test_size=0.1, random_state=state, stratify=self.data['label'])
        self.train_df, self.val_df = train_test_split(self.train_df, test_size=1.0/9.0, random_state=state, stratify=self.train_df['label'])
        logging.info(f"Training size: {len(self.train_df)}, testing size: {len(self.test_df)}, validation size: {len(self.val_df)}")
        return self.train_df, self.test_df, self.val_df

    class TorchWrapper:
        def __init__(self, ds, df):
            self.paths = df["file_name"].tolist()
            self.labels = df["label"].tolist()
            self.dataset = ds
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx: int):
            return self.dataset[idx]
            """
            img_path = self.paths[idx]
            img_label = self.labels[idx]
            try:
                img = Image.open(img_path).convert("RGB")  # Ensure RGB format
                if self.dataset.transform:
                    img = self.dataset.transform(img)  # Resize + Normalize
                return img, img_label
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                return None
            """

    def get_loader(self, **kwargs):
        loader_args = {
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True,
            "batch_size": 32,
        }
        loader_args.update(kwargs)
        # from copy import deepcopy
        # test_args = deepcopy(loader_args)
        # test_args["batch_size"] = 1

        return DataLoader(self.TorchWrapper(self, self.data), **loader_args)
