import argparse
import better_exceptions
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset
from imgaug import augmenters as iaa


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.OneOf([
                iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=0.1 * 255)),
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0)))
                ]),
            iaa.Affine(
                rotate=(-20, 20), mode="edge",
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}
            ),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
            iaa.GammaContrast((0.3, 2)),
            iaa.Fliplr(0.5),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        return img

def f(x):return x

class FaceDataset(Dataset):
    def __init__(self, data_dir, data_type, img_size=224, augment=False, age_stddev=1.0):
        assert(data_type in ("train", "valid", "test"))
        csv_path = Path(data_dir).joinpath(f"gt_avg_{data_type}.csv")
        img_dir = Path(data_dir).joinpath(data_type)
        self.img_size = img_size
        self.augment = augment
        self.age_stddev = age_stddev

        if augment:
            self.transform = ImgAugTransform()
        else:
            self.transform = f

        self.x = []
        self.y = []
        self.std = []
        df = pd.read_csv(str(csv_path))
        ignore_path = Path(__file__).resolve().parent.joinpath("ignore_list.csv")
        ignore_img_names = list(pd.read_csv(str(ignore_path))["img_name"].values)

        for _, row in df.iterrows():
            img_name = row["file_name"]

            if img_name in ignore_img_names:
                continue

            img_path = img_dir.joinpath(img_name + "_face.jpg")
            assert(img_path.is_file())
            self.x.append(str(img_path))
            self.y.append(row["apparent_age_avg"])
            self.std.append(row["apparent_age_std"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_path = self.x[idx]
        age = self.y[idx]

        if self.augment:
            age += np.random.randn() * self.std[idx] * self.age_stddev

        img = cv2.imread(str(img_path), 1)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.transform(img).astype(np.float32)
        return torch.from_numpy(np.transpose(img, (2, 0, 1))), np.clip(round(age), 0, 100)

class AugmentationFunctions:
    def __init__(self):
        self.functions = [
        ]
        self.functions.append(iaa.AdditiveGaussianNoise(scale=0.1 * 255))
        self.functions.append(iaa.GaussianBlur(sigma=1))
        self.functions.append(iaa.GaussianBlur(sigma=2))
        self.functions.append(iaa.AddToHueAndSaturation(value=-5, per_channel=True))
        self.functions.append(iaa.AddToHueAndSaturation(value=5, per_channel=True))
        self.functions.append(iaa.GammaContrast(1.5))
        self.functions.append(iaa.Fliplr())
        self.functions.append(iaa.Sequential([iaa.AdditiveGaussianNoise(scale=0.2 * 255),
            iaa.GaussianBlur(sigma=3),
            iaa.AddToHueAndSaturation(value=1, per_channel=True),
            iaa.GammaContrast(0.5),
            iaa.Fliplr()]))
        self.functions.append(iaa.Sequential([iaa.AdditiveGaussianNoise(scale=0.1 * 255),
            iaa.GaussianBlur(sigma=0),
            iaa.AddToHueAndSaturation(value=1, per_channel=True),
            iaa.GammaContrast(1.5)]))
        
        self.nb = 10

    def __call__(self, img):
        res = [torch.from_numpy(np.transpose(np.array(img), (2, 0, 1))).float()]
        savimg = img
        for f in self.functions:
            img = f.augment_image(np.array(savimg))
            res.append(torch.from_numpy(np.transpose(img, (2, 0, 1))).float())
        return torch.stack(res)


class TTADataset(Dataset):
    def __init__(self, data_dir, data_type, img_size=224, age_stddev=1.0):
        assert(data_type in ("train", "valid", "test"))
        csv_path = Path(data_dir).joinpath(f"gt_avg_{data_type}.csv")
        img_dir = Path(data_dir).joinpath(data_type)
        self.img_size = img_size
        self.age_stddev = age_stddev

        ag = AugmentationFunctions()

        self.transform = ag

        self.x = []
        self.y = []
        self.std = []
        df = pd.read_csv(str(csv_path))
        ignore_path = Path(__file__).resolve().parent.joinpath("ignore_list.csv")
        ignore_img_names = list(pd.read_csv(str(ignore_path))["img_name"].values)

        for _, row in df.iterrows():
            img_name = row["file_name"]

            if img_name in ignore_img_names:
                continue

            img_path = img_dir.joinpath(img_name + "_face.jpg")
            assert(img_path.is_file())
            self.x.append(str(img_path))
            self.y.append(row["apparent_age_avg"])
            self.std.append(row["apparent_age_std"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_path = self.x[idx]
        age = self.y[idx]

        img = cv2.imread(str(img_path), 1)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.transform(img)
        return img, np.clip(round(age), 0, 100)


class FaceDatasetMultiTask(Dataset):
    def __init__(self, data_dir, data_type, img_size=224, augment=False, age_stddev=1.0, name_variables = [], variables_to_int = []):
        assert(data_type in ("train", "valid", "test"))
        csv_path = Path(data_dir).joinpath(f"gt_avg_{data_type}.csv")
        csv_path_2 = Path(data_dir).joinpath(f"allcategories_{data_type}.csv")

        img_dir = Path(data_dir).joinpath(data_type)
        self.img_size = img_size
        self.augment = augment
        self.age_stddev = age_stddev

        if augment:
            self.transform = ImgAugTransform()
        else:
            self.transform = f

        self.x = []
        self.y_age = []
        self.y = []
        self.std = []
        df = pd.read_csv(str(csv_path))
        df_2 = pd.read_csv(str(csv_path_2))
        ignore_path = Path(__file__).resolve().parent.joinpath("ignore_list.csv")
        ignore_img_names = list(pd.read_csv(str(ignore_path))["img_name"].values)

        for (_, row), (_, row_2) in zip(df.iterrows(), df_2.iterrows()):
            img_name = row["file_name"]

            if img_name in ignore_img_names:
                continue

            img_path = img_dir.joinpath(img_name + "_face.jpg")
            assert(img_path.is_file())

            self.x.append(str(img_path))
            self.y_age.append(row["apparent_age_avg"])
            self.y.append([hash(row_2[name]) for (name, hash) in zip(name_variables, variables_to_int)])

            self.std.append(row["apparent_age_std"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_path = self.x[idx]
        y = self.y[idx].copy()
        y_age = self.y_age[idx]

        if self.augment:
            y_age += np.random.randn() * self.std[idx] * self.age_stddev
        y_age = np.clip(round(y_age), 0, 100)

        img = cv2.imread(str(img_path), 1)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.transform(img).astype(np.float32)
        return torch.from_numpy(np.transpose(img, (2, 0, 1))), (y_age, torch.LongTensor(y))

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    dataset = FaceDataset(args.data_dir, "train")
    print("train dataset len: {}".format(len(dataset)))
    dataset = FaceDataset(args.data_dir, "valid")
    print("valid dataset len: {}".format(len(dataset)))
    dataset = FaceDataset(args.data_dir, "test")
    print("test dataset len: {}".format(len(dataset)))


if __name__ == '__main__':
    main()
