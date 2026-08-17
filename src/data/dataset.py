import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, METADATA_FILE, RAW_DIR

def get_transforms(split):
    """Return the transform pipeline for one split.

    get_transforms("train") -> Compose with 4 steps (includes flip)
    get_transforms("val")   -> Compose with 3 steps
    """
    steps = [transforms.Resize(IMAGE_SIZE)]

    if split == 'train':
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return transforms.Compose(steps)

class VarroaDataset(Dataset):
    def __init__(self, split=None, transform=None, videos=None):
        """Load metadata by split name or by explicit video list.

        VarroaDataset("train") -> dataset with 8225 rows
        VarroaDataset(videos=["2017-09-29_15-31-49"]) -> dataset with 128 rows
        labels 1 and 3 both mapped to class 1
        """
        assert (split is None) != (videos is None), "pass exactly one of split / videos"

        df = pd.read_csv(METADATA_FILE)
        if videos is not None:
            df = df[df["video"].isin(videos)]
        else:
            df = df[df["split"] == split]
        df = df.reset_index(drop=True)
        self.df = df
        # labels 1 and 3 are both infected per the authors' published counts
        self.df["binary_label"] = (self.df["label"] > 0).astype(int)
        self.transform = transform

    def __len__(self):
        """Return the number of images in this split.

        len(ds) -> 8225
        """
        return len(self.df)

    def __getitem__(self, index):
        """Return (image tensor, label) for one image.

        ds[0] -> (tensor of shape [3, 224, 224], 1)
        """
        row = self.df.iloc[index]
        path = RAW_DIR / row["path"]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["binary_label"])
        return image, label


def main():
    print(len(VarroaDataset("train", get_transforms("train"))))

    two = ["2017-09-01_20-00-17", "2017-09-29_15-31-49"]
    print(len(VarroaDataset(transform=get_transforms("val"), videos=two)))


if __name__ == "__main__":
    main()