import torch
from rich import print
from villa.dataloaders.Stage1Dataset import Stage1Dataset

ATTRIBUTES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "red",
    "yellow",
    "green",
    "blue",
    "purple",
    "rectangle",
    "circle",
    "small",
    "medium",
    "large",
]


class DocMNIST_Stage1Dataset(Stage1Dataset):
    def __init__(self, split: str, data_dir: str):
        """
        Initialize DocMNIST dataloader for Stage 1.

        Parameters:
            split (str): Indicates the split (e.g. "train", "val")
            data_dir (str): Directory where DocMNIST data is stored
        """

        super(DocMNIST_Stage1Dataset, self).__init__(split, data_dir)

        self.examples.insert(
            self.examples.shape[1],
            "attr_binary_vec",
            self.encode_attributes(ATTRIBUTES),
        )

        self.region_embs = self.get_region_embs()

        print(f"=> Split {self.split} includes {self.__len__()} samples")

    def __len__(self):
        return self.examples.shape[0]

    def __getitem__(self, idx: int):
        example = self.examples.iloc[[idx]]
        image_id = str(example["image_id"].values[0])

        out_dict = {
            "idx": torch.tensor(idx),
            "image_id": image_id,
        }

        out_dict.update(self.get_inputs(example, image_id))
        return out_dict


if __name__ == "__main__":
    pass
