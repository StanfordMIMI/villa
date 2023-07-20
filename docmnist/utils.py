import numpy as np
import pandas as pd
import torchvision
import torch
import matplotlib.cm as cm
from skimage import draw
from tqdm.contrib.concurrent import process_map
import sparse


DIGIT_TO_TEXT = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}


def load_mnist_data(data_dir: str):
    """
    Load the MNIST dataset and create train, val, and test splits.

    Parameters:
        data_dir (str): Directory for storing MNIST dataset
    Returns:
        data (dict): Contains MNIST image data associated with train, val, and test splits
        targets (dict): Contains MNIST digit labels associated with each image
    """
    data = {}
    targets = {}

    mnist_train = torchvision.datasets.MNIST(
        root=f"{data_dir}/mnist", train=True, download=True
    )
    mnist_test = torchvision.datasets.MNIST(
        root=f"{data_dir}/mnist", train=False, download=True
    )

    idx_divider = int(0.2 * len(mnist_train))
    all_indices = np.arange(len(mnist_train))
    np.random.shuffle(all_indices)

    train_indices = all_indices[idx_divider:]
    data["train"] = mnist_train.data[train_indices].unsqueeze(1).repeat(1, 3, 1, 1)
    targets["train"] = mnist_train.targets[train_indices]

    val_indices = all_indices[:idx_divider]
    data["val"] = mnist_train.data[val_indices].unsqueeze(1).repeat(1, 3, 1, 1)
    targets["val"] = mnist_train.targets[val_indices]

    data["test"] = mnist_test.data.unsqueeze(1).repeat(1, 3, 1, 1)
    targets["test"] = mnist_test.targets

    return data, targets


def get_colors():
    """
    Get list of valid colors and color names.

    Returns:
        color_names (list): Indicates names for each color
        colors (list): Contains RGB values for each color
    """

    cmap = cm.get_cmap("hsv")
    cmap_vals = np.arange(0, 1, step=1 / 5)
    colors = []
    for val in cmap_vals:
        rgb = cmap(val)[:3]
        rgb = [int(np.float(x)) for x in np.array(rgb) * 255]
        colors.append(rgb)
    color_names = ["red", "yellow", "green", "blue", "purple"]
    return color_names, colors


def apply_color(region: torch.Tensor, prop: list = [0.2, 0.2, 0.2, 0.2, 0.2]):
    """
    Apply a randomly selected color to a digit

    Parameters:
        region (torch.Tensor): Contains pixel data associated with an MNIST digit
        prop (list): Indicates the probability of selecting each color
    Returns:
        region (torch.Tensor): Contains pixel data associated with an MNIST digit (with color applied)
        color_name (str): Indicates the color applied to the region
    """
    color_names, colors = get_colors()
    color_ix = np.random.choice(np.arange(len(color_names)), 1, p=prop).item()
    color = torch.tensor(colors[color_ix], dtype=torch.uint8)

    pixels = torch.where(region[0, :, :] >= 120)
    region[:, pixels[0], pixels[1]] = color.unsqueeze(1).repeat(1, len(pixels[0]))

    return region, color_names[color_ix]


def apply_shape(region: torch.Tensor):
    """
    Insert a randomly selected shape with a randomly selected size in a region

    Parameters:
        region (torch.Tensor): Contains pixel data associated with an MNIST digit
    Returns:
        region (torch.Tensor): Contains pixel data associated with an MNIST digit (with shape and size applied)
        shape (str): Indicates the shape inserted in the region
        size (str): Indicates the size of the inserted shape
    """
    all_sizes = ["small", "medium", "large"]
    all_shapes = ["circle", "rectangle", None]

    shape = np.random.choice(all_shapes)
    size = np.random.choice(all_sizes)
    if shape == "circle":
        if size == "small":
            rad = 1
        elif size == "medium":
            rad = 3
        else:
            rad = 5
        row, col = draw.circle_perimeter(
            np.random.randint(5, 20), np.random.randint(5, 20), rad
        )
        region[:, row, col] = torch.tensor([255, 255, 255], dtype=torch.uint8).reshape(
            -1, 1
        )
    elif shape == "rectangle":
        if size == "small":
            rad = 1
        elif size == "medium":
            rad = 4
        else:
            rad = 7
        start = (np.random.randint(5, 20), np.random.randint(5, 20))
        end = (start[0] + rad, start[1] + rad)
        row, col = draw.rectangle_perimeter(start, end, shape=(28, 28), clip=True)
        region[:, row, col] = torch.tensor([255, 255, 255], dtype=torch.uint8).reshape(
            -1, 1
        )
    else:
        shape = None
        size = None

    return region, shape, size


def get_caption(digit_label: int, color_label: str, shape_label: str, size_label: str):
    """
    Generate a caption given a set of digit, color, shape, and size attributes

    Parameters:
        digit_label (int): Label associated with the MNIST digit
        color_label (str): Label associated with the digit color
        shape_label (str): Label associated with the shape
        size_label (str): Label associated with the size of the shape
    Returns:
        all_captions (list): List of captions generated based on the input attributes
    """

    def get_color_caption(c):
        templates = [
            f"the color is {c}",
            f"the digit appears to be {c}",
            f"there is a {c} image",
            f"the image is {c}",
        ]
        return np.random.choice(templates)

    def get_digit_caption(d):
        templates = [
            f"the image shows a {DIGIT_TO_TEXT[d]}",
            f"the digit appears to be {DIGIT_TO_TEXT[d]}",
            f"there is an image showing a {DIGIT_TO_TEXT[d]}",
            f"the number is a {DIGIT_TO_TEXT[d]}",
        ]
        return np.random.choice(templates)

    def get_shape_caption(s):
        if s is None:
            return ""

        templates = [
            f"the shape is a {s}",
            f"the shape appears to be a {s}",
            f"there is a {s}",
            f"the image has a {s}",
        ]
        return np.random.choice(templates)

    def get_size_caption(s):
        if s is None:
            return ""

        templates = [
            f"the shape size is {s}",
            f"the size of the shape is {s}",
            f"the shape is {s}",
        ]
        return np.random.choice(templates)

    color_caption = get_color_caption(color_label)
    digit_caption = get_digit_caption(digit_label)
    shape_caption = get_shape_caption(shape_label)
    size_caption = get_size_caption(size_label)

    all_captions = np.unique(
        [color_caption] + [digit_caption] + [shape_caption] + [size_caption]
    )
    all_captions = [x for x in all_captions.tolist() if x]
    np.random.shuffle(all_captions)
    return all_captions


def _save_image(id: int, image: np.array, out_dir: str):
    """
    Save pixel data associated with a single image to disk

    Parameters:
        id (int): Image index in dataset
        image (np.array): Pixel data associated with a DocMNIST image
        out_dir (str): Directory for storing images
    Returns:
        img_fp (str): Filepath for saved image
    """
    img_fp = out_dir / f"{id}.npz"
    sparse.save_npz(img_fp, sparse.COO(image))
    return str(img_fp)


def save_images(images: list, out_dir: str, num_processes: int = 10):
    """
    Save pixel data associated with a list of images to disk

    Parameters:
        images (list): List of pixel data associated with DocMNIST images
        out_dir (str): Directory for storing images
        num_processes (int): Number of parallel workers
    Returns:
        paths (list): Filepath for all saved images
    """
    print(f"Saving images to {out_dir}")
    paths = process_map(
        _save_image,
        np.arange(len(images)),
        images,
        [out_dir] * len(images),
        max_workers=num_processes,
        chunksize=1,
    )
    return paths


def save_ann(train_data: dict, val_data: dict, img_paths: list, out_dir: str):
    """
    Save annotations associated with each DocMNIST image

    Parameters:
        train_data (dict): Contains metadata associated with each training image
        val_data (dict): Contains metadata associated with each validation image
        img_paths (list): Contains filepaths for all saved images
        out_dir (str): Directory for storing annotations
    """
    df = {}
    num_train = len(train_data["img"])
    num_val = len(val_data["img"])

    # Image properties
    df["image_id"] = np.arange(num_train + num_val)
    df["image_size"] = [[84, 84]] * (num_train + num_val)
    df["image_filepath"] = img_paths

    # Region properties
    df["region_coord"] = train_data["boxes"] + val_data["boxes"]
    df["num_regions"] = [len(x) for x in (train_data["boxes"] + val_data["boxes"])]

    # Splits and text
    df["split"] = ["train"] * num_train + ["val"] * num_val
    df["text"] = train_data["text"] + val_data["text"]

    # Metadata
    df["digit_label"] = [
        [DIGIT_TO_TEXT[y] for y in x["digit"]]
        for x in train_data["metadata"] + val_data["metadata"]
    ]
    df["color_label"] = [
        x["color"] for x in train_data["metadata"] + val_data["metadata"]
    ]
    df["shape_label"] = [
        x["shape"] for x in train_data["metadata"] + val_data["metadata"]
    ]
    df["size_label"] = [
        x["size"] for x in train_data["metadata"] + val_data["metadata"]
    ]

    df = pd.DataFrame(df)

    # Ground truth mappings between regions and attributes
    mappings = df.apply(
        lambda x: list(zip(x.digit_label, x.color_label, x.shape_label, x.size_label)),
        axis=1,
    )
    df["reg_to_attr"] = mappings.apply(
        lambda x: [list(filter(lambda item: item is not None, a)) for a in x]
    )

    # Save to disk
    print(f"Saving annotations to {out_dir}/annotations.feather")
    df.to_feather(out_dir / "annotations.feather")
