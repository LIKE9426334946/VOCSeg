import random

from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode, RandomCrop
from torchvision.transforms import functional as F


def pad_pair(image, mask, pad_right, pad_bottom, ignore_index):
    """只在右侧和底部补齐，mask 的补齐部分不参与损失和指标计算。"""
    padding = [0, 0, pad_right, pad_bottom]
    image = F.pad(image, padding, fill=0)
    mask = F.pad(mask, padding, fill=ignore_index)
    return image, mask


class SegmentationTransform:
    def __init__(self, dataset_config, training):
        self.config = dataset_config
        self.training = training

    def __call__(self, image, mask):
        cfg = self.config

        if self.training:
            # 图像和标签使用相同的缩放比例，保持原来的长宽比。
            scale = random.uniform(*cfg["scale_range"])
            width, height = image.size
            size = [max(1, round(height * scale)), max(1, round(width * scale))]
            image = F.resize(image, size, interpolation=InterpolationMode.BILINEAR)
            mask = F.resize(mask, size, interpolation=InterpolationMode.NEAREST)

            # 小于裁剪尺寸时先补齐，再对 image 和 mask 同步随机裁剪。
            crop_h, crop_w = cfg["crop_size"]
            width, height = image.size
            image, mask = pad_pair(
                image, mask, max(0, crop_w - width), max(0, crop_h - height),
                cfg["ignore_index"],
            )
            top, left, height, width = RandomCrop.get_params(image, (crop_h, crop_w))
            image = F.crop(image, top, left, height, width)
            mask = F.crop(mask, top, left, height, width)

            if random.random() < cfg["hflip_prob"]:
                image = F.hflip(image)
                mask = F.hflip(mask)
        else:
            # 验证保留原始分辨率和完整内容，不随机裁剪、不缩放。
            width, height = image.size
            divisor = cfg["pad_divisor"]
            image, mask = pad_pair(
                image, mask, (-width) % divisor, (-height) % divisor,
                cfg["ignore_index"],
            )

        image = F.to_tensor(image)
        image = F.normalize(image, cfg["mean"], cfg["std"])
        # 保留 PNG 调色板中的类别编号；不能对 mask 使用 to_tensor 除以 255。
        mask = F.pil_to_tensor(mask).squeeze(0).long()
        return image, mask


def build_loaders(config, device):
    data_cfg = config["dataset"]
    train_cfg = config["train"]

    train_dataset = VOCSegmentation(
        root=data_cfg["root"],
        year=data_cfg["year"],
        image_set="train",
        download=data_cfg["download"],
        transforms=SegmentationTransform(data_cfg, training=True),
    )
    val_dataset = VOCSegmentation(
        root=data_cfg["root"],
        year=data_cfg["year"],
        image_set="val",
        download=data_cfg["download"],
        transforms=SegmentationTransform(data_cfg, training=False),
    )

    loader_args = {
        "num_workers": train_cfg["num_workers"],
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg["batch_size"], shuffle=True,
        **loader_args,
    )
    # 原尺寸验证时，各图像大小可以不同，因此逐张验证。
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **loader_args)
    return train_loader, val_loader
