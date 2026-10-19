import argparse
import csv
import random
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch import nn
from tqdm import tqdm

from models import build_model
from utils.dataset import build_loaders
from utils.metrics import SegmentationMetrics


def run_epoch(
    model,
    loader,
    criterion,
    device,
    num_classes,
    ignore_index,
    optimizer=None,
    description="Train",
):
    """传入 optimizer 时训练；不传入时验证。"""
    training = optimizer is not None
    model.train(training)
    metrics = SegmentationMetrics(num_classes, ignore_index, device)
    total_loss = 0.0
    total_pixels = 0
    progress = tqdm(loader, desc=description)

    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        valid_pixels = (masks != ignore_index).sum()

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss_sum = criterion(logits, masks)
            # 正常情况下等同于 CrossEntropyLoss 的 mean；全为忽略像素时为 0。
            loss = loss_sum / valid_pixels.clamp_min(1)

            if training:
                loss.backward()
                optimizer.step()

        predictions = logits.detach().argmax(dim=1)
        metrics.update(predictions, masks)
        total_loss += loss_sum.detach().item()
        total_pixels += valid_pixels.item()
        progress.set_postfix(loss=f"{total_loss / max(total_pixels, 1):.4f}")

    return {"loss": total_loss / max(total_pixels, 1), **metrics.compute()}


def main(config_path):
    config_path = Path(config_path).resolve()
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # 数据和输出的相对路径都以 config.yaml 所在目录为基准。
    data_cfg = config["dataset"]
    train_cfg = config["train"]
    data_cfg["root"] = str((config_path.parent / data_cfg["root"]).resolve())

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    device_name = config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    train_loader, val_loader = build_loaders(config, device)
    model = build_model(config["model"], data_cfg["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss(
        ignore_index=data_cfg["ignore_index"], reduction="sum"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = config_path.parent / train_cfg["output_dir"] / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

    print(f"Device: {device}")
    print(
        f"Train: {len(train_loader.dataset)} images | Val: {len(val_loader.dataset)} images"
    )
    print(f"Model: {config['model']['name']} / {config['model']['encoder_name']}")
    print(f"Output: {output_dir}")

    metric_names = ["loss", "miou", "f1", "precision", "recall"]
    fieldnames = ["epoch"] + [
        f"{split}_{name}" for split in ("train", "val") for name in metric_names
    ]
    best_miou = -1.0

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, train_cfg["epochs"] + 1):
            train_result = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                data_cfg["num_classes"],
                data_cfg["ignore_index"],
                optimizer=optimizer,
                description=f"Epoch {epoch} Train",
            )
            val_result = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                data_cfg["num_classes"],
                data_cfg["ignore_index"],
                description=f"Epoch {epoch} Val",
            )

            writer.writerow(
                {
                    "epoch": epoch,
                    **{f"train_{key}": value for key, value in train_result.items()},
                    **{f"val_{key}": value for key, value in val_result.items()},
                }
            )
            file.flush()

            is_best = val_result["miou"] > best_miou
            best_miou = max(best_miou, val_result["miou"])
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_miou": best_miou,
                "config": config,
            }
            torch.save(checkpoint, output_dir / "last.pth")
            if is_best:
                torch.save(checkpoint, output_dir / "best.pth")

            print(
                f"Epoch {epoch}/{train_cfg['epochs']} | "
                f"train loss {train_result['loss']:.4f}, mIoU {train_result['miou']:.4f} | "
                f"val loss {val_result['loss']:.4f}, mIoU {val_result['miou']:.4f}, "
                f"F1 {val_result['f1']:.4f} | best mIoU {best_miou:.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VOC 2012 语义分割：训练与验证")
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()
    main(args.config)
