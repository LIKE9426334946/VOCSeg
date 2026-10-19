import segmentation_models_pytorch as smp


def build_unet(model_config, num_classes):
    """输出形状为 [B, num_classes, H, W] 的原始 logits。"""
    return smp.Unet(
        encoder_name=model_config["encoder_name"],
        encoder_weights=model_config["encoder_weights"],
        in_channels=3,
        classes=num_classes,
        activation=None,
    )


if __name__ == "__main__":
    import yaml
    from pathlib import Path
    import torch

    with open("config.yaml", "r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)
    model = build_unet(config["model"], 21)
    x = torch.randn(4, 3, 384, 384)
    with torch.no_grad():
        model(x)
        torch.onnx.export(
            model,
            x,
            "temp1.onnx",
            input_names=["input"],
            output_names=["output"],
            external_data=False,
        )
