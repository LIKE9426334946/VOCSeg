import torch
import segmentation_models_pytorch as smp


class SegmentationMetrics:
    """累计整个 epoch 的统计量，再使用 SMP 计算各类别的宏平均指标。"""

    def __init__(self, num_classes, ignore_index, device):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # 分别保存 TP、FP、FN、TN，维度为 [1, C]。
        self.stats = [
            torch.zeros((1, num_classes), dtype=torch.long, device=device)
            for _ in range(4)
        ]

    @torch.no_grad()
    def update(self, predictions, targets):
        batch_stats = smp.metrics.get_stats(
            predictions,
            targets,
            mode="multiclass",
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )
        for total, batch in zip(self.stats, batch_stats):
            # SMP 的部分版本会在 CPU 返回统计量，统一设备后再累计。
            total += batch.sum(dim=0, keepdim=True).to(total.device)

    def compute(self):
        functions = {
            "miou": smp.metrics.iou_score,
            "f1": smp.metrics.f1_score,
            "precision": smp.metrics.precision,
            "recall": smp.metrics.recall,
        }
        return {
            name: function(
                *self.stats, reduction="macro", zero_division=0.0
            ).item()
            for name, function in functions.items()
        }
