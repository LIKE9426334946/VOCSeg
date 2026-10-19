from .unet import build_unet


MODEL_BUILDERS = {
    "unet": build_unet,
}


def build_model(model_config, num_classes):
    """新增模型时，在 MODEL_BUILDERS 中注册对应的构建函数。"""
    return MODEL_BUILDERS[model_config["name"]](model_config, num_classes)
