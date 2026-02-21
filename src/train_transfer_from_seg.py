import torch
from models import ClassificationModel
from utils import console


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_backbone(model, backbone_path):
    seg_weights = torch.load(backbone_path)

    model_dict = model.model.state_dict()
    seg_weights = {k: v for k, v in seg_weights.items() if k in model_dict}

    model_dict.update(seg_weights)
    model.model.load_state_dict(model_dict)

    console.print("[green]Backbone transferred successfully[/green]")
    return model


def main():
    model = ClassificationModel(num_classes=7, pretrained=False)
    model = load_backbone(
        model,
        "../data/processed/segmentation_backbone.pth"
    )
    model = model.to(DEVICE)

    console.print("[bold blue]Now fine-tune on HAM10000[/bold blue]")


if __name__ == "__main__":
    main()