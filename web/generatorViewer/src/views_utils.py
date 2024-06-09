import json
import torch
import torchvision.utils as vutils

from dataclasses import dataclass

from .image_converter import ImageUtils
from .dataloader import get_dataloader
from .ml_models import GeneratorCELEBA, GeneratorCIFAR10, GeneratorSTL10


@dataclass
class Keys:
    IMAGE_TENSORS = "image_tensors"
    BATCH_INDEX = "batch_index"
    BATCH_SIZE = "batch_size"
    OFFSET = "offset"
    IMAGES = "images"
    IMAGE = "image"
    ENTRIES = "entries"
    META = "meta"
    MODEL_NAME = "model_name"


def create_meta(
    batch_size: int, model_name: str, offset=0, batch_index: int | None = None
):
    return json.dumps(
        {
            Keys.BATCH_INDEX: batch_index,
            Keys.OFFSET: offset,
            Keys.MODEL_NAME: model_name,
            "batch_size": batch_size,
        }
    )


def create_image(
    batch_size: int,
    image_tensor,
    offset: int,
    model_name: str,
    batch_index: int | None = None,
):
    return {
        "image_base64": ImageUtils.tensor2base64(image_tensor),
        Keys.META: create_meta(
            batch_size=batch_size,
            batch_index=batch_index,
            offset=offset,
            model_name=model_name,
        ),
        Keys.OFFSET: offset,
    }


def create_images(**kwargs):
    image_tensors = kwargs[Keys.IMAGE_TENSORS]
    del kwargs[Keys.IMAGE_TENSORS]
    return {
        Keys.ENTRIES: [
            create_image(**{**kwargs, "offset": i, "image_tensor": image_tensor})
            for i, image_tensor in enumerate(image_tensors)
        ],
        Keys.META: create_meta(**kwargs),
        Keys.MODEL_NAME: kwargs["model_name"],
    }


def load_images_from_dataset(batch_size: int):
    dataloader = get_dataloader(batch_size=batch_size)
    data_iterator = iter(dataloader)
    image_tensors, _ = next(data_iterator)
    return image_tensors, data_iterator


def pick_model(model_name):
    match model_name:
        case "celeba":
            return GeneratorCELEBA, 128
        case "stl10":
            return GeneratorSTL10, 64
        case _:
            return GeneratorCIFAR10, 32


def load_generated_images(
    batch_size: int,
    model_name: str,
):
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    generator_callable, image_size = pick_model(model_name)
    generator = generator_callable().to(device)
    generator.load_state_dict(
        torch.load(f"web/generatorViewer/src/saved_models/generator-{model_name}.pth")
    )
    noise = torch.randn(image_size, 100, 1, 1, device=device)
    image_tensors = generator(noise)
    print(image_tensors.shape)
    tensors = [
        vutils.make_grid(
            image_tensor.to(device)[:image_size], padding=5, normalize=True
        ).cpu()
        for image_tensor in image_tensors
    ]
    response = torch.stack(tensors)
    print(response.shape)
    return response[:batch_size]
