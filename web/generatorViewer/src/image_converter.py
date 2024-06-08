import io
import base64
from PIL import Image
from torchvision.transforms import ToPILImage, PILToTensor


class ImageUtils:
    @classmethod
    def _tensor2pillow(cls, image_tensor):
        return ToPILImage()(image_tensor)

    @classmethod
    def _pillow2tensor(cls, image_pillow):
        return PILToTensor()(image_pillow)

    @classmethod
    def _pillow2base64(cls, image_pillow: Image):
        buffered = io.BytesIO()
        image_pillow.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @classmethod
    def _base64_2pillow(cls, image_base64: str):
        image_data = base64.b64decode(image_base64)
        buffered = io.BytesIO(image_data)
        return Image.open(buffered)

    @classmethod
    def tensor2base64(cls, image_tensor):
        return cls._pillow2base64(cls._tensor2pillow(image_tensor))

    @classmethod
    def base64_2tensor(cls, image_base64):
        return cls._base64_2pillow(cls._pillow2tensor(image_base64))
