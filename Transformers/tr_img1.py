from albumentations import (
    Resize,
    Normalize,
    HorizontalFlip,
    Blur,
    IAASharpen,
    ShiftScaleRotate,
    IAAPiecewiseAffine,
    OneOf,
    Compose)
from albumentations.pytorch import ToTensor

class IMG1:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, original_image):
        self.augmentation_pipeline = Compose(
            [
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(rotate_limit=25.0, p=0.7),
                OneOf([
                       IAASharpen(p=1),
                       Blur(p=1)], p=0.5),
                IAAPiecewiseAffine(p=0.5),
                Resize(self.height, self.width, always_apply=True),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    always_apply=True
                ),
                ToTensor()
            ]
        )

        augmented = self.augmentation_pipeline(
            image=original_image
        )
        image = augmented["image"]
        return image

    def __str__(self):
        string = str(self.height) + "x" + str(self.width) + \
            " | " + "IMG1"
        return string