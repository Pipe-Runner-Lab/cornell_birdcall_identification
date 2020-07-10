from albumentations import (
    Resize,
    Normalize,
    HorizontalFlip,
    IAASharpen,
    OneOf,
    Compose,
    Rotate,
    CenterCrop,
    GridDropout,
    RandomCrop
)
from albumentations.pytorch import ToTensor


class IMG2:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, original_image):
        self.augmentation_pipeline = Compose(
            [
                HorizontalFlip(p=0.5),
                Rotate(limit=45, p=0.5),
                RandomCrop(height=int(0.9*224),width=int(0.9*224), p=0.5),
                Resize(224, 224, always_apply=True),
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
            " | " + "IMG2"
        return string
