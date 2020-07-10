from albumentations import (
    Resize,
    Normalize,
    Compose)
from albumentations.pytorch import ToTensor
from itertools import chain

class DefaultTransformer:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, original_image):
        self.augmentation_pipeline = Compose(
            [
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
        string = str(self.height) + "x" + str(self.width) + " | " + "DFLT" 
        return string