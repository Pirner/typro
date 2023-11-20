import cv2
import albumentations as A


class TransformerConfig:
    @staticmethod
    def get_train_transforms(im_h: int, im_w: int):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        t_train = A.Compose([
            A.Resize(im_h, im_w, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
            A.GaussNoise()]
        )

        return t_train

    @staticmethod
    def get_val_transform(im_h: int, im_w: int):
        t_val = A.Compose([A.Resize(im_h, im_w, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(),
                           A.GridDistortion(p=0.2)])

        return t_val
