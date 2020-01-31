
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import preprocessing as preproc

aug_kwargs = {'row_axis': 0, 'col_axis': 1, 'channel_axis': 2}
class ImageUtilities:
    
    AUGMENTATIONS = {
        #'bright': lambda img: preproc.image.random_brightness(img, (0.00, 0.02)),
        'shear': lambda img: preproc.image.random_shear(img, 0.1, fill_mode='reflect', **aug_kwargs),
        'zoom': lambda img: preproc.image.random_zoom(img, (0.9, 1.1), fill_mode='reflect', **aug_kwargs),
        'rotate': lambda img: preproc.image.random_rotation(img, 10, fill_mode='reflect', **aug_kwargs),
        'hflip': lambda img: np.flip(img, axis=1) if np.random.choice([True, False]) else img,
    }
    
    @staticmethod
    def load_image(fullpath):
        """
        The np.flip reverses the channels so that it matches matplotlib's native imshow
        """
        out = np.flip(cv2.imread(fullpath) / 255., axis=-1)
        # All training images were resized down to smaller versions before training...
        if max(out.shape[:2]) > 512:
            ref_size = max(out.shape[:2])
            scale = float(512) / ref_size
            new_sz0 = int(scale * out.shape[0])
            new_sz1 = int(scale * out.shape[1])
            out = cv2.resize(out, (new_sz1, new_sz0))
            
        return out
    
    @staticmethod
    def rotate_image_90deg(img, clicks):
        """
        Rotate image in multiples of 90 deg. `clicks` is the number of multiples.
        """
        if clicks == 0:
            return img
        if clicks in [1, -3]:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif clicks in [2, -2]:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif clicks in [3, -1]:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            raise Exception(f'Invalid click count for rotation (0-3): {clicks}')
            
    @staticmethod
    def augment_image(img):
        """
        We most likely don't want to apply all augmentations to all images.
        We'll supply a list of augs we allow, and how many to randomly select
        """
        aug_keys = list(ImageUtilities.AUGMENTATIONS.keys())
        np.random.shuffle(aug_keys)
        print(aug_keys)
        
        out = img
        for aug_name in aug_keys:
            out = ImageUtilities.AUGMENTATIONS[aug_name](out)
        
        return out

