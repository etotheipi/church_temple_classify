
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as preproc

aug_kwargs = {'row_axis': 0, 'col_axis': 1, 'channel_axis': 2}
class ImageUtilities:
    
    
    # All augmentations should be called for all images.  Must have randomness built-in
    AUGMENTATIONS = {
        'shear':   lambda img: preproc.random_shear(img, 0.1, fill_mode='reflect', **aug_kwargs),
        'zoom':    lambda img: preproc.random_zoom(img, (0.9, 1.1), fill_mode='reflect', **aug_kwargs),
        'rotate':  lambda img: preproc.random_rotation(img, 10, fill_mode='reflect', **aug_kwargs),
        'channel': lambda img: preproc.random_channel_shift(img, 0.25, channel_axis=2),
        'hflip':   lambda img: np.flip(img, axis=1) if np.random.choice([True, False]) else img,
        'crop':    lambda img: ImageUtilities.random_crop(img)
    }
    
    @staticmethod
    def load_image(fullpath, preproc_func=lambda x: x / 255.):
        """
        This loads and image from file, doing a couple of basic checks before
        """
        out = preproc.img_to_array(preproc.load_img(fullpath))
        out = preproc_func(out)
        
        # All training images were resized down to smaller versions before training...
        if max(out.shape[:2]) > 512:
            ref_size = max(out.shape[:2])
            scale = float(512) / ref_size
            new_sz0 = int(scale * out.shape[0])
            new_sz1 = int(scale * out.shape[1])
            out = cv2.resize(out, (new_sz1, new_sz0))
            
        return out.astype('float32')
    
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
    def random_crop(img):
        # Randomly choose to crop the image, and if crop, randomly choose crop size and shift
        h,w = img.shape[:2]
        
        # 1/3 chance we do no cropping (if not already square)
        if h == w or np.random.choice([True, False], p=[0.33, 0.67]):
            return img
        
        if h > w:
            new_h = np.random.choice(range(w, h)) # random new height between full and square
            max_offset = h - new_h
            offset = np.random.choice(range(max_offset)) # random offset from top of image
            return img[offset:offset+new_h, :, :]
        else:
            new_w = np.random.choice(range(h, w)) # random new width between full and square
            max_offset = w - new_w
            offset = np.random.choice(range(max_offset)) # random offset from left of image
            return img[:, offset:offset+new_w, :]
            
            
    @staticmethod
    def augment_image(img, resize_to=None):
        """
        Apply all augmentations to each image, but in a randomized order
        """
        aug_keys = list(ImageUtilities.AUGMENTATIONS.keys())
        np.random.shuffle(aug_keys)
        
        out = img
        for aug_name in aug_keys:
            out = ImageUtilities.AUGMENTATIONS[aug_name](out)
            
        if resize_to:
            out = cv2.resize(out, resize_to)
        
        return out

