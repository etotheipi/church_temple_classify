import cv2
import os
import pickle
import numpy as np
import tensorflow as tf
import subprocess
from image_utilities import ImageUtilities
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.inception_v3 import preprocess_input

class RotationCorrection:
    DEFAULT_MODEL_FILE = 'rotation_detect_xcept_logreg.clf'
    
    def __init__(self, logreg_model_file=None):
        """
        The logistic regression model will be downloaded from S3 if it doesn't exist locally (yet)
        """
        # I like to put all the members of the class at the top
        self.model_file = logreg_model_file if logreg_model_file else self.DEFAULT_MODEL_FILE
        self.xception_no_head = None
        self.rotate_detector_head = None
        
        # If the model file does not exist locally, download it
        if self.model_file is None or not os.path.exists(self.model_file):
            # Download with wget (to avoid the caller needing boto3 library)
            try:
                s3obj = f'https://acr-toptal-codingproj.s3.amazonaws.com/{self.DEFAULT_MODEL_FILE}'
                print(f'Attempting to download: {s3obj}')
                subprocess.check_call(['wget', s3obj])
                print(f'Successful!')
            except Exception as e:
                raise 
                
            self.model_file = s3obj.split('/')[-1]
        else:
            print(f'Using already-downloaded model file: {self.model_file}')
                                    
        self.xception_no_head = Xception(weights='imagenet', include_top=False)
        with open(self.model_file, 'rb') as fread:
            self.rotate_detector_head = pickle.load(fread)
        
        
    def fix_one_image(self, img, quiet=False):
        """
        This assumes you've already applied the preprocessing function to it
        """
        
        orig_shape = img.shape
        
        # First squeeze it down to 3D (if necessary) for cv2 to resize it
        if len(orig_shape) == 4:
            img = np.squeeze(img, axis=0)
            
        # If it's not currently the right size, resize it
        if img.shape[:2] != (299, 299):
            img = cv2.resize(img, (299, 299))
            
        # We need to (re-add) the fourth dim to pass it like a batch (of size=1) to Xception
        img = np.expand_dims(img, axis=0)
        
        # Apply the classifier
        xcept_out = self.xception_no_head.predict(img).reshape([1, -1])
        
        num_clicks = self.rotate_detector_head.predict(xcept_out)[0]
        if num_clicks > 0:
            if not quiet:
                print(f'Image was rotated {num_clicks} clicks.  Rotating it to normal orientation.')
            img = np.squeeze(img, axis=0)
            img = ImageUtilities.rotate_image_90deg(img, -num_clicks)
            
        if len(orig_shape) == 3 and len(img.shape) == 4:
            img = np.squeeze(img, axis=0)
            
        if len(orig_shape) == 4 and len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)
            
        return img, num_clicks
    
