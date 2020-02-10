import argparse
import cv2
import os
import numpy as np
import subprocess
from tensorflow import keras
import tensorflow.keras.layers as L
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.inception_v3 import preprocess_input as xcept_preproc

from image_utilities import ImageUtilities
from rotation_correction import RotationCorrection

# Some global parameters
INPUT_IMG_SIZE_2D = (299, 299)
INPUT_IMG_SIZE_3D = (299, 299, 3)
INTERMEDIATE_LAYERS = [45, 65, 95] # Output of Xception blocks 3, 5 and 8
COUNTRY_NAMES = ['Armenia', 'Australia', 'Germany', 'Hungary+Slovakia+Croatia',
                 'Indonesia-Bali', 'Japan', 'Malaysia+Indonesia', 'Portugal+Brazil',
                 'Russia', 'Spain', 'Thailand']
NUM_COUNTRIES = len(COUNTRY_NAMES)
FINAL_MODEL_WEIGHTS_FILENAME = 'church_temple_classify_weights.hdf5'
FEX_BASE_NAME = 'Xception'

# This can be imported directly to create the model without the trained weights
def generate_multi_out_xception_model(weights_file=None):
    fex_base = Xception(weights='imagenet', include_top=False)
    inp_layer = fex_base.layers[0].input
    inter_layers = [fex_base.layers[l].output for l in INTERMEDIATE_LAYERS]
    out_layer = fex_base.layers[-1].output
    
    # GlobalAvgPooling of all 3 intermediate layers and the normal Xception output layer
    x = multiple_inters = [L.GlobalAveragePooling2D()(lout) for lout in (inter_layers + [out_layer])]
    
    # Concatenate those four layers together into a 4,232-feature vector
    x = combined = L.Concatenate(axis=-1)(multiple_inters)
    
    # Intermediate 64-dense layer between the above and the logistic regression head
    x = L.Dense(64)(x)
    x = L.LeakyReLU(alpha=0.15)(x)
    x = L.Dropout(0.15)(x)
    
    # Here's the logistic regression part...
    x = predictions = L.Dense(NUM_COUNTRIES, activation='sigmoid')(x)
    
    model = keras.models.Model(inputs=inp_layer, outputs=predictions)
    if weights_file:
        model.load_weights(weights_file)
        
    return model


class ChurchTempleClassifier:
    
    def __init__(self, main_hdf5_file=None, rotate_clf_file=None):
        self.main_model_file = FINAL_MODEL_WEIGHTS_FILENAME
        self.rotate_model_file = rotate_clf_file
        
        # This handles downloading the file if it doesn't exist yet (or unspecified)
        self.rotate_model = RotationCorrection(self.rotate_model_file)
        
        # Need to implement downloading file if it doesn't exist, for main model
        if self.main_model_file is None or not os.path.exists(self.main_model_file):
            # Download with wget (to avoid the caller needing boto3 library)
            try:
                s3obj = f'https://acr-toptal-codingproj.s3.amazonaws.com/{FINAL_MODEL_WEIGHTS_FILENAME}'
                print(f'Attempting to download: {s3obj}')
                subprocess.check_call(['wget', s3obj])
                print(f'Successful!')
            except Exception as e:
                raise 
                
            self.main_model_file = s3obj.split('/')[-1]
        else:
            print(f'Using already-downloaded model file: {self.main_model_file}')
                                    
        print(f'Loading main classifier weights:  {self.main_model_file}')
        self.main_model = generate_multi_out_xception_model(self.main_model_file)
        

    def process_one_file(self, img_path):
        """
        Given an image path, run the model to predict the country
        """
        if not os.path.splitext(img_path)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            print(f'Skipping non-image file: {img_path}')
            return None, None
        
        img = ImageUtilities.load_image(img_path, xcept_preproc)
        img = cv2.resize(img, INPUT_IMG_SIZE_2D)
        img, _ = self.rotate_model.fix_one_image(img)
        img = np.expand_dims(img, axis=0)
        model_out = self.main_model(img)
        out_idx = np.argmax(model_out.numpy().reshape([-1]))
        
        # Want to get probs out but model was built with sigmoid, no access to logits.  Invert
        inverse_sigmoid = lambda y: -np.log(1.0 / y - 1)
        logits = inverse_sigmoid(model_out.numpy().reshape([-1]))
        softmax_norm = np.sum(np.exp(logits))
        probas = np.exp(logits) / softmax_norm
        
        return COUNTRY_NAMES[out_idx], probas[out_idx]
        
        
    def process_directory(self, dir_path):
        """
        Given a directory, run process_one_file on all images in the dir.  Creates .csv as output.
        """
        print(f'Processing all images in directory: {dir_path}')
        out_filename = f'classify_results_{dir_path.replace("/", "_")}.csv'
        result_map = {}
        with open(out_filename, 'w') as fwrite:
            for img_file in os.listdir(dir_path):
                full_path = os.path.join(dir_path, img_file)
                country, probs = self.process_one_file(full_path)
                if country is not None:
                    fwrite.write(f'{img_file},{country}\n')
                    result_map[img_file] = (country, probs)
                
        return out_filename, result_map
              
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir', type=str, help='Dir w/ imgs of churches/temples to classify')
    parser.add_argument('-m', '--main-model', default=None, help='Filename of main model file (.hdf5)')
    parser.add_argument('-r', '--rotate-model', default=None, help='Filename of rotation-correction model (.clf)')
    args = parser.parse_args()
    
    clf = ChurchTempleClassifier(args.main_model, args.rotate_model)
    out_filename, result_map = clf.process_directory(args.img_dir)
    print(f'Classification complete, results are in the file: {out_filename}')
    
        
