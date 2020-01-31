import os
import numpy as np
import pandas as pd
import cv2
 
import tensorflow as tf
from tensorflow.keras import preprocessing as preproc

def load_image(filename):
    return cv2.imread(filename)

