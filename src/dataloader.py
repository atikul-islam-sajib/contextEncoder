import os
import sys
import zipfile
import torch

sys.path.append("src/")

from utils import config

class Loader:
    def __init__(self, image_path=None, channels=3, batch_size=1, split_size=0.25):
        self.image_path = image_path
        self.channels = channels
        self.batch_size = batch_size
        self.split_size = split_size
        
        self.CONFIG = config()
        
    def unzip_folder(self):
        
