import numpy as np
import pandas as pd

from pathlib import Path
from xml.dom.minidom import parse
from shutil import copyfile
import os


def create_train_val_test(path_to_images, path_to_labels, file_list, modes):

    for mode in modes:

        root_file = Path(path_to_images + '/' + mode)
        if not root_file.exists():
            print(f"Path {root_file} does not exit")
            os.makedirs(root_file)

        root_file = Path(path_to_labels + '/' + mode)
        if not root_file.exists():
            print(f"Path {root_file} does not exit")
            os.makedirs(root_file)

        for file in file_list:
            img_name = file.replace('.png', '')        
            img_src_file = path_to_images + '/' + img_name + '.png'
            label_src_file = path_to_labels + '/' + img_name + '.txt'

            # Copy image
            DICT_DIR = path_to_images  + '/' + mode
            img_dict_file = DICT_DIR + '/' + img_name + '.png'
           
            copyfile(img_src_file, img_dict_file)
           
            # Copy label
            DICT_DIR = path_to_labels + '/' + mode
            img_dict_file = DICT_DIR + '/' + img_name + '.txt'
            copyfile(label_src_file, img_dict_file)