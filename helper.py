from skimage import io, filters, measure, color, exposure, morphology, feature, img_as_float, img_as_uint
import pandas as pd
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import argparse
import json
from datetime import datetime


def read_metadata(input_args):
    metadata_path = input_args.metadata_path

    if not os.path.isdir(metadata_path):
        print('ERROR: Could not read or find  metadata file')
        sys.exit(0)

    metadata_dir = os.path.dirname(metadata_path)
    metadata_name = os.path.splitext(metadata_path)[0]

    output_dirs = []

    if input_args.o:
        output_dirs.append(os.path.join(metadata_dir, input_args.o))
    else:
        output_dirs.append(os.path.join(metadata_dir, metadata_name + '_output'))

    if not os.path.isdir(output_dirs[0]):
        os.mkdir(output_dirs[0])

    metadata = pd.read_excel(metadata_path)

    return metadata, output_dirs


def analyze_replicate(metadata, input_args):

    channels = np.unique(metadata['channel'])
    num_of_channels = len(channels)

    if num_of_channels > 2:
        print('ERROR: Currently can only do up to 2 channels')
        sys.exit(0)

    scaffold_image_flag = False
    client_image_flag = False

    if type(input_args.s) is int:
        scaffold_channel_name = input_args.s
        scaffold_test = [b == scaffold_channel_name for b in channels]

        for idx, c in enumerate(channels):
            if scaffold_test[idx]:
                scaffold_image_path = metadata[idx]['image_path'].copy()
                scaffold = io.imread(scaffold_image_path)
                scaffold_image_flag = True
            else:
                client_image_path = metadata[idx]['image_path'].copy()
                client = io.imread(client_image_path)
                client_image_flag = True

    elif type(input_args.s) is str:
        if input_args.s is 'avg':
            sys.exit(0)  # @Remove
            # do average scaffold here @Jon START HERE!

    else:
        print('ERROR: Could not identify scaffold parameter for replicate ', metadata['replicate'][0], ' in sample ', metadata['experiment_name'][0])
        sys.exit(0)

    #  define crop region
    crop_mask = np.full(shape=(scaffold.shape[0], scaffold.shape[1]), fill_value=False, dtype=bool)

    if input_args.crop:
        c_width = input_args.crop
        center_coord = scaffold.shape[0]/2
        crop_mask[range(center_coord-c_width, center_coord+c_width), range(center_coord-c_width, center_coord+c_width)] = True
    else:
        crop_mask.fill(True)

    scaffold = scaffold(crop_mask)

    if client_image_flag:
        client = client(crop_mask)

    if input_args.bsub_flag:

        if scaffold_image_flag:
            scaffold_bsub = subtract_background(scaffold)

        if client_image_flag:
            client_bsub = subtract_background(client)


def subtract_background(input_image):
    image_hist = np.histogram(input_image, bins='auto')
    background_threshold = np.argmax(image_hist)  # assumes that the max hist peak corresponds to background pixels
    output_image = input_image - background_threshold

    return output_image


def analyze_sample(metadata, input_args):
    sys.exit(0)  # @Remove
