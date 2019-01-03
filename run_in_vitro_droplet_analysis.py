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
from skimage import io, filters, measure, color, exposure, morphology, feature, img_as_float, img_as_uint
import helper



# metadata has the following columns:
#
# image_path : full paths to every channel image
#
# experiment_name : unique name for each sample/experiment.
# items with the same experiment_name will be grouped together
# as replicates
#
# replicate : integer number corresponding to the replicate
#
# channel_id : integer number corresponding to the channel (488,561,642).
# this will be used if the scaffold parameter is set

# parse input
parser = argparse.ArgumentParser()
parser.add_argument("metadata_path")
parser.add_argument("--o", type=str)  # optional output directory name
parser.add_argument("--s", default='avg')  # what channel to use for scaffolding. Defaults to average.
parser.add_argument('--crop', type=int)  # width from center point to include in pixels.
                                         # Defaults to entire image (width/2)

parser.add_argument('--no-bsub', dest='bsub_flag', action='store_false', default=True)

input_args = parser.parse_args()

# load and check metadata

metadata, output_dirs = helper.read_metadata(input_args)

# get number of unique experiments
samples = np.unique(metadata['experiment_name'])
num_of_samples = len(samples)

for s in samples:
    print()
    print("Sample: ", s)
    print()
    metadata_sample = metadata[(metadata['experiment_name'] == s)].copy()

    # get number of replicates
    replicates = np.unique(metadata_sample['replicate'])
    num_of_replicates = len(replicates)

    for r in replicates:
        # print('replicate: ', r)

        metadata_replicate = metadata_sample[metadata_sample['replicate'] == r].copy()

        helper.analyze_replicate(metadata_replicate, input_args)

# load images

    # background subtract

    # define illumination region (circle of specific diameter)

    # define scaffold or do average scaffold

    # identify droplets

    # filter droplets by