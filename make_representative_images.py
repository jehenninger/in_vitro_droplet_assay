import sys
import os
import json
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
from skimage import io, filters, measure, color, exposure, morphology, feature, img_as_float, img_as_uint

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import re

def read_metadata(input_args):
    metadata_file = input_args.metadata_file
    metadata_dir = os.path.dirname(metadata_file)

    output_dirs = []

    if input_args.o:
        output_dirs.append(os.path.join(metadata_dir, input_args.o))
    else:
        output_dirs.append(os.path.join(metadata_dir, 'representative_output'))

    if not os.path.isdir(output_dirs[0]):
        os.mkdir(output_dirs[0])

    return output_dirs


def generate_images(output_dirs, input_args):

    metadata = pd.read_excel(input_args.metadata_file)

    output_groups = np.unique(metadata['sample_name'])

    for s in output_groups:
        metadata_sample_subset = metadata[metadata['sample_name'] == s].copy()

        replicates = np.unique(metadata_sample_subset['replicate'])

        for r in replicates:
            metadata_replicate_subset = metadata_sample_subset[metadata_sample_subset['replicate'] == r].copy()
            unique_id = s + str(r)

            channel_A_path = metadata_replicate_subset[metadata_replicate_subset['channel_id'] == 1]['file_path']
            channel_A_path = channel_A_path.iloc[0]

            channel_B_path = metadata_replicate_subset[metadata_replicate_subset['channel_id'] == 2]['file_path']
            channel_B_path = channel_B_path.iloc[0]

            # read images
            channel_A = io.imread(channel_A_path)
            channel_B = io.imread(channel_B_path)

            channel_A = img_as_float(channel_A)
            channel_B = img_as_float(channel_B)

            # Gaussian blur
            channel_A_blur = filters.gaussian(channel_A, sigma=1)
            channel_B_blur = filters.gaussian(channel_B, sigma=1)

            # threshold based on max and min

            threshold_A_min = metadata_replicate_subset[metadata_replicate_subset['channel_id'] == 1]['min_intensity'].iloc[0]
            threshold_A_min = threshold_A_min/65536
            threshold_A_max = metadata_replicate_subset[metadata_replicate_subset['channel_id'] == 1]['max_intensity'].iloc[0]
            threshold_A_max = threshold_A_max/65536

            threshold_B_min = metadata_replicate_subset[metadata_replicate_subset['channel_id'] == 2]['min_intensity'].iloc[0]
            threshold_B_min = threshold_B_min/65536
            threshold_B_max = metadata_replicate_subset[metadata_replicate_subset['channel_id'] == 2]['max_intensity'].iloc[0]
            threshold_B_max = threshold_B_max/65536


            # channel_A_blur_contrast = exposure.rescale_intensity(channel_A_blur, in_range='image',
            #                                                      out_range=(threshold_A_min, threshold_A_max))
            # channel_B_blur_contrast = exposure.rescale_intensity(channel_B_blur, in_range='image',
            #                                                     out_range=(threshold_B_min, threshold_B_max))


            green_multiplier = [0, 1, 0]
            magenta_multplier = [1, 0, 1]

            channel_A_final = color.gray2rgb(channel_A_blur) * green_multiplier
            channel_B_final = color.gray2rgb(channel_B_blur) * magenta_multplier


            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(8, 5.5), sharex=True, sharey=True)

            ax1.imshow(channel_A_final)
            ax2.imshow(channel_B_final)
            ax3.imshow(channel_A_final + channel_B_final)

            ax1.get_xaxis().set_ticks([])
            ax1.get_yaxis().set_ticks([])
            ax2.get_xaxis().set_ticks([])
            ax2.get_yaxis().set_ticks([])
            ax3.get_xaxis().set_ticks([])
            ax3.get_yaxis().set_ticks([])

            plt.savefig(os.path.join(output_dirs[0], unique_id + ".png"), dpi=300)
            plt.savefig(os.path.join(output_dirs[0], unique_id + ".eps"))

            plt.close()


# parse input
parser = argparse.ArgumentParser()

# path to metadata file with specific columns:
#  1. file_path = path to each separate image file (even separated by channels)
#  2. sample_name = name of sample that you want to be output
#  3. replicate = the replicate number for each set of images. The combination of output_name + replicate should be unique for that sample
#  4. channel_id = 1 or 2 (currently only supports 2 channels)
#  5. min_intensity = minimum intensity threshold for that channel. Anything below = 0.
#  6. max_intensity = maximum intensity threshold for that channel. Anything above = max_intensity

parser.add_argument("metadata_file")

parser.add_argument("--o", type=str)  # output directory name

# parser.add_argument('--no-legend', dest='legend_flag', action='store_false')
# parser.set_defaults(threshold_flag=True, fit_flag=True, legend_flag=True)

input_args = parser.parse_args()

metadata_path = input_args.metadata_file

if not os.path.isfile(metadata_path):
    sys.exit('ERROR: Could not find metadata file')

output_dirs = read_metadata(input_args)
generate_images(output_dirs, input_args)