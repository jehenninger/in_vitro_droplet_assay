from skimage import io, filters, measure, color, exposure, morphology, feature, img_as_float, img_as_uint
from scipy import ndimage as ndi
import pandas as pd
import numpy as np
import os
import sys
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import argparse
import json
from datetime import datetime

circ = lambda r: (4 * math.pi * r.area) / (r.perimeter * r.perimeter)

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
    sample_name = np.unique(metadata['experiment_name'])
    replicate_name = np.unique(metadata['replicate'])
    channels = np.unique(metadata['channel'])
    num_of_channels = len(channels)

    if num_of_channels > 2:
        print('ERROR: Currently can only do up to 2 channels')
        sys.exit(0)

    scaffold_image_flag = False
    client_image_flag = False
    avg_image_flag = False

    if type(input_args.s) is int:
        scaffold_channel_name = input_args.s
        scaffold_test = [b == scaffold_channel_name for b in channels]

        for idx, c in enumerate(channels):
            if scaffold_test[idx]:
                scaffold_image_path = metadata[idx]['image_path'].copy()
                scaffold = io.imread(scaffold_image_path)
                scaffold = img_as_float(scaffold)
                scaffold_image_flag = True
            else:
                client_image_path = metadata[idx]['image_path'].copy()
                client = io.imread(client_image_path)
                client = img_as_float(client)
                client_image_flag = True

    elif type(input_args.s) is str:
        if input_args.s is 'avg':
            for idx, c in enumerate(channels):
                count = 0
                if idx == 0:
                    avg_image = io.imread(metadata[idx]['image_path'].copy())
                    avg_image = img_as_float(avg_image)
                    client_a = avg_image
                    count = count + 1
                else:
                    image_to_add = io.imread(metadata[idx]['image_path'].copy())
                    image_to_add = img_as_float(image_to_add)
                    client_b = image_to_add

                    avg_image = avg_image + image_to_add
                    count = count + 1

            avg_image = avg_image/count
            scaffold = avg_image

            avg_image_flag = True

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

    if avg_image_flag:
        client_a = client_a(crop_mask)
        client_b = client_b(crop_mask)

    # find std of image for later thresholding @Important before background subtraction
    scaffold_std = np.std(scaffold)
    threshold_multiplier = input_args.tm

    # background subtraction
    if input_args.bsub_flag:

        if scaffold_image_flag:
            scaffold_bsub, scaffold_bg_peak = subtract_background(scaffold)

        if client_image_flag:
            client_bsub, client_bg_peak = subtract_background(client)

        if avg_image_flag:
            scaffold_bsub, scaffold_bg_peak = subtract_background(scaffold)
            client_a_bsub, client_a_bg_peak = subtract_background(client_a)
            client_b_bsub, client_b_bg_peak = subtract_background(client_b)

    # make binary image of scaffold with threshold intensity. Threshold is multiplier of std above background
    # @Important only add to background peak if we haven't subtracted background
    if input_args.bsub_flag:
        scaffold_binary = scaffold[scaffold > (threshold_multiplier * scaffold_std)]
    else:
        scaffold_binary = scaffold[scaffold > (scaffold_bg_peak + (threshold_multiplier * scaffold_std))]

    scaffold_binary = ndi.morphology.binary_fill_holes(scaffold_binary)
    scaffold_binary_labeled = measure.label(scaffold_binary)
    scaffold_regionprops = measure.regionprops(scaffold_binary_labeled)

    # filter droplets by size and circularity
    min_area_threshold = input_args.min_a
    max_area_threshold = input_args.max_a
    circ_threshold = input_args.circ

    scaffold_mask = np.full(shape=(scaffold.shape[0], scaffold.shape[1]), fill_value=False, dtype=bool)
    for i, region in enumerate(scaffold_regionprops):
        if (min_area_threshold <= region.area <= max_area_threshold) and (circ(region) >= circ_threshold):
            for coords in region.coords:
                scaffold_mask[coords[0], coords[1]] = True

    # re-segment droplets after filtering them
    scaffold_filtered_binary_labeled = measure.label(scaffold_mask)
    scaffold_filtered_regionprops = measure.regionprops(scaffold_filtered_binary_labeled)

    # get measurements of regions from centered circle with radius sqrt(minimum droplet area)
    # what we want:
    # area, centroid, circularity, mean_intensity, max_intensity, mean intensity inside circle
    if num_of_channels == 1:
        replicate_output = pd.DataFrame(columns = ['sample', 'replicate', 'droplet_id', 'subset_I_'+str(channels),
                                                   'mean_I_' + str(channels), 'max_I_' + str(channels),
                                                   'area', 'centroid_x', 'centroid_y', 'circularity'])
    elif num_of_channels == 2:
        replicate_output = pd.DataFrame(columns = ['sample', 'replicate', 'droplet_id',
                                                   'subset_I_'+str(channels[0]), 'subset_I_'+str(channels[1]),
                                                   'mean_I_' + str(channels[0]), 'mean_I_' + str(channels[1]),
                                                   'max_I_' + str(channels)[0], 'max_I_' + str(channels)[1],
                                                   'area', 'centroid_x', 'centroid_y', 'circularity'])

    for i, region in enumerate(scaffold_filtered_regionprops):
        s = sample_name
        r = replicate_name
        droplet_id = i
        coords_x, coords_y = region.coords

        if num_of_channels == 1:
            mean_intensity = region.mean_intensity
            max_intensity = region.max_intensity
            area = region.area
            centroid_x, centroid_y = region.centroid
            circularity = circ(region)



        replicate_output = replicate_output.append({'sample': , 'replicate': ,
                                                    'droplet_id': , },
                                                    ignore_index=True)


    # get measurements of bulk regions excluding droplets

def subtract_background(input_image):
    image_hist = np.histogram(input_image, bins='auto')
    background_threshold = np.argmax(image_hist)  # assumes that the max hist peak corresponds to background pixels
    output_image = input_image - background_threshold
    output_image[output_image < 0] = 0

    return output_image, background_threshold


def analyze_sample(metadata, input_args):
    sys.exit(0)  # @Remove
