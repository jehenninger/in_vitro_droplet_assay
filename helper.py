from skimage import io, filters, measure, color, exposure, morphology, feature, img_as_float, img_as_uint, draw
from scipy import ndimage as ndi
import pandas as pd
import numpy as np
import os
import sys
import math
import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import argparse
import json
from datetime import datetime

circ = lambda r: (4 * math.pi * r.area) / (r.perimeter * r.perimeter)


def read_metadata(input_args, metadata_path):
    # metadata_path = input_args.metadata_path

    metadata_dir = os.path.dirname(metadata_path)
    metadata_name = os.path.splitext(metadata_path)[0]
    metadata_name = os.path.basename(metadata_name)

    if not os.path.isdir(metadata_dir) or not os.path.isfile(metadata_path):
        print('ERROR: Could not read or find  metadata file')
        sys.exit(0)

    if input_args.o:
        output_parent_dir = os.path.join(metadata_dir, input_args.o)
        # output_dirs.append(os.path.join(metadata_dir, input_args.o))
    else:
        output_parent_dir = os.path.join(metadata_dir, metadata_name + '_output')
        # output_dirs.append(os.path.join(metadata_dir, metadata_name + '_output'))

    output_dirs = {'output_parent': output_parent_dir,
                   'output_individual': os.path.join(output_parent_dir, 'individual'),
                   'output_summary': os.path.join(output_parent_dir, 'summary')}

    # make folders if they don't exist
    if not os.path.isdir(output_parent_dir):
        os.mkdir(output_parent_dir)

    for key, folder in output_dirs.items():
        if key is not 'output_parent':
            if not os.path.isdir(folder):
                os.mkdir(folder)


    metadata = pd.read_excel(metadata_path)

    return metadata, output_dirs


def analyze_replicate(metadata, input_args):
    sample_name = np.unique(metadata['experiment_name'])[0]
    replicate_name = np.unique(metadata['replicate'])[0]
    channels = np.unique(metadata['channel_id'])
    num_of_channels = len(channels)

    if num_of_channels > 2:
        print('ERROR: Currently can only do up to 2 channels')
        sys.exit(0)

    scaffold_image_flag = False
    client_image_flag = False
    avg_image_flag = False

    bsub_flag = False
    if input_args.b > 0:
        background_value_to_subtract = input_args.b
        bsub_flag = True

    if type(input_args.s) is int:
        scaffold_channel_name = input_args.s
        scaffold_test = [b == scaffold_channel_name for b in channels]

        for idx, c in enumerate(channels):
            if scaffold_test[idx]:
                scaffold_image_path = metadata['image_path'][idx]
                scaffold = io.imread(scaffold_image_path)
                scaffold = img_as_float(scaffold)
                client_a = scaffold
                scaffold_image_flag = True
            else:
                client_b_image_path = metadata['image_path'][idx]
                client_b = io.imread(client_b_image_path)
                client_b = img_as_float(client_b)
                client_b_image_flag = True

    elif type(input_args.s) is str:
        if input_args.s is 'avg':
            for idx, c in enumerate(channels):
                count = 0
                if idx == 0:
                    avg_image = io.imread(metadata['image_path'][idx])
                    avg_image = img_as_float(avg_image)
                    client_a = avg_image
                    count = count + 1
                else:
                    image_to_add = io.imread(metadata['image_path'][idx])
                    image_to_add = img_as_float(image_to_add)
                    client_b = image_to_add

                    client_b_image_flag = True

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
        crop_shape = (2*c_width, 2*c_width)
    else:
        crop_mask.fill(True)
        crop_shape = scaffold.shape

    scaffold = scaffold[crop_mask].reshape(crop_shape)
    client_a = client_a[crop_mask].reshape(crop_shape)

    if client_b_image_flag:
        client_b = client_b[crop_mask].reshape(crop_shape)
        client_a = client
        client_b = scaffold

    # background subtraction
    if input_args.bsub_flag:
        scaffold = scaffold - background_value_to_subtract
        client_a = client_a - background_value_to_subtract

        if client_b_image_flag:
            client_b = client_b - background_value_to_subtract

    # find std of image for later thresholding
    scaffold_std = np.std(scaffold)
    client_a_std = np.std(client_a)
    client_b_std = np.std(client_b)

    threshold_multiplier = input_args.tm

    # make binary image of scaffold with threshold intensity. Threshold is multiplier of std above background

    scaffold_mean = np.mean(scaffold)
    client_a_mean = np.mean(client_a)
    if client_b_image_flag:
        client_b_mean = np.mean(client_b)

    binary_mask = np.full(shape=(scaffold.shape[0], scaffold.shape[1]), fill_value=False, dtype=bool)
    scaffold_binary = np.full(shape=(scaffold.shape[0], scaffold.shape[1]), fill_value= False, dtype=bool)

    binary_mask[scaffold > (scaffold_mean + (threshold_multiplier * scaffold_std))] = True
    scaffold_binary[binary_mask] = True

    scaffold_binary = ndi.morphology.binary_fill_holes(scaffold_binary)
    scaffold_binary_labeled = measure.label(scaffold_binary)
    scaffold_regionprops = measure.regionprops(scaffold_binary_labeled)

    # filter droplets by size and circularity
    min_area_threshold = input_args.min_a
    max_area_threshold = input_args.max_a
    circ_threshold = input_args.circ
    subset_area = input_args.r

    subset_area_less_than_min_area_flag = False
    if subset_area < min_area_threshold:
        subset_area_less_than_min_area_flag = True

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
                                                   'area', 'centroid_r', 'centroid_c', 'circularity'])
    elif num_of_channels == 2:
        replicate_output = pd.DataFrame(columns = ['sample', 'replicate', 'droplet_id',
                                                   'subset_I_'+str(channels[0]), 'subset_I_'+str(channels[1]),
                                                   'mean_I_' + str(channels[0]), 'mean_I_' + str(channels[1]),
                                                   'max_I_' + str(channels[0]), 'max_I_' + str(channels[1]),
                                                   'area', 'centroid_r', 'centroid_c', 'circularity'])

    for i, region in enumerate(scaffold_filtered_regionprops):
        s = sample_name
        r = replicate_name
        droplet_id = i
        area = region.area

        use_min_area_flag = False
        if subset_area_less_than_min_area_flag:
            if area < subset_area:
                use_min_area_flag = True

        centroid_r, centroid_c = region.centroid
        circularity = circ(region)
        coordinates = region.coords
        coords_r = coordinates[:, 0]
        coords_c = coordinates[:, 1]

        if use_min_area_flag:
            subset_coords_r = coords_r
            subset_coords_c = coords_c
        else:
            subset_coords_r, subset_coords_c = draw.circle(r=centroid_r, c=centroid_c,
                                                           radius=round(math.sqrt(subset_area)))

        # in cases where droplets are near the edge, the circle will go beyond the image. In that case,
        # we simply ignore the droplet
        edge_r_test = all(0 < r < scaffold.shape[0] for r in subset_coords_r)
        edge_c_test = all(0 < c < scaffold.shape[1] for c in subset_coords_c)

        if edge_r_test and edge_c_test:
            if num_of_channels == 1:
                mean_intensity = region.mean_intensity * 65536
                max_intensity = region.max_intensity * 65536
                subset_intensity = np.mean(client_a[subset_coords_c, subset_coords_r]) * 65536

                replicate_output = replicate_output.append({'sample': s, 'replicate': r,
                                                            'droplet_id': droplet_id,
                                                            'subset_I_' + str(channels): subset_intensity,
                                                            'mean_I_' + str(channels): mean_intensity,
                                                            'max_I_' + str(channels): max_intensity,
                                                            'area': area, 'centroid_r': centroid_r, 'centroid_c': centroid_c,
                                                            'circularity': circularity},
                                                           ignore_index=True)

            elif num_of_channels == 2:
                mean_intensity_a = np.mean(client_a[coords_r, coords_c]) * 65536
                mean_intensity_b = np.mean(client_b[coords_r, coords_c]) * 65536

                max_intensity_a = np.max(client_a[coords_r, coords_c]) * 65536
                max_intensity_b = np.max(client_b[coords_r, coords_c]) * 65536

                subset_intensity_a = np.mean(client_a[subset_coords_r, subset_coords_c]) * 65536
                subset_intensity_b = np.mean(client_b[subset_coords_r, subset_coords_c]) * 65536

                replicate_output = replicate_output.append({'sample': s, 'replicate': r, 'droplet_id': droplet_id,
                                                            'subset_I_'+str(channels[0]): subset_intensity_a,
                                                            'subset_I_'+str(channels[1]): subset_intensity_b,
                                                            'mean_I_' + str(channels[0]): mean_intensity_a,
                                                            'mean_I_' + str(channels[1]): mean_intensity_b,
                                                            'max_I_' + str(channels[0]): max_intensity_a,
                                                            'max_I_' + str(channels[1]): max_intensity_b,
                                                            'area': area, 'centroid_r': centroid_r, 'centroid_c': centroid_c,
                                                            'circularity': circularity},
                                                           ignore_index=True)

    # get measurements of bulk regions excluding droplets
    bulk_mask = np.invert(scaffold_mask)
    bulk_I = []
    if num_of_channels == 1:
        bulk_I.append(np.mean(scaffold[bulk_mask]) * 65536)
    elif num_of_channels == 2:
        bulk_I.append(np.mean(client_a[bulk_mask]) * 65536)
        bulk_I.append(np.mean(client_b[bulk_mask]) * 65536)

    return replicate_output, bulk_I


def subtract_background(input_image):
    image_hist, image_bin_edges = np.histogram(input_image, bins='auto')
    background_threshold = image_bin_edges[np.argmax(image_hist)]  # assumes that the max hist peak corresponds to background pixels
    output_image = input_image - background_threshold
    output_image[output_image < 0] = 0

    # output_image = np.reshape(output_image, input_image.shape)
    return output_image, background_threshold


def analyze_sample(metadata, input_args, replicate_output, bulk_I):
    sample_name = np.unique(metadata['experiment_name'])
    channels = np.unique(metadata['channel_id'])
    num_of_channels = len(channels)

    if num_of_channels == 1:
        print('Done')
    elif num_of_channels == 2:
        for idx, r in enumerate(replicate_output['replicate']):
            return
           # print('JON START HERE')