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
from matplotlib import patches
import argparse
import json
from datetime import datetime
import cv2
from types import SimpleNamespace



def parse_arguments(parser):

    # required arguments
    parser.add_argument('parent_path', type=str,
                        help='Full path to folder that contains subfolders of experiments with data')
    parser.add_argument('output_path', type=str,
                        help='Full path to folder where you want the output to be stored. The folder will be made if it does not exist')

    # optional arguments
    parser.add_argument("--tm", type=float, default=3.0,
                        help='Optional threshold multiplier. Defaults to 3. mean + std*tm')
    parser.add_argument("--r", type=float, default=30,
                        help='Area of subset circle to use in middle of droplet. Default 30 px^2. Per droplet, --min_a supercedes --r')
    parser.add_argument("--min_a", type=float, default=20,
                        help='Optional threshold for minimum droplet area. Default 20 px^2')
    parser.add_argument("--max_a", type=float, default=500,
                        help='Optional threshold for max droplet area. Default is 500 px^2')
    parser.add_argument("--circ", type=float, default=0.8,
                        help='Optional threshold for droplet circularity (defined 0.0-1.0). Default is 0.8')
    parser.add_argument("--s",
                        help='What channel to use for scaffolding. Defaults to standardized average of all channels.')
    parser.add_argument("--b", type=float, default=0.0,
                        help='Optional absolute value to use to subtract background. Default is 0.0.')
    parser.add_argument("--pr", type=str, default='subset',
                        help='Value to use for [C](in) to calculate partition ratio. Options are subset, mean, and max. Default is subset')
    parser.add_argument('--no-image', dest='output_image_flag', action='store_false', default=True,
                        help='Flag to set if you do not want output images of the droplets saved to the output directory')
    parser.add_argument('--rand-bulk', dest='randomize_bulk_flag', action='store_true', default=False,
                        help='Flag to calculate bulk by randomzing the image and taking the average intensity. NOT YET IMPLEMENTED')
    parser.add_argument('--bf', dest='bf_flag', action='store_true', default=False,
                        help='Flag to include DIC brightfield as the scaffold')

    input_params = parser.parse_args()

    return input_params


def load_images(replicate_files, data, input_params, folder):
    # get replicate sample name
    nd_file_name = [n for n in replicate_files if '.nd' in n]
    if len(nd_file_name) == 1:
        sample_name = get_sample_name(nd_file_name[0])
        data.sample_name = sample_name
    else:
        print('Error: Found too many .nd files in sample directory')
        sys.exit(0)

    print(sample_name)

    # load images
    channel_image_files = [c for c in replicate_files if get_file_extension(c) == '.TIF']

    if len(channel_image_files) < 1:
        print('Error: Could not find image files')
        sys.exit(0)

    channel_image_paths = []
    channel_images = []
    channel_names = []
    for idx, p in enumerate(channel_image_files):
        channel_image_paths.append(os.path.join(input_params.parent_path, folder, p))
        channel_names.append(find_image_channel_name(p))
        channel_images.append(img_as_float(io.imread(channel_image_paths[idx])))

    data.channel_images = channel_images
    data.channel_names = channel_names

    return data


def make_output_directories(input_params):

    output_parent_dir = input_params.output_path

    output_dirs = {'output_parent': output_parent_dir,
                   'output_individual': os.path.join(output_parent_dir, 'individual'),
                   'output_summary': os.path.join(output_parent_dir, 'summary'),
                   'output_individual_images': os.path.join(output_parent_dir,'individual','droplet_images')}

    # make folders if they don't exist
    if not os.path.isdir(output_parent_dir):
        os.mkdir(output_parent_dir)

    for key, folder in output_dirs.items():
        if key is not 'output_parent':
            if not os.path.isdir(folder):
                if not os.path.isdir(os.path.dirname(folder)):  # so I guess .items() is random order of dictionary keys. So when making subfolders, if the parent doesn't exist, then we would get an error. This accounts for that.
                        os.mkdir(os.path.dirname(folder))

                os.mkdir(folder)

    input_params.output_dirs = output_dirs

    return input_params


def find_scaffold(data, input_params):
    scaffold = np.zeros(shape=data.channel_images[0].shape, dtype=np.float)
    scaffold = scaffold - input_params.b
    num_of_channels = len(data.channel_names)

    # identify what value to use for [C](in) in partition ratio calculation
    pr_parameter = input_params.pr
    if pr_parameter != 'subset':
        if pr_parameter != 'mean':
            if pr_parameter != 'max':
                print('ERROR: Could not identify user input for value to use to calculate partition ratio')
                sys.exit(0)

    if input_params.bf_flag:  # if you want to use BF as the scaffold
        scaffold_channel_name = 'chDIC'
        scaffold_test = [b == scaffold_channel_name for b in data.channel_names]

        for idx, c in enumerate(data.channel_names):
            if scaffold_test[idx]:
                scaffold = scaffold + data.channel_images[idx]
                if input_params.b > 0.0:
                    print('Error: Cannot do background subtract on brightfield image')
                    sys.exit(0)

                scaffold = modify_bf_img(scaffold)
                data.scaffold_output_img = scaffold
                scaffold = standardize_img(scaffold)
    else:
        if input_params.s:  # when the user specifies a scaffold channel
            scaffold_channel_name = 'ch' + input_params.s
            scaffold_test = [b == scaffold_channel_name for b in data.channel_names]

            for idx, c in enumerate(data.channel_names):
                if scaffold_test[idx]:
                    scaffold = scaffold + data.channel_images[idx]
                    scaffold[np.where(scaffold < 0)] = 0  # to correct for background subtraction
                    data.scaffold_output_img = scaffold
                    scaffold = standardize_img(scaffold)

        else:  # default using the average scaffold

            for img in data.channel_images:
                scaffold = scaffold + img/num_of_channels

            scaffold[np.where(scaffold < 0)] = 0  # to correct for background subtraction
            data.scaffold_output_img = scaffold
            scaffold = standardize_img(scaffold)

    data.scaffold = scaffold

    return data


def modify_bf_img(img):
    #process brightfield image that should already be type float
    median_img = img_as_uint(img)
    median_img = cv2.medianBlur(median_img, ksize=5)
    median_img = img_as_float(median_img)

    img = img - median_img
    img[np.where(img < 0)] = 0

    return img


def find_droplets(data, input_params):
    threshold_multiplier = input_params.tm
    scaffold = data.scaffold
    channels = data.channel_names

    # make binary image of scaffold with threshold intensity. Threshold is multiplier of std above background.
    # Since we have already standardized the image, the threshold is simply the value of the image.

    binary_mask = np.full(shape=(scaffold.shape[0], scaffold.shape[1]), fill_value=False, dtype=bool)
    scaffold_binary = np.full(shape=(scaffold.shape[0], scaffold.shape[1]), fill_value= False, dtype=bool)

    binary_mask[scaffold > threshold_multiplier] = True
    scaffold_binary[binary_mask] = True

    if input_params.bf_flag:
        scaffold_binary = ndi.morphology.binary_fill_holes(scaffold_binary)
        scaffold_binary = ndi.morphology.binary_opening(scaffold_binary)
        scaffold_binary = ndi.morphology.binary_dilation(scaffold_binary)
    else:
        scaffold_binary = ndi.morphology.binary_fill_holes(scaffold_binary)

    scaffold_binary_labeled = measure.label(scaffold_binary)
    scaffold_regionprops = measure.regionprops(scaffold_binary_labeled)

    # filter droplets by size and circularity
    min_area_threshold = input_params.min_a
    max_area_threshold = input_params.max_a
    circ_threshold = input_params.circ
    subset_area = input_params.r

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



    # get measurements of bulk regions excluding droplets and total intensity of entire image (including droplets)
    # Not implemented yet
    if input_params.randomize_bulk_flag:
        print('Randomized bulk not implemented yet')
        sys.exit(0)

        # num_of_iterations = 100
        #
        # total_I = []
        #
        # if num_of_channels == 1:
        #     rand_scaffold_storage = np.zeros(shape=(scaffold.shape[0] * scaffold.shape[1], num_of_iterations))
        #
        #     scaffold_1d = np.reshape(scaffold, (scaffold.shape[0] * scaffold.shape[1]))
        #
        #     for n in range(num_of_iterations):
        #         rand_scaffold_storage[:,n] = np.random.shuffle(scaffold_1d)
        #
        #     scaffold_random_average_image = np.reshape(np.mean(rand_scaffold_storage, axis=1), shape=scaffold.shape)
        #
        #     bulk_I = []
        #     bulk_I.append(np.mean(scaffold_random_average_image) * 65536)
        #     total_I.append(np.sum(scaffold) * 65536)
        #
        # elif num_of_channels == 2:
        #     rand_client_a_storage = np.zeros(shape=(client_a.shape[0] * client_a.shape[1], num_of_iterations))
        #     rand_client_b_storage = np.zeros(shape=(client_b.shape[0] * client_b.shape[1], num_of_iterations))
        #
        #     # doc on shuffle: multi-dimensional arrays are only shuffled along the first axis
        #     # so let's make the image an array of (N) instead of (m,n)
        #     client_a_1d = np.reshape(client_a, (client_a.shape[0] * client_a.shape[1]))
        #     client_b_1d = np.reshape(client_b, (client_b.shape[0] * client_b.shape[1]))
        #
        #     rand_client_a_sum = np.zeros(shape=(1, client_a.shape[0] * client_a.shape[1]))
        #     rand_client_b_sum = np.zeros(shape=(1, client_b.shape[0] * client_b.shape[1]))
        #     for n in range(num_of_iterations):
        #         # rand_client_a_storage[n,:] = np.random.shuffle(client_a_1d)
        #         # rand_client_b_storage[n,:] = np.random.shuffle(client_b_1d)
        #         np.random.shuffle(client_a_1d)
        #         np.random.shuffle(client_b_1d)
        #         rand_client_a_sum = rand_client_a_sum + client_a_1d
        #         rand_client_b_sum = rand_client_b_sum + client_b_1d
        #
        #     # client_a_random_average_image = np.reshape(np.mean(rand_client_a_storage, axis=1), client_a.shape)
        #     client_a_random_average_image = np.reshape(rand_client_a_sum/num_of_iterations, client_a.shape)
        #     client_b_random_average_image = np.reshape(rand_client_b_sum/num_of_iterations, client_b.shape)
        #
        #     # client_b_random_average_image = np.reshape(np.mean(rand_client_b_storage, axis=1), client_b.shape)
        #
        #     bulk_I = []
        #     bulk_I.append(np.mean(client_a_random_average_image) * 65536)
        #     bulk_I.append(np.mean(client_b_random_average_image) * 65536)
        #
        #     total_I.append(np.sum(client_a) * 65536)
        #     total_I.append(np.sum(client_b) * 65536)
        #
        # if num_of_channels == 1:
        #     random_bulk_image = client_a_random_average_image
        # elif num_of_channels == 2:
        #     random_bulk_image = np.zeros(shape=(scaffold.shape[0], scaffold.shape[0], 3))
        #     # we will assume that first channel is green and second channel is magenta
        #     random_bulk_image[..., 0] = client_b_random_average_image  # R
        #     random_bulk_image[..., 1] = client_a_random_average_image  # G
        #     random_bulk_image[..., 2] = client_b_random_average_image  # B
    else:
        bulk_mask = np.invert(scaffold_mask)
        bulk_I = np.zeros(len(channels))
        total_I = np.zeros(len(channels))

        for idx, c in enumerate(channels):
            bulk_I[idx] = np.mean(scaffold[bulk_mask]) * 65536
            total_I[idx] = np.sum(scaffold) * 65536

    # get measurements of regions from centered circle with radius sqrt(minimum droplet area)
    # what we want:
    # area, centroid, circularity, mean_intensity, max_intensity, mean intensity inside circle

    # initialize labeled image to generate output of what droplets were called
    label_image = np.full(shape=(scaffold.shape[0], scaffold.shape[1]), fill_value=False, dtype=bool)
    droplet_id_list = []
    droplet_id_centroid_r = []
    droplet_id_centroid_c = []

    # iterate over regions to collect information on individual droplets
    s = data.sample_name
    r = input_params.replicate_count

    if len(scaffold_filtered_regionprops) > 0:
        for i, region in enumerate(scaffold_filtered_regionprops):
            area = region.area
            use_min_area_flag = False  # this is if the subset area is less than the min droplet area parameter. In this case, we just use the min area.
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
                label_image[coords_r, coords_c] = True
                droplet_id = i
                droplet_id_list.append(droplet_id)
                droplet_id_centroid_r.append(centroid_r)
                droplet_id_centroid_c.append(centroid_c)

                for c_idx, img in enumerate(data.channel_images):
                    mean_intensity = np.mean(img[coords_r, coords_c]) * 65536
                    max_intensity = np.max(img[coords_r, coords_c]) * 65536

                    subset_intensity = np.mean(img[subset_coords_r, subset_coords_c]) * 65536
                    total_intensity = np.sum(img[coords_r, coords_c]) * 65536

                    if input_params.pr == 'sub':
                        partition_ratio = subset_intensity/bulk_I[0]
                    elif input_params.pr == 'mean':
                        partition_ratio = mean_intensity/bulk_I[0]
                    elif input_params.pr== 'max':
                        partition_ratio = max_intensity/bulk_I[0]
                    else:
                        partition_ratio = -2  # just a sanity check. Should never happen.

                    replicate_output = pd.concat(replicate_output,
                                                 pd.DataFrame({'sample': s, 'replicate': r,
                                                               'droplet_id': droplet_id,
                                                               'subset_I_' + str(channels[c_idx]): subset_intensity,
                                                               'mean_I_' + str(channels[c_idx]): mean_intensity,
                                                               'max_I_' + str(channels[c_idx]): max_intensity,
                                                               'total_I_' + str(channels[c_idx]): total_intensity,
                                                               'bulk_I_' + str(channels[c_idx]): bulk_I[c_idx],
                                                               'partition_ratio_' + str(channels[c_idx]): partition_ratio,
                                                               'area': area, 'centroid_r': centroid_r, 'centroid_c': centroid_c,
                                                               'circularity': circularity}),
                                                 sort=False)


    else:
        for c in channels:
            replicate_output = pd.concat(replicate_output, pd.DataFrame({'sample': s, 'replicate': r,
                                                        'droplet_id': 0,
                                                        'subset_I_' + str(c): 0.0,
                                                        'mean_I_' + str(c): 0.0,
                                                        'max_I_' + str(c): 0.0,
                                                        'total_I_' + str(c): 0.0,
                                                        'bulk_I_' + str(c): 0.0,
                                                        'partition_ratio_' + str(c): 0.0,
                                                        'area': 0.0, 'centroid_r': 0.0,
                                                        'centroid_c': 0.0,
                                                        'circularity': 0.0}),
                                                       sort=False)

    data.label_image = label_image
    data.replicate_output = replicate_output
    data.bulk_I = bulk_I
    data.total_I = total_I

    if input_params.output_image_flag:
        if input_params.randomize_bulk_flag:
            pass
            # make_droplet_image(output_dirs['output_individual_images'], orig_image, scaffold, label_image,
            #                    num_of_channels, str(s) + '_' + str(r), droplet_id_list, droplet_id_centroid_c, droplet_id_centroid_r,
            #                    input_args, random_bulk_image=random_bulk_image)
        else:
            make_droplet_image(input_params.output_dirs['output_individual_images'], data, droplet_id_list, droplet_id_centroid_r,
                               droplet_id_centroid_c, input_params)

    return data


def subtract_background(input_image):
    image_hist, image_bin_edges = np.histogram(input_image, bins='auto')
    background_threshold = image_bin_edges[np.argmax(image_hist)]  # assumes that the max hist peak corresponds to background pixels
    output_image = input_image - background_threshold
    output_image[output_image < 0] = 0

    # output_image = np.reshape(output_image, input_image.shape)
    return output_image, background_threshold


def analyze_sample(metadata, input_args, replicate_output, bulk_I, total_I):
    sample_name = np.unique(metadata['experiment_name'])[0]
    channels = np.unique(metadata['channel_id'])
    num_of_channels = len(channels)

    # get number of replicates
    replicates = np.unique(metadata['replicate'])

    # initialize outputs
    if num_of_channels == 1:
        sample_output = pd.DataFrame(columns=['sample', 'partition_ratio_mean_' + str(channels[0]),
                                              'partition_ratio_std_' + str(channels[0]),
                                              'condensed_fraction_mean_' + str(channels[0]),
                                              'condensed_fraction_std_' + str(channels[0])])
    elif num_of_channels == 2:
        sample_output = pd.DataFrame(columns=['sample', 'partition_ratio_mean_' + str(channels[0]),
                                              'partition_ratio_std_'            + str(channels[0]),
                                              'condensed_fraction_mean_'        + str(channels[0]),
                                              'condensed_fraction_std_'         + str(channels[0]),
                                              'partition_ratio_mean_'           + str(channels[1]),
                                              'partition_ratio_std_'            + str(channels[1]),
                                              'condensed_fraction_mean_'        + str(channels[1]),
                                              'condensed_fraction_std_'         + str(channels[1])])
    if len(replicate_output) > 0:
        replicate_partition_ratio = []
        replicate_condensed_fraction = []

        if num_of_channels == 1:
            for idx, r in enumerate(replicates):
                mean_partition_ratio = np.mean(
                    replicate_output['partition_ratio_' + str(channels[0])][replicate_output['replicate'] == r])

                replicate_partition_ratio.append(mean_partition_ratio)

                condensed_fraction = np.sum(
                    replicate_output['total_I_' + str(channels[0])][replicate_output['replicate'] == r])/total_I[idx]

                replicate_condensed_fraction.append(condensed_fraction)

            sample_partition_ratio_mean = np.mean(replicate_partition_ratio)
            sample_partition_ratio_std = np.std(replicate_partition_ratio)

            sample_condensed_fraction_mean = np.mean(replicate_condensed_fraction)
            sample_condensed_fraction_std = np.std(replicate_condensed_fraction)

            sample_output = sample_output.append({'sample': sample_name,
                                                  'partition_ratio_mean_' + str(channels[0]): sample_partition_ratio_mean,
                                                  'partition_ratio_std_' + str(channels[0]): sample_partition_ratio_std,
                                                  'condensed_fraction_mean_' + str(channels[0]): sample_condensed_fraction_mean,
                                                  'condensed_fraction_std_' + str(channels[0]): sample_condensed_fraction_std},
                                                  ignore_index=True)

        elif num_of_channels == 2:
            count = 0
            for idx, r in enumerate(replicates):
                # client_a
                mean_partition_ratio = np.mean(
                    replicate_output['partition_ratio_' + str(channels[0])][replicate_output['replicate'] == r])
                replicate_partition_ratio.append(mean_partition_ratio)

                condensed_fraction = np.sum(
                    replicate_output['total_I_' + str(channels[0])][replicate_output['replicate'] == r]) / total_I[count]
                replicate_condensed_fraction.append(condensed_fraction)

                # client_b
                mean_partition_ratio = np.mean(
                    replicate_output['partition_ratio_' + str(channels[1])][replicate_output['replicate'] == r])
                replicate_partition_ratio.append(mean_partition_ratio)

                condensed_fraction = np.sum(
                    replicate_output['total_I_' + str(channels[1])][replicate_output['replicate'] == r]) / total_I[count+1]  #@Bug Should this be +1? I think so, but need to check.
                replicate_condensed_fraction.append(condensed_fraction)

                count = count + 2  # 2 here because we append both total_I's to the same list

            sample_client_a_partition_ratio_mean = np.mean(replicate_partition_ratio[::2])  # every other since we append both clients
            sample_client_a_partition_ratio_std = np.std(replicate_partition_ratio[::2])

            sample_client_a_condensed_fraction_mean = np.mean(replicate_condensed_fraction[::2])
            sample_client_a_condensed_fraction_std = np.std(replicate_condensed_fraction[::2])

            sample_client_b_partition_ratio_mean = np.mean(
                replicate_partition_ratio[1::2])  # every other since we append both clients
            sample_client_b_partition_ratio_std = np.std(replicate_partition_ratio[1::2])

            sample_client_b_condensed_fraction_mean = np.mean(replicate_condensed_fraction[1::2])
            sample_client_b_condensed_fraction_std = np.std(replicate_condensed_fraction[1::2])

            sample_output = sample_output.append({'sample': sample_name,
                                                  'partition_ratio_mean_' +    str(channels[0]): sample_client_a_partition_ratio_mean,
                                                  'partition_ratio_std_' +     str(channels[0]): sample_client_a_partition_ratio_std,
                                                  'condensed_fraction_mean_' + str(channels[0]): sample_client_a_condensed_fraction_mean,
                                                  'condensed_fraction_std_' +  str(channels[0]): sample_client_a_condensed_fraction_std,
                                                  'partition_ratio_mean_' +    str(channels[1]): sample_client_b_partition_ratio_mean,
                                                  'partition_ratio_std_' +     str(channels[1]): sample_client_b_partition_ratio_std,
                                                  'condensed_fraction_mean_' + str(channels[1]): sample_client_b_condensed_fraction_mean,
                                                  'condensed_fraction_std_' +  str(channels[1]): sample_client_b_condensed_fraction_std
                                                  }, ignore_index=True)

    else:
        if num_of_channels == 1:
            sample_output = sample_output.append({'sample': sample_name,
                                                  'partition_ratio_mean_' + str(channels[0]): 0.0,
                                                  'partition_ratio_std_' + str(channels[0]): 0.0,
                                                  'condensed_fraction_mean_' + str(channels[0]): 0.0,
                                                  'condensed_fraction_std_' + str(channels[0]): 0.0},
                                                  ignore_index=True)
        elif num_of_channels == 2:
            sample_output = sample_output.append({'sample': sample_name,
                                                  'partition_ratio_mean_' + str(
                                                      channels[0]): 0.0,
                                                  'partition_ratio_std_' + str(
                                                      channels[0]): 0.0,
                                                  'condensed_fraction_mean_' + str(
                                                      channels[0]): 0.0,
                                                  'condensed_fraction_std_' + str(
                                                      channels[0]): 0.0,
                                                  'partition_ratio_mean_' + str(
                                                      channels[1]): 0,
                                                  'partition_ratio_std_' + str(
                                                      channels[1]): 0.0,
                                                  'condensed_fraction_mean_' + str(
                                                      channels[1]): 0.0,
                                                  'condensed_fraction_std_' + str(
                                                      channels[1]): 0.0
                                                  }, ignore_index=True)

    return sample_output


def find_region_edge_pixels(a):  # this is a way to maybe find boundary pixels if we ever need to do that
    distance = ndi.distance_transform_edt(a)
    distance[distance != 1] = 0
    np.where(distance == 1)


def make_axes_blank(ax):
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


def make_droplet_image(output_path, data, droplet_list, droplet_r, droplet_c, input_params):

    fig, ax = plt.subplot(figsize=(3, 3))
    scaffold_image = exposure.rescale_intensity(data.scaffold_output_image)

    label = np.zeros(shape=data.label_image.shape)
    label[data.label_image] = 1

    region_overlay = color.label2rgb(label, image=scaffold_image,
                                     alpha=0.5, image_alpha=1, bg_label=0, bg_color=None)

    ax.imshow(region_overlay)

    ax.set_title(data.sample_name + '_rep' + str(input_params.replicate_count))
    make_axes_blank(ax)

    text_offset = 10
    droplet_r = [(int(round(r)) + text_offset) for r in droplet_r]
    droplet_c = [(int(round(c)) + text_offset) for c in droplet_c]

    for i, drop_id in enumerate(droplet_list):
        ax.text(droplet_r[i], droplet_c[i], drop_id, color='w', fontsize=4)

    plt.savefig(os.path.join(output_path, data.sample_name + '_rep' + str(input_params.replicate_count) + '.png'), dpi=300)
    plt.close()

    #
    # if random_bulk_image is not None:
    #     fig, ax = plt.subplots(nrows=1, ncols=2)
    #     orig_image = exposure.rescale_intensity(orig_image)
    #     random_bulk_image = exposure.rescale_intensity(random_bulk_image)
    #
    #     if num_of_channels == 1:
    #         ax[0].imshow(orig_image, cmap='gray')
    #         ax[1].imshow(random_bulk_image, cmap='gray')
    #     elif num_of_channels == 2:
    #         ax[0].imshow(orig_image)
    #         ax[1].imshow(random_bulk_image)
    #
    #     ax[0].set_title(name)
    #     ax[1].set_title('Randomized bulk image')
    #     make_axes_blank(ax[0])
    #     make_axes_blank(ax[1])
    #
    #     plt.savefig(os.path.join(output_path, name + '_randomized_bulk.png'))
    #     plt.close()


def find_image_channel_name(file_name):
    str_idx = file_name.find('Conf ')  # this is specific to our microscopes file name format
    channel_name = file_name[str_idx + 5 : str_idx + 8]

    channel_name = 'ch' + channel_name

    return channel_name


def circ(r):
    output = (4 * math.pi * r.area) / (r.perimeter * r.perimeter)

    return output


def get_sample_name(nd_file_name):
    sample_name, ext = os.path.splitext(nd_file_name)

    return sample_name

def get_file_extension(file_path):
    file_ext = os.path.splitext(file_path)

    return file_ext[1]  # because splitext returns a tuple, and the extension is the second element


def standardize_img(img):
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std

    return img