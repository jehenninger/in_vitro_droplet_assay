import helper
import grapher
import pandas as pd
import numpy as np
import os
import sys
#import matplotlib
# matplotlib.use('Agg')
# matplotlib.use('Qt5Agg')
# from matplotlib import pyplot as plt
import argparse
import json
from datetime import datetime
from skimage import io, filters, measure, color, exposure, morphology, feature, img_as_float, img_as_uint


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
parser.add_argument("--tm", type=float, default=3.0)  # optional threshold multiplier. Defaults to 3. Multiplies std plus background peak
parser.add_argument("--r", type=float, default=9.0)  # area of subset circle to use in middle of droplet
parser.add_argument("--min_a", type=float, default=9)  # optional threshold for minimum droplet area @Temporary
parser.add_argument("--max_a", type=float, default=500)  # optional threshold for max droplet area
parser.add_argument("--circ", type=float, default=0.8)  # optional threshold for droplet circularity
parser.add_argument("--s", default='avg')  # what channel to use for scaffolding. Defaults to average.
parser.add_argument("--b", type=float, default=0.0)  # background subtraction
parser.add_argument("--pr", type=str, default='sub')  # Value to use for [C](in) to calculate partition ratio.
    # Options: 'sub' for subset circle, 'mean' for mean intensity of whole droplet, 'max' for max intensity in droplet

# parser.add_argument("--s", default=561) # @Temporary
parser.add_argument('--crop', type=int)  # width from center point to include in pixels.
                                         # Defaults to entire image (width/2)

parser.add_argument('--no-image', dest='output_image_flag', action='store_false', default=True)  # flag to set whether output images of the droplets are saved to a directory
parser.add_argument('--rand-bulk', dest='randomize_bulk_flag', action='store_true', default=False)  # flag to calculate bulk by randomzing the image 100 times and taking the average intensity
parser.add_argument('--no-meta', dest='metadata_flag', action='store_false', default=True)  # flag to automatically parse experiment from folders instead of providing metadata.
# parser.add_argument('--no-bsub', dest='bsub_flag', action='store_false', default=True)  # @Deprecated


input_args = parser.parse_args()

# load and check metadata
# metadata_path = '/Users/jon/PycharmProjects/in_vitro_droplet_assay/test_MED_CTD/metadata.xlsx'
# metadata, output_dirs = helper.read_metadata(input_args, metadata_path)  # @Temporary

if input_args.metadata_flag:
    metadata, output_dirs = helper.read_metadata(input_args)
    # get number of unique experiments
    samples = np.unique(metadata['experiment_name'])
    num_of_samples = len(samples)

    channels = np.unique(metadata['channel_id'])
    num_of_channels = len(channels)
else:
    metadata = pd.DataFrame(columns=['image_path', 'experiment_name', 'replicate', 'channel_id'])
    # get number of experiments/sub-directories to analyze
    dir_list = os.listdir(input_args.metadata_path)
    dir_list.sort(reverse=False)
    file_ext = ".nd"


    samples = []
    for folder in dir_list:
        if not folder.startswith('.') and not folder.endswith('output') and \
            os.path.isdir(os.path.join(input_args.metadata_path, folder)):

            samples.append(folder)
            file_list = os.listdir(os.path.join(input_args.metadata_path, folder))

            base_name_files = [f for f in file_list if file_ext in f
                               and os.path.isfile(os.path.join(input_args.metadata_path, folder, f))]

            base_name_files.sort(reverse=False)

            count = 1
            for idx, file in enumerate(base_name_files):
                sample_name = file.replace(file_ext, '')
                replicate_files = [os.path.join(input_args.metadata_path, folder, r) for r in file_list if sample_name in r
                                   and os.path.isfile(os.path.join(input_args.metadata_path, folder, r))
                                   and file_ext not in r]
                replicate_files = np.sort(replicate_files)
                for rep in replicate_files:
                    metadata = metadata.append({'image_path' : rep,
                                                'experiment_name' : folder,
                                                'replicate' : count,
                                                'channel_id' : int(helper.find_image_channel_name(rep))
                                                }, ignore_index=True)
                count += 1

            num_of_channels = len(replicate_files)

    output_dirs = helper.make_output_directories(input_args.metadata_path, 'metadata', input_args)

replicate_writer = pd.ExcelWriter(os.path.join(output_dirs['output_individual'], 'individual_droplet_output.xlsx'),
                                  engine='xlsxwriter')

sample_writer = pd.ExcelWriter(os.path.join(output_dirs['output_summary'], 'summary_droplet_output.xlsx'),
                                  engine='xlsxwriter')

graph_input = list()

sample_count = 0
for s in samples:
    print()
    print("Sample: ", s)

    metadata_sample = metadata[(metadata['experiment_name'] == s)].copy()
    metadata_sample = metadata_sample.reset_index(drop=True)

    # get number of replicates
    replicates = np.unique(metadata_sample['replicate'])
    num_of_replicates = len(replicates)

    count = 0
    for r in replicates:
        # print('replicate: ', r)

        metadata_replicate = metadata_sample[metadata_sample['replicate'] == r].copy()
        metadata_replicate = metadata_replicate.reset_index(drop=True)

        temp_rep, temp_bulk, temp_total = helper.analyze_replicate(metadata_replicate, input_args, output_dirs)

        if count == 0:
            replicate_output = temp_rep.copy()
            bulk_I = temp_bulk
            total_I = temp_total
            count = count + 1
        else:
            replicate_output = replicate_output.append(temp_rep, ignore_index=True)
            bulk_I = bulk_I + temp_bulk
            total_I = total_I + temp_total
            count = count + 1

    replicate_output.to_excel(replicate_writer, sheet_name=s, index=False)
    graph_input.append(replicate_output)

    if len(replicate_output > 0):
        grapher.make_droplet_size_histogram(replicate_output, output_dirs, input_args)

        if num_of_channels == 2:
            grapher.make_droplet_intensity_scatter(replicate_output, output_dirs, input_args)

    temp_sample_output = helper.analyze_sample(metadata_sample, input_args, replicate_output, bulk_I, total_I)

    if sample_count == 0:
        sample_output = temp_sample_output.copy()
        sample_count = sample_count + 1
    else:
        sample_output = sample_output.append(temp_sample_output, ignore_index=True)
        sample_count = sample_count + 1

    print('Finished at: ', datetime.now())


sample_output.to_excel(sample_writer, sheet_name='summary', index=False)

# adjust width of Excel columns in output to make for easier reading before writing the file
for key, sheet in replicate_writer.sheets.items():
    for idx, name in enumerate(replicate_output.columns):
        col_width = len(name) + 2
        sheet.set_column(idx, idx, col_width)

for key, sheet in sample_writer.sheets.items():
    for idx, name in enumerate(sample_output.columns):
        col_width = len(name) + 2
        sheet.set_column(idx, idx, col_width)

replicate_writer.save()
sample_writer.save()

# make boxplot with all droplets
if len(graph_input) > 0:
    grapher.make_droplet_boxplot(graph_input, output_dirs, input_args)

if len(sample_output) > 0:
    grapher.make_average_sample_graph(sample_output, output_dirs, input_args)

print()
print('Finished making plots at: ', datetime.now())

# write parameters that were used for this analysis
output_params = {'metadata_file'    : input_args.metadata_path,
                 'time_of_analysis' : datetime.now(),
                 'tm'               : input_args.tm,
                 'r'                : input_args.r,
                 'min_a'            : input_args.min_a,
                 'max_a'            : input_args.max_a,
                 'circ'             : input_args.circ,
                 's'                : input_args.s,
                 'b'                : input_args.b,
                 'pr'               : input_args.pr,
                 'crop'             : input_args.crop}

with open(os.path.join(output_dirs['output_parent'], 'output_analysis_parameters.txt'), 'w') as file:
        file.write(json.dumps(output_params, default=str))