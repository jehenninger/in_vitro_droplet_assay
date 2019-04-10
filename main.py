#!/Users/jon/PycharmProjects/in_vitro_droplet_assay/venv/bin/python

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family'] = 'sans-serif'

import in_vitro_droplet_assay.methods as methods
import in_vitro_droplet_assay.grapher as grapher
import pandas as pd
import numpy as np
import os
import sys
import math
#import matplotlib
# matplotlib.use('Agg')
# matplotlib.use('Qt5Agg')
# from matplotlib import pyplot as plt
import argparse
import json
from datetime import datetime
from skimage import io, filters, measure, color, exposure, morphology, feature, img_as_float, img_as_uint


# This is written so that all replicates for a given experiment are in a folder together (both .TIF and .nd files)

# parse input
parser = argparse.ArgumentParser()

input_params = methods.parse_arguments(parser)
input_params.parent_dir = input_params.parent_dir.replace("Volumes","lab")
input_params.output_path = input_params.output_path.replace("Volumes","lab")

input_params = methods.make_output_directories(input_params)

# get number of experiments/sub-directories to analyze
dir_list = os.listdir(input_params.parent_path)
dir_list.sort(reverse=False)
file_ext = ".nd"

samples = []
# this loops over EXPERIMENT FOLDERS
for folder in dir_list:
    if not folder.startswith('.') and os.path.isdir(os.path.join(input_params.parent_path, folder)):
        print()
        print(f'Sample: {folder}')

        file_list = os.listdir(os.path.join(input_params.parent_path, folder))

        base_name_files = [f for f in file_list if file_ext in f
                           and os.path.isfile(os.path.join(input_params.parent_path, folder, f))]

        base_name_files.sort(reverse=False)

        rep_count = 1

        replicate_writer = pd.ExcelWriter(
            os.path.join(input_params.output_dirs['output_individual'], 'individual_droplet_output.xlsx'),
            engine='xlsxwriter')

        # this loops over REPLICATES
        for idx, file in enumerate(base_name_files):
            sample_name = file.replace(file_ext, '')
            replicate_files = [os.path.join(input_params.parent_path, folder, r) for r in file_list if sample_name in r
                               and os.path.isfile(os.path.join(input_params.parent_path, folder, r))
                               and file_ext not in r]

            replicate_files = np.sort(replicate_files)

            data = methods.load_images(replicate_files, input_params, folder)

            temp_rep, temp_bulk, temp_total = methods.analyze_replicate(data, input_params)



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

    sheet_name = s
    if len(sheet_name) > 30:
        total_length = len(sheet_name)
        start_idx = math.floor((total_length - 30)/2)
        stop_idx = total_length - start_idx
        sheet_name = sheet_name[start_idx:stop_idx]

    replicate_output.to_excel(replicate_writer, sheet_name=sheet_name, index=False)
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