#!/lab/solexa_young/scratch/jon_henninger/tools/venv/bin/python

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family'] = 'sans-serif'

import sys
import os
import json
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import re


def exponential_curve(x, a, b, c):
    return a / (1 + np.exp(-b*(x-c)))


def read_metadata(input_args):
    metadata_dir = input_args.metadata_dir
    metadata_dir_name = os.path.basename(os.path.normpath(metadata_dir))

    output_dirs = []

    if input_args.o:
        output_dirs.append(os.path.join(metadata_dir, input_args.o))
    else:
        output_dirs.append(os.path.join(metadata_dir, metadata_dir_name + '_output'))

    if not os.path.isdir(output_dirs[0]):
        os.mkdir(output_dirs[0])

    return metadata_dir, output_dirs


def generate_graph(metadata_dir, output_dirs, input_args):

    if input_args.i:
        column_to_plot = input_args.i
    else:
        column_to_plot = 'Partition_ratio_mean_488'

    if input_args.s:
        error_column = input_args.s
    else:
        error_column = 'Partition_ratio_std488'  # I think there's a missing "_" in Krishna's pipeline before the 488

    if input_args.i and not input_args.s:
        sys.exit("ERROR: If you give mean column, you must also give standard deviation column")

    if input_args.s and not input_args.i:
        sys.exit("ERROR: If you give standard deviation column, you must also give mean column")

    if input_args.l:
        x_axis_label = input_args.l
    else:
        x_axis_label = '[protein] (µM)'

    if input_args.p:
        exponential_curve_param_guess = dict(input_args._get_kwargs())
        exponential_curve_param_guess = exponential_curve_param_guess['p']
    else:
        exponential_curve_param_guess = [1.0, 0.001, 1000]

    # default columns for specific values
    # pr_488_mean = 0
    # pr_488_std = 1
    # pr_561_mean = 6
    # pr_561_std = 7

    # cf_488_mean = 2
    # cf_488_std = 3
    # cf_561_mean = 8
    # cf_561_std = 9

    # ti_488_mean = 4
    # ti_488_std = 5
    # ti_561_mean = 10
    # ti_561_std = 11

    file_list = os.listdir(metadata_dir)

    # load metadata and parse data
    metadata_file_list = [s for s in file_list if 'metadata' in s and '$' not in s]  # metadata rows MUST MATCH summary rows
    metadata_file = metadata_file_list[0]
    metadata = pd.read_excel(os.path.join(metadata_dir, metadata_file), sheet=0)
   
    data_file_list = [s for s in file_list if 'summary' in s and '$' not in s]
    
    data = pd.read_excel(os.path.join(metadata_dir, data_file_list[0]), sheet_name='summary')
    sample_name = data_file_list[0].replace('summary_statistics_', '')

    plot_group = metadata['plot_group'].unique()

    for p in plot_group:

        experimental_group = metadata[(metadata.plot_group == p)].experimental_group.unique()
        for i in experimental_group:
            data_subset = data[(metadata.plot_group == p) & (metadata.experimental_group == i)].copy()
            data_subset.replace(np.nan, 0, inplace=True)

            # Add a 0,0 point assuming that partition is 0 with no protein (know this from GFP?)
            data_subset.loc[-1] = 0
            data_subset.index = data_subset.index + 1
            data_subset = data_subset.sort_index()

            metadata_subset = metadata[(metadata.plot_group == p) & (metadata.experimental_group == i)].copy()

            subset_color = metadata_subset['color'].iloc[0]
            line_color = re.sub('o', '-', subset_color)

            # partition graph
            x = metadata_subset['concentration'].tolist()
            x = [0] + x
            y = data_subset[column_to_plot].tolist()
            y_error = data_subset[error_column].tolist()

            if input_args.threshold_flag:
                for idx, value in enumerate(y):  # if Partition ratio less than 1, then set it and error to 0
                    if value < 1:
                        y[idx] = 0.0
                        y_error[idx] = 0.0

            plt.errorbar(x, y, yerr=y_error, xerr=None, fmt=subset_color, capsize=5, label=i)

            if input_args.fit_flag:
                # fit exponential curve
                popt, pcov = curve_fit(exponential_curve, x, y, p0=exponential_curve_param_guess)

                # smooth curve and exponential fit
                x_values = np.arange(0, np.max(x), 5)
                # x_values = range(0, np.max(x), 5)

                plt.plot(x_values, exponential_curve(x_values, *popt), line_color)

        if input_args.legend_flag:
            plt.legend()

        plt.gca().autoscale(enable=True, axis='y')
        y_bottom, y_top = plt.ylim()
        x_bottom, x_top = plt.xlim()
        plt.ylim(y_bottom - (y_bottom * 0.10), y_top)
        plt.xlim(x_bottom - (x_bottom * 0.10), x_top)

        plt.suptitle(sample_name)
        plt.xlabel(x_axis_label)
        plt.ylabel(column_to_plot)

        plt.savefig(os.path.join(output_dirs[0], sample_name + '.pdf'))
        plt.savefig(os.path.join(output_dirs[0], sample_name + '.png'), dpi=300)

        output_params = {'metadata_folder': metadata_dir,
                         'time_of_analysis': datetime.now(),
                         'plotted_column': column_to_plot,
                         'curve_parameters': exponential_curve_param_guess
                         }

        with open(os.path.join(output_dirs[0], 'output_analysis_parameters.txt'), 'w') as file:
            file.write(json.dumps(output_params, default=str))

        print("Finished all: ", datetime.now())


# parse input
parser = argparse.ArgumentParser()
parser.add_argument("metadata_dir")  # IMPORTANT: Must have 'metadata' in the Excel file name. Krishna's output MUST have 'summary' in Excel file name

parser.add_argument("--o", type=str)  # output directory name
parser.add_argument("--i", type=str)  # column to use from Krishna's output. Defaults to 'Partition_ratio_mean_488'
parser.add_argument("--s", type=str)  # column to use from Krishna's output for stdev. Defaults to 'Partition_ratio_std488'
parser.add_argument("--l", type=str)  # label for x-axis. Defaults to '[protein] (µM)'
parser.add_argument('--p', nargs='+', type=float)  # parameters for exponential fit. Write '--p a b c'. Defaults to [1.0, 0.001, 1000].
parser.add_argument('--no-threshold', dest='threshold_flag', action='store_false')  # whether to set all PRs < 1 to 0. Default True.
parser.add_argument('--no-fit', dest='fit_flag', action='store_false')   # whether to fit exponential curve. Default True.
parser.add_argument('--no-legend', dest='legend_flag', action='store_false')

parser.set_defaults(threshold_flag=True, fit_flag=True, legend_flag=True)

input_args = parser.parse_args()

metadata_path = input_args.metadata_dir

if not os.path.isdir(metadata_path):
    sys.exit('ERROR: Could not find metadata file')

metadata_dir, output_dirs = read_metadata(input_args)
generate_graph(metadata_dir, output_dirs, input_args)
