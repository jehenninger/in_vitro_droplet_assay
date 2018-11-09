import sys
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PyQt5 import QtWidgets
from scipy.optimize import curve_fit
from scipy import interpolate
import re


def exponential_curve(x, a, b, c):
    return a / (1 + np.exp(-b*(x-c)))


# graph properties
# @TODO Figure out this stupid figure system to get axes and everything the right size
# @TODO change error to plus or minus on data points
# @TODO Either figure out interpolation or don't use it
# @TODO Maybe exclude some far out data points to make curve fitting better?
# @TODO Add in ability to exclude data
# @TODO output the exponential curve parameters
figure_size = None
figure_dpi = None
# figure_size = [3, 2]
# figure_dpi = 300

# default columns for specific values
pr_488_mean = 0
pr_488_std = 1
pr_561_mean = 6
pr_561_std = 7

cf_488_mean = 2
cf_488_std = 3
cf_561_mean = 8
cf_561_std = 9

ti_488_mean = 4
ti_488_std = 5
ti_561_mean = 10
ti_561_std = 11

# get directory
#app = QtWidgets.QApplication(sys.argv)
#excel_dir = QtWidgets.QFileDialog.getExistingDirectory(caption='Select ')
#QtWidgets.QApplication.processEvents()

excel_dir = '/Users/jon/data_analysis/young/to_do/SignalingHomotypicDropletCollection/graphs/Bcat/'

temp_xlabel = '[Bcat] (µM)'

exponential_curve_param_guess = [1.0, 0.001, 1000]  # in the realm of 1.0, 0.001, 1000

# load excel droplet summary file
if not excel_dir:
    exit()

file_list = os.listdir(excel_dir)
graph_output_dir = os.path.join(excel_dir, 'graph_output')

if not os.path.isdir(graph_output_dir):
    os.mkdir(os.path.join(excel_dir, 'graph_output'))

# load metadata and parse data
metadata_file_list = [s for s in file_list if 'metadata' in s]  # metadata rows MUST MATCH summary rows
metadata_file = metadata_file_list[0]
metadata = pd.read_excel(os.path.join(excel_dir, metadata_file))


data_file_list = [s for s in file_list if 'summary' in s]

# def make_plot(metadata, data, type)

for data_file in data_file_list:
    data = pd.read_excel(os.path.join(excel_dir, data_file), sheet_name=3)  # fourth sheet because of Krishna's output
    sample_name = data_file.replace('summary_statistics_', '')

    concentration = metadata['concentration'].unique()
    experimental_group = metadata['experimental_group'].unique()
    plot_group = metadata['plot_group'].unique()

    # make graphs @TODO this is currently set for only 2 channels (488 and 561). Need to add ability to distinguish

    for p in plot_group:
        # partition ratio graph initiation
        # pr_figure, pr_ax = plt.subplots()
        # if figure_size:
        #     pr_figure.set_size_inches(figure_size[0], figure_size[1])
        # if figure_dpi:
        #     pr_figure.set_dpi(figure_dpi)


        # # partition ratio graph initiation
        # pr_figure, (pr_ax1, pr_ax2) = plt.subplots(1, 2)
        # if figure_size:
        #     pr_figure.set_size_inches(figure_size[0], figure_size[1])
        # if figure_dpi:
        #     pr_figure.set_dpi(figure_dpi)
        # pr_figure.suptitle(sample_name)
        # pr_ax1.set(xlabel=temp_xlabel, ylabel='Partition Ratio 488')
        # pr_ax2.set(xlabel=temp_xlabel, ylabel='Partition Ratio 561')
        #
        # # condensed fraction graph initiation
        # cf_figure, (cf_ax1, cf_ax2) = plt.subplots(1, 2)
        # if figure_size:
        #     cf_figure.set_size_inches(figure_size[0], figure_size[1])
        # if figure_dpi:
        #     cf_figure.set_dpi(figure_dpi)
        # cf_figure.suptitle(sample_name)
        # cf_ax1.set(xlabel=temp_xlabel, ylabel='Condensed Fraction 488')
        # cf_ax2.set(xlabel=temp_xlabel, ylabel='Condensed Fraction 561')
        #
        # # total intensity graph initiation
        # ti_figure, (ti_ax1, ti_ax2) = plt.subplots(1, 2)
        # if figure_size:
        #     ti_figure.set_size_inches(figure_size[0], figure_size[1])
        # if figure_dpi:
        #     ti_figure.set_dpi(figure_dpi)
        # ti_figure.suptitle(sample_name)
        # ti_ax1.set(xlabel=temp_xlabel, ylabel='Total Intensity 488')
        # ti_ax2.set(xlabel=temp_xlabel, ylabel='Total Intensity 561')

        experimental_group = metadata[(metadata.plot_group == p)].experimental_group.unique()
        for i in experimental_group:
            data_subset = data[(metadata.plot_group == p) & (metadata.experimental_group == i)].copy()
            data_subset.replace(np.nan, 0, inplace=True)

            # Add a 0,0 point assuming that partition is 0 with no protein (know this from GFP?)
            data_subset.loc[-1] = 0
            # np.zeros(shape=(1, data_subset.shape[1]))
            data_subset.index = data_subset.index + 1
            data_subset = data_subset.sort_index()

            metadata_subset = metadata[(metadata.plot_group == p) & (metadata.experimental_group == i)].copy()

            subset_color = metadata_subset['color'].iloc[0]
            line_color = re.sub('o','-',subset_color)

            # partition graph
            x = metadata_subset['concentration'].tolist()
            x = [0] + x
            y = data_subset['Partition_ratio_mean_488'].tolist()
            y_error = data_subset.iloc[:, pr_488_std].tolist()

            for idx, value in enumerate(y):  # if Partition ratio less than 1, then set it and error to 0
                if value < 1:
                    y[idx] = 0.0
                    y_error[idx] = 0.0

            plt.errorbar(x, y, yerr=y_error, xerr=None, fmt=subset_color, capsize=5, label=i)

            # fit exponential curve
            popt, pcov = curve_fit(exponential_curve, x, y, p0=exponential_curve_param_guess)

            # smooth curve and exponential fit
            x_values = range(0, np.max(x), 5)

            plt.plot(x_values, exponential_curve(x_values, *popt), line_color)

            # bspl = interpolate.splrep(x, y, s=10)
            # bspl_y = interpolate.splev(x_values, bspl)

            # pr_ax.plot(x_values, bspl_y, 'b-')

            # pr_ax.fill_between(metadata_subset['concentration'],
            #                    data_subset.iloc[:, pr_488_mean] - data_subset.iloc[:, pr_488_std],
            #                    data_subset.iloc[:, pr_488_mean] + data_subset.iloc[:, pr_488_std],
            #                    alpha=0.1
            #                    )

            # pr_ax1.set_aspect(aspect='equal', adjustable='box')
            # pr_ax2.set_aspect(aspect='equal', adjustable='box')

        plt.legend()

        plt.gca().autoscale(enable=True, axis='y')
        y_bottom, y_top = plt.ylim()
        x_bottom, x_top = plt.xlim()
        plt.ylim(-0.5, y_top)
        plt.xlim(-500, x_top)

        plt.suptitle(sample_name)
        plt.xlabel(temp_xlabel)
        plt.ylabel('Partition Ratio 488')

        #plt.figure(pr_figure)
        plt.savefig(os.path.join(graph_output_dir, sample_name + '_PR.pdf'))
        plt.savefig(os.path.join(graph_output_dir, sample_name + '_PR.svg'))

            # # partition graph
            # pr_ax1.plot(metadata_subset['concentration'], data_subset.iloc[:, pr_488_mean], subset_color)
            # pr_ax1.fill_between(metadata_subset['concentration'],
            #                     data_subset.iloc[:, pr_488_mean] - data_subset.iloc[:, pr_488_std],
            #                     data_subset.iloc[:, pr_488_mean] + data_subset.iloc[:, pr_488_std],
            #                     alpha=0.1
            #                     )
            #
            # pr_ax2.plot(metadata_subset['concentration'], data_subset.iloc[:, pr_561_mean], subset_color)
            # pr_ax2.fill_between(metadata_subset['concentration'],
            #                     data_subset.iloc[:, pr_561_mean] - data_subset.iloc[:, pr_561_std],
            #                     data_subset.iloc[:, pr_561_mean] + data_subset.iloc[:, pr_561_std],
            #                     alpha=0.1
            #                     )
            # # pr_ax1.set_aspect(aspect='equal', adjustable='box')
            # # pr_ax2.set_aspect(aspect='equal', adjustable='box')
            # pr_ax1.legend(labels=experimental_group)
            # pr_ax2.legend(labels=experimental_group)
            #
            # plt.figure(pr_figure)
            # plt.savefig(os.path.join(graph_output_dir, sample_name + '_PR.pdf'))
            # plt.savefig(os.path.join(graph_output_dir, sample_name + '_PR.svg'))
            #
            # # condensed fraction graph
            # cf_ax1.plot(metadata_subset['concentration'], data_subset.iloc[:, cf_488_mean], subset_color)
            # cf_ax1.fill_between(metadata_subset['concentration'],
            #                     data_subset.iloc[:, cf_488_mean] - data_subset.iloc[:, cf_488_std],
            #                     data_subset.iloc[:, cf_488_mean] + data_subset.iloc[:, cf_488_std],
            #                     alpha=0.1
            #                     )
            #
            # cf_ax2.plot(metadata_subset['concentration'], data_subset.iloc[:, cf_561_mean], subset_color)
            # cf_ax2.fill_between(metadata_subset['concentration'],
            #                     data_subset.iloc[:, cf_561_mean] - data_subset.iloc[:, cf_561_std],
            #                     data_subset.iloc[:, cf_561_mean] + data_subset.iloc[:, cf_561_std],
            #                     alpha=0.1
            #                     )
            # # cf_ax1.set_aspect(aspect='equal', adjustable='box')
            # # cf_ax2.set_aspect(aspect='equal', adjustable='box')
            # cf_ax1.legend(labels=experimental_group)
            # cf_ax2.legend(labels=experimental_group)
            #
            # plt.figure(cf_figure)
            # plt.savefig(os.path.join(graph_output_dir, sample_name + '_CF.pdf'))
            # plt.savefig(os.path.join(graph_output_dir, sample_name + '_CF.svg'))
            #
            # # total raw intensity
            # ti_ax1.plot(metadata_subset['concentration'], data_subset.iloc[:, ti_488_mean], subset_color)
            # ti_ax1.fill_between(metadata_subset['concentration'],
            #                     data_subset.iloc[:, ti_488_mean] - data_subset.iloc[:, ti_488_std],
            #                     data_subset.iloc[:, ti_488_mean] + data_subset.iloc[:, ti_488_std],
            #                     alpha=0.1
            #                     )
            #
            # ti_ax2.plot(metadata_subset['concentration'], data_subset.iloc[:, ti_561_mean], subset_color)
            # ti_ax2.fill_between(metadata_subset['concentration'],
            #                     data_subset.iloc[:, ti_561_mean] - data_subset.iloc[:, ti_561_std],
            #                     data_subset.iloc[:, ti_561_mean] + data_subset.iloc[:, ti_561_std],
            #                     alpha=0.1
            #                     )
            # # ti_ax1.set_aspect(aspect='equal', adjustable='box')
            # # ti_ax2.set_aspect(aspect='equal', adjustable='box')
            # ti_ax1.legend(labels=experimental_group)
            # ti_ax2.legend(labels=experimental_group)
            #
            # plt.figure(ti_figure)
            # plt.savefig(os.path.join(graph_output_dir, sample_name + '_TI.pdf'))
            # plt.savefig(os.path.join(graph_output_dir, sample_name + '_TI.svg'))

# plt.show()

    # output graphs



