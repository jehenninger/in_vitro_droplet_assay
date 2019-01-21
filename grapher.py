from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import math

# Graphs to consider making:
#     Boxplot and dots for all droplets
#     Mean +/- stdev for image averages
#     Droplet size/area histogram
#     Intensity in one channel vs. the other (to look at heterotypic vs homotypic droplets)


def make_droplet_boxplot(data, output_dirs, input_args):
    pr_cols = [col for col in data[0].columns if 'partition' in col]  # need to identify all columns with partition ratios
    for channel in pr_cols:
        plot_data = [d[channel] for d in data]
        groups = [np.unique(g['sample']) for g in data]
        groups = [item for items in groups for item in items]

        fig, ax = plt.subplots()
        ax.boxplot(plot_data, labels=groups, showfliers=False)

        for i in range(len(groups)):
            y = plot_data[i]
            x = np.random.normal(1 + i, 0.04, size=len(y))
            ax.plot(x, y, 'b.', markersize=10, markeredgewidth=0, alpha=0.3)

        plt.ylim(bottom=0)
        plt.ylabel(channel)

        plt.savefig(os.path.join(output_dirs['output_individual'], channel + '_droplet_boxplot.png'), dpi=300, format='png')
        plt.savefig(os.path.join(output_dirs['output_individual'], channel + '_droplet_boxplot.eps'), format='eps')
        plt.close()


def make_droplet_size_histogram(data, output_dirs, input_args):
    size_data = data['area'].tolist()
    sample_name = np.unique(data['sample'])[0]

    fig, ax = plt.subplots()
    ax.hist(size_data, bins=50, density=True)

    plt.title(sample_name)

    plt.ylabel('Fraction of total')
    plt.xlabel('Droplet area')
    plt.axis('tight')

    plt.savefig(os.path.join(output_dirs['output_individual'], sample_name + '_droplet_size_histogram.png'), dpi=300, format='png')
    plt.savefig(os.path.join(output_dirs['output_individual'], sample_name + '_droplet_size_histogram.eps'), format='eps')
    plt.close()


def make_droplet_intensity_scatter(data, output_dirs, input_args):
    # for now, we only support this feature for 2 channels because it will be hard-coded
    mean_intensity_cols = [col for col in data.columns if 'mean' in col]

    channel_a = data[mean_intensity_cols[0]]
    channel_b = data[mean_intensity_cols[1]]

    sample_name = np.unique(data['sample'])[0]

    fig, ax = plt.subplots()
    ax.plot(channel_a, channel_b, 'b.', markersize=15, markeredgewidth=0, alpha=0.3)

    plt.ylabel(mean_intensity_cols[1])
    plt.xlabel(mean_intensity_cols[0])
    plt.title(sample_name)
    plt.axis('tight')

    plt.savefig(os.path.join(output_dirs['output_individual'], sample_name + '_droplet_intensity_scatter.png'), dpi=300,
                format='png')
    plt.savefig(os.path.join(output_dirs['output_individual'], sample_name + '_droplet_intensity_scatter.eps'),
                format='eps')
    plt.close()

def make_average_sample_graph(data, output_dirs, input_args):

    # partition ratio graph
    pr_cols = [col for col in data.columns if 'partition_ratio_mean' in col]

    for p in pr_cols:
        fig, ax = plt.subplots()
        for i, s in enumerate(data['sample']):
            ax.errorbar(x=i+1, y=data[p][data['sample'] == s], yerr=data[p.replace('_mean_', '_std_')][data['sample'] == s],
                        fmt='bo', capsize=20, markersize=10)

        plt.ylabel(p)
        plt.xlim(0, len(data.index) + 1)
        # plt.ylim(bottom=1, top=math.floor(np.max(data[p]) + 1))
        plt.ylim(bottom=1)
        x_labels = data['sample'].tolist()
        plt.xticks(list(range(1, len(data.index)+1)), x_labels)

        plt.savefig(os.path.join(output_dirs['output_summary'], p + '_average.png'),
                    dpi=300,
                    format='png')
        plt.savefig(os.path.join(output_dirs['output_summary'], p + '_average.eps'),
                    format='eps')
        plt.close()

    # condensed fraction graph
    cf_cols = [col for col in data.columns if 'condensed_fraction_mean' in col]

    for c in cf_cols:
        fig, ax = plt.subplots()
        for i, s in enumerate(data['sample']):
            ax.errorbar(x=i + 1, y=data[c][data['sample'] == s],
                        yerr=data[c.replace('_mean_', '_std_')][data['sample'] == s],
                        fmt='bo', capsize=20, markersize=10)

        plt.ylabel(c)
        plt.xlim(0, len(data.index) + 1)
        plt.ylim(bottom=0)
        x_labels = data['sample'].tolist()
        plt.xticks(list(range(1, len(data.index) + 1)), x_labels)

        plt.savefig(os.path.join(output_dirs['output_summary'], c + '_average.png'),
                    dpi=300,
                    format='png')
        plt.savefig(os.path.join(output_dirs['output_summary'], c + '_average.eps'),
                    format='eps')
        plt.close()




