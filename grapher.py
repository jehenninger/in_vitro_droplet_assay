from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

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

        plt.ylim(bottom=1)
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


def make_droplet_intensity_scatter(data, output_dirs, input_args):
    mean_intensity_cols = [col for col in data.columns if 'mean' in col]
