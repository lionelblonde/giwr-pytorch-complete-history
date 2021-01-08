from collections import defaultdict
from copy import deepcopy
import glob
import argparse
import os
import os.path as osp
import hashlib
import time

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt  # noqa
import matplotlib.font_manager as fm  # noqa

from helpers.math_util import smooth_out_w_ema  # noqa


parser = argparse.ArgumentParser(description="Plotter")
parser.add_argument('--font', type=str, default='Colfax')
parser.add_argument('--dir', type=str, default=None, help='csv files location')
parser.add_argument('--xcolkey', type=str, default=None, help='name of the X column')
parser.add_argument('--ycolkey', type=str, default=None, help='name of the Y column')
parser.add_argument('--stdfrac', type=float, default=1., help='std envelope fraction')
args = parser.parse_args()


def plot(args):

    # Font (must be first)
    font_dir = "/Users/lionelblonde/Library/Fonts/"
    if args.font == 'Colfax':
        f1 = fm.FontProperties(fname=osp.join(font_dir, 'Colfax-Light.otf'), size=12)
        f2 = fm.FontProperties(fname=osp.join(font_dir, 'Colfax-Light.otf'), size=24)
        f3 = fm.FontProperties(fname=osp.join(font_dir, 'Colfax-Regular.otf'), size=14)
        f4 = fm.FontProperties(fname=osp.join(font_dir, 'Colfax-Medium.otf'), size=16)
    elif args.font == 'SourceCodePro':
        f1 = fm.FontProperties(fname=osp.join(font_dir, 'SourceCodePro-Light.otf'), size=12)
        f2 = fm.FontProperties(fname=osp.join(font_dir, 'SourceCodePro-Regular.otf'), size=24)
        f3 = fm.FontProperties(fname=osp.join(font_dir, 'SourceCodePro-Regular.otf'), size=14)
        f4 = fm.FontProperties(fname=osp.join(font_dir, 'SourceCodePro-Medium.otf'), size=16)
    else:
        raise ValueError("invalid font")
    # Create unique destination dir name
    hash_ = hashlib.sha1()
    hash_.update(str(time.time()).encode('utf-8'))
    dest_dir = "plots/batchplots_{}".format(hash_.hexdigest()[:20])
    os.makedirs(dest_dir, exist_ok=False)
    # Palette
    curves = [
        'xkcd:sky blue',
        'xkcd:pinkish brown',
        'xkcd:maize',
        'xkcd:wisteria',
        'xkcd:mango',
        'xkcd:bubblegum',
        'xkcd:turtle green',
        'xkcd:peacock blue',
        'xkcd:orangered',
        'xkcd:camo green',
        'xkcd:petrol',
        'xkcd:pea soup',
    ]
    palette = {
        'grid': (231, 234, 236),
        'face': (255, 255, 255),  # (245, 249, 249)
        'axes': (200, 200, 208),
        'font': (108, 108, 126),
        'symbol': (64, 68, 82),
        'expert': (0, 0, 0),
        'curves': curves,
    }
    for k, v in palette.items():
        if k != 'curves':
            palette[k] = tuple(float(e) / 255. for e in v)
    # Figure color
    plt.rcParams['axes.facecolor'] = palette['face']
    # DPI
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    # X and Y axes
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['axes.linewidth'] = 0.8
    # Lines
    plt.rcParams['lines.linewidth'] = 1.4
    plt.rcParams['lines.markersize'] = 1
    # Grid
    plt.rcParams['grid.linewidth'] = 0.6
    plt.rcParams['grid.linestyle'] = '-'

    # Dirs
    experiment_map = defaultdict(list)
    xcol_dump = defaultdict(list)
    ycol_dump = defaultdict(list)
    color_map = defaultdict(str)
    dirs = [d.split('/')[-1] for d in glob.glob("{}/*".format(args.dir))]
    print("pulling logs from sub-directories: {}".format(dirs))
    dirs.sort()
    dnames = deepcopy(dirs)
    dirs = ["{}/{}".format(args.dir, d) for d in dirs]
    print(dirs)
    # Colors
    colors = {d: palette['curves'][i] for i, d in enumerate(dirs)}

    for d in dirs:

        path = "{}/*/progress.csv".format(d)

        for fname in glob.glob(path):
            print("fname: {}".format(fname))
            # Extract the expriment name from the file's full path
            experiment_name = fname.split('/')[-2]
            # Remove what comes after the uuid
            key = experiment_name.split('.')[0] + "." + experiment_name.split('.')[2]
            env = experiment_name.split('.')[2]
            experiment_map[env].append(key)
            # Load data from the CSV file
            data = pd.read_csv(fname,
                               skipinitialspace=True,
                               usecols=[args.xcolkey, args.ycolkey])
            # Retrieve the desired columns from the data
            xcol = data[args.xcolkey].to_numpy()
            ycol = data[args.ycolkey].to_numpy()
            # Add the experiment's data to the dictionary
            xcol_dump[key].append(xcol)
            ycol_dump[key].append(ycol)
            # Add color
            color_map[key] = colors[d]

    for k, v in experiment_map.items():
        print(k, v)

    # Remove duplicate
    experiment_map = {k: list(set(v)) for k, v in experiment_map.items()}

    # Display summary of the extracted data
    assert len(xcol_dump.keys()) == len(ycol_dump.keys())  # then use X col arbitrarily
    print("summary -> {} different keys.".format(len(xcol_dump.keys())))
    for i, key in enumerate(xcol_dump.keys()):
        print(">>>> [key #{}] {} | #values: {}".format(i, key, len(xcol_dump[key])))

    print("\n>>>>>>>>>>>>>>>>>>>> Visualizing.")

    texts = deepcopy(dnames)
    texts.sort()
    texts = [text.split('__')[-1] for text in texts]
    num_cols = len(texts)  # noqa
    print("Legend's texts (ordered): {}".format(texts))

    patches = [plt.plot([],
                        [],
                        marker="o",
                        ms=18,
                        ls="",
                        color=palette['curves'][i],
                        label="{:s}".format(texts[i]))[0]
               for i in range(len(texts))]

    # Calculate the x axis upper bound
    xmaxes = defaultdict(int)
    for i, key in enumerate(xcol_dump.keys()):
        xmax = np.infty
        for i_, key_ in enumerate(xcol_dump[key]):
            xmax = len(key_) if xmax > len(key_) else xmax
        xmaxes[key] = xmax

    GRID_SIZE_X = 3
    GRID_SIZE_Y = 5
    fig, axs = plt.subplots(GRID_SIZE_X, GRID_SIZE_Y, figsize=(35, 20))
    for i in range(GRID_SIZE_X):
        for j in range(GRID_SIZE_Y):
            axs[i, j].axis('off')

    # Plot mean and standard deviation
    for j, env in enumerate(sorted(experiment_map.keys())):

        # Get the maximum Y value accross all the experiments
        ymax = -np.infty

        # Create subplot
        ax = axs[j // GRID_SIZE_Y, j % GRID_SIZE_Y]
        ax.axis('on')

        # Create grid
        ax.grid(color=palette['grid'])
        # Only leave the left and bottom axes
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Set the color of the axes
        ax.spines['left'].set_color(palette['axes'])
        ax.spines['bottom'].set_color(palette['axes'])

        # Go over the experiments and plot for each on the same subplot
        for i, key in enumerate(experiment_map[env]):

            xmax = deepcopy(xmaxes[key])

            print(">>>> {}, in color RGB={}".format(key, color_map[key]))

            if len(ycol_dump[key]) > 1:
                # Calculate statistics to plot
                mean = np.mean(np.column_stack([col_[0:xmax] for col_ in ycol_dump[key]]), axis=-1)
                std = np.std(np.column_stack([col_[0:xmax] for col_ in ycol_dump[key]]), axis=-1)

                # Plot the computed statistics
                WEIGHT = 0.85
                smooth_mean = np.array(smooth_out_w_ema(mean, weight=WEIGHT))
                smooth_std = np.array(smooth_out_w_ema(std, weight=WEIGHT))
                ax.plot(xcol_dump[key][0][0:xmax], smooth_mean, color=color_map[key], alpha=0.8)
                ax.fill_between(xcol_dump[key][0][0:xmax],
                                smooth_mean - (args.stdfrac * smooth_std),
                                smooth_mean + (args.stdfrac * smooth_std),
                                facecolor=color_map[key],
                                alpha=0.1)

                # Get the maximum Y value accross all the experiments
                _ymax = np.amax(mean + (args.stdfrac * std))
                ymax = max(_ymax, ymax)
            else:
                ax.plot(xcol_dump[key][0], ycol_dump[key][0])

        # Create the axes labels
        ax.tick_params(width=0.2, length=1, pad=1, colors=palette['axes'], labelcolor=palette['font'])
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-5, 5), useOffset=(False), useMathText=True)
        ax.xaxis.offsetText.set_fontproperties(f1)
        ax.xaxis.offsetText.set_position((0.95, 0))
        for tick in ax.get_xticklabels():
            tick.set_fontproperties(f1)
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(f1)
        ax.set_xlabel("Timesteps", color=palette['font'], fontproperties=f3)  # , labelpad=6
        ax.set_ylabel("Episodic Return", color=palette['font'], fontproperties=f3)  # , labelpad=12
        # Create title
        ax.set_title("{}".format(env), color=palette['font'], fontproperties=f4, pad=-10)

    # Create legend
    legend = fig.legend(
        handles=patches,
        # ncol=num_cols,
        loc='center right',
        # borderaxespad=0,
        facecolor='w',
        # bbox_to_anchor=(0.0, -0.01)
    )
    legend.get_frame().set_linewidth(0.0)
    for text in legend.get_texts():
        text.set_color(palette['font'])
        text.set_fontproperties(f2)

    fig.set_tight_layout(True)

    # Save figure to disk
    plt.savefig("{}/plots_{}.pdf".format(dest_dir, args.ycolkey),
                format='pdf', bbox_inches='tight')
    print("mean plot done for env {}.".format(env))

    print(">>>>>>>>>>>>>>>>>>>> Bye.")


if __name__ == "__main__":
    # Plot
    plot(args)
