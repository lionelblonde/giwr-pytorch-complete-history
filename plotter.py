from collections import defaultdict
from copy import deepcopy
import glob
import argparse
import os
import os.path as osp
import hashlib
import time

import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt  # noqa
import matplotlib.font_manager as fm  # noqa

from helpers.math_util import smooth_out_w_ema  # noqa


def plot(args, dest_dir, ycolkey, barplot):

    # Font (must be first)
    font_dir = "/Users/lionelblonde/Library/Fonts/"
    if args.font == 'Colfax':
        f1 = fm.FontProperties(fname=osp.join(font_dir, 'Colfax-Light.otf'), size=12)
        f2 = fm.FontProperties(fname=osp.join(font_dir, 'Colfax-Light.otf'), size=24)
        f3 = fm.FontProperties(fname=osp.join(font_dir, 'Colfax-Regular.otf'), size=14)
        f4 = fm.FontProperties(fname=osp.join(font_dir, 'Colfax-Medium.otf'), size=16)
    elif args.font == 'SourceCodePro':
        f1 = fm.FontProperties(fname=osp.join(font_dir, 'SourceCodePro-Light.otf'), size=20)
        f2 = fm.FontProperties(fname=osp.join(font_dir, 'SourceCodePro-Regular.otf'), size=32)
        f3 = fm.FontProperties(fname=osp.join(font_dir, 'SourceCodePro-Regular.otf'), size=22)
        f4 = fm.FontProperties(fname=osp.join(font_dir, 'SourceCodePro-Medium.otf'), size=24)
    else:
        raise ValueError("invalid font")

    # Palette
    palette = {
        'grid': (231, 234, 236),
        'face': (255, 255, 255),
        'axes': (200, 200, 208),
        'font': (108, 108, 126),
        'symbol': (64, 68, 82),
        'expert': (0, 0, 0),
        'curves': sb.color_palette(),
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
    text_map = defaultdict(str)
    dirs = [d.split('/')[-1] for d in glob.glob(f"{args.dir}/*")]
    print(f"pulling logs from sub-directories: {dirs}")
    dirs.sort()
    dnames = deepcopy(dirs)
    dirs = ["{}/{}".format(args.dir, d) for d in dirs]
    print(dirs)
    # Colors
    colors = {d: palette['curves'][i] for i, d in enumerate(dirs)}

    for d in dirs:

        path = f"{d}/*/progress.csv"

        for fname in glob.glob(path):
            # Extract the expriment name from the file's full path
            experiment_name = fname.split('/')[-2]
            # Remove what comes after the uuid
            key = experiment_name.split('.')[0] + "." + experiment_name.split('.')[2]
            env = experiment_name.split('.')[2]
            experiment_map[env].append(key)
            # Load data from the CSV file
            data = pd.read_csv(fname,
                               skipinitialspace=True,
                               usecols=[args.xcolkey, ycolkey])
            data.fillna(0.0, inplace=True)
            # Retrieve the desired columns from the data
            xcol = data[args.xcolkey].to_numpy()
            ycol = data[ycolkey].to_numpy()
            # Add the experiment's data to the dictionary
            xcol_dump[key].append(xcol)
            ycol_dump[key].append(ycol)
            # Add color
            color_map[key] = colors[d]
            # Add text
            text_map[key] = fname.split('/')[-3]

    for k, v in experiment_map.items():
        print(k, v)

    # Remove duplicate
    experiment_map = {k: list(set(v)) for k, v in experiment_map.items()}

    # Display summary of the extracted data
    assert len(xcol_dump.keys()) == len(ycol_dump.keys())  # then use X col arbitrarily
    print(f"summary -> {len(xcol_dump.keys())} different keys.")
    for i, key in enumerate(xcol_dump.keys()):
        print(f">>>> [key #{i}] {key} | #values: {len(xcol_dump[key])}")

    print("\n>>>>>>>>>>>>>>>>>>>> Visualizing.")

    texts = deepcopy(dnames)
    texts.sort()
    texts = [text.split('__')[-1] for text in texts]
    num_cols = len(texts)  # noqa
    print(f"Legend's texts (ordered): {texts}")

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
    # GRID_SIZE_X = 1
    # GRID_SIZE_Y = 3
    # fig, axs = plt.subplots(GRID_SIZE_X, GRID_SIZE_Y, figsize=(22, 7))

    if GRID_SIZE_X == 1:
        axs = np.expand_dims(axs, axis=0)
    for i in range(GRID_SIZE_X):
        for j in range(GRID_SIZE_Y):
            axs[i, j].axis('off')

    # Plot mean and standard deviation
    for j, env in enumerate(sorted(experiment_map.keys())):

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

        if barplot:
            bars = {}
            bars_errors = {}
            bars_colors = {}

        # Go over the experiments and plot for each on the same subplot
        for i, key in enumerate(experiment_map[env]):

            xmax = deepcopy(xmaxes[key])

            print(f">>>> {key}, in color RGB={color_map[key]}")

            if len(ycol_dump[key]) > 1:
                # Calculate statistics to plot
                mean = np.mean(np.column_stack([col_[0:xmax] for col_ in ycol_dump[key]]), axis=-1)
                std = np.std(np.column_stack([col_[0:xmax] for col_ in ycol_dump[key]]), axis=-1)

                # Plot the computed statistics
                WEIGHT = 0.85
                smooth_mean = np.array(smooth_out_w_ema(mean, weight=WEIGHT))
                smooth_std = np.array(smooth_out_w_ema(std, weight=WEIGHT))

                if barplot:
                    bars[text_map[key]] = smooth_mean[-1]
                    bars_errors[text_map[key]] = smooth_std[-1]
                    bars_colors[text_map[key]] = color_map[key]
                else:
                    ax.plot(xcol_dump[key][0][0:xmax], smooth_mean, color=color_map[key], alpha=1.0)
                    ax.fill_between(xcol_dump[key][0][0:xmax],
                                    smooth_mean - (args.stdfrac * smooth_std),
                                    smooth_mean + (args.stdfrac * smooth_std),
                                    facecolor=color_map[key],
                                    alpha=0.2)
            else:
                if not barplot:
                    ax.plot(xcol_dump[key][0], ycol_dump[key][0])
                else:
                    pass

        if barplot:
            PLOT_NAME = False  # XXX
            ax.bar(x=[(v.split('__')[-1] if PLOT_NAME else v.split('__')[0])
                      for v in sorted(set(list(text_map.values())))],
                   height=[bars[k] for k in sorted(list(bars.keys()))],
                   yerr=[bars_errors[k] for k in sorted(list(bars_errors.keys()))],
                   color=[bars_colors[k] for k in sorted(list(bars_colors.keys()))],
                   width=0.8,
                   alpha=0.9,
                   capsize=5)

        # Create the axes labels
        ax.tick_params(width=0.2, length=1, pad=1, colors=palette['axes'], labelcolor=palette['font'])
        if not barplot:
            ax.ticklabel_format(axis='x', style='sci', scilimits=(-5, 5), useOffset=(False), useMathText=True)
        ax.xaxis.offsetText.set_fontproperties(f1)
        ax.xaxis.offsetText.set_position((0.95, 0))
        for tick in ax.get_xticklabels():
            tick.set_fontproperties(f1)
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(f1)
        if not barplot:
            ax.set_xlabel("Timesteps", color=palette['font'], fontproperties=f3)  # , labelpad=6
        ax.set_ylabel("Episodic Return", color=palette['font'], fontproperties=f3)  # , labelpad=12
        # Create title
        ax.set_title(f"{env}", color=palette['font'], fontproperties=f4, pad=-10)

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
    plt.savefig(f"{dest_dir}/plots_{ycolkey}_{'barplot' if barplot else 'plot'}.pdf",
                format='pdf',
                bbox_inches='tight')
    print(f"mean plot done for env {env}.")

    print(">>>>>>>>>>>>>>>>>>>> bye.")


if __name__ == "__main__":
    # Parse
    parser = argparse.ArgumentParser(description="Plotter")
    parser.add_argument('--font', type=str, default='Colfax')
    parser.add_argument('--dir', type=str, default=None, help='csv files location')
    parser.add_argument('--xcolkey', type=str, default=None, help='name of the X column')
    parser.add_argument('--ycolkeys', nargs='+', type=str, default=None, help='name of the Y column')
    parser.add_argument('--stdfrac', type=float, default=1., help='std envelope fraction')
    args = parser.parse_args()

    # Create unique destination dir name
    hash_ = hashlib.sha1()
    hash_.update(str(time.time()).encode('utf-8'))
    dest_dir = f"plots/batchplots_{hash_.hexdigest()[:20]}"
    os.makedirs(dest_dir, exist_ok=False)

    # Plot
    for ycolkey in args.ycolkeys:
        plot(args, dest_dir=dest_dir, ycolkey=ycolkey, barplot=False)
        plot(args, dest_dir=dest_dir, ycolkey=ycolkey, barplot=True)
