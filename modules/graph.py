import matplotlib.pylab as plt
import matplotlib
from io import BytesIO, StringIO
import matplotlib.pyplot as plt
import mpld3
from matplotlib.pyplot import figure
from typing import List
import numpy as np
import math


class Timeseries:
    # declare props as object NOT class props!
    def __init__(self):
        self.label = "None"
        self.color = "blue"
        self.x = []
        self.y = []


class Barseries:
    # declare props as object NOT class props!
    def __init__(self):
        self.label = "None"
        self.color = "blue"
        self.data = []
        self.average = None


class CMapMatrixElement:
    def __init__(self):
        self.i = 0
        self.j = 0
        self.ilabel = ""
        self.jlabel = ""
        self.val = 0
        self.auxval = 0

# class Timeseries(object):
#     label = "None"
#     color = "blue"
#     x = []
#     y = []


def plot_timeseries_multi_sub2(timeseries_arrays: List[List[Timeseries]], title, xlabel, ylabel):
    matplotlib.style.use('default')
    id = 0

    fig = plt.figure(id, figsize=(9, 16))
    n_sub = len(timeseries_arrays)
    for (i, timeseries_array) in enumerate(timeseries_arrays):
        plt.subplot(n_sub * 100 + 11 + i)
        for ts in timeseries_array:
            x = ts.x
            y = ts.y
            plt.plot(x, y, label=ts.label, color=ts.color)
            if i == 0:
                set_disp(title[i], "", ylabel[i])
            else:
                set_disp(title[i], xlabel, ylabel[i])
            plt.legend()

    fig = plt.gcf()

    plt.show()

    return fig, mpld3.fig_to_html(fig)


def plot_timeseries_multi(timeseries_array: List[Timeseries], title, xlabel, ylabel, separate):
    matplotlib.style.use('default')
    id = 0

    fig = None

    if not separate:
        fig = plt.figure(id)

    for ts in timeseries_array:
        if separate:
            fig = plt.figure(id)
        id += 1
        x = ts.x
        y = ts.y
        plt.plot(x, y, label=ts.label, color=ts.color)

        if separate:
            set_disp(title, xlabel, ylabel)
            plt.legend()
            plt.show(block=False)

    if not separate:
        set_disp(title, xlabel, ylabel)
        plt.legend()
        fig = plt.gcf()
        plt.show()

    if separate:
        plt.show()

    return fig, mpld3.fig_to_html(fig) if fig is not None else None


def stem_timeseries_multi(timeseries_array: List[Timeseries], title, xlabel, ylabel, separate):
    matplotlib.style.use('default')
    fig = plt.figure()

    bottom = None
    for ts in timeseries_array:
        for y in ts.y:
            if bottom is None or y < bottom:
                bottom = y

    for ts in timeseries_array:
        plt.stem(ts.x, ts.y, label=ts.label, bottom=bottom)

    set_disp(title, xlabel, ylabel)

    plt.legend()
    fig = plt.gcf()

    plt.show()

    return fig, mpld3.fig_to_html(fig)


def plot_timeseries_ax(timeseries: Timeseries, title, xlabel, ylabel, fig, ax):
    ax.plot(timeseries.x, timeseries.y)
    # set_disp(title, xlabel, ylabel)
    # plt.legend()
    # plt.show()
    return fig


def plot_timeseries(timeseries: Timeseries, title, xlabel, ylabel):
    fig = plt.figure()
    plt.plot(timeseries.x, timeseries.y)
    set_disp(title, xlabel, ylabel)

    # sio = BytesIO()
    # fig.savefig(sio, format='png')
    # sio.seek(0)
    # return sio

    # plt.show()
    # plt.close()

    plt.legend()

    fig = plt.gcf()

    plt.show()

    return fig, mpld3.fig_to_html(fig)


def save_figure(fig, file):
    fig.savefig(file, dpi=300)


def set_disp(title, xlabel, ylabel):
    if title:
        plt.gca().set_title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)


# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(x, y1)
# ax1.set_ylabel('y1')

# ax2 = ax1.twinx()
# ax2.plot(x, y2, 'r-')
# ax2.set_ylabel('y2', color='r')
# for tl in ax2.get_yticklabels():
#     tl.set_color('r')

# plt.savefig('images/two-scales-5.png')

def plot_barchart_multi(bss: List[Barseries], xlabel, ylabel, title, xlabels, top):
    return plot_barchart_multi_core(bss, xlabel, ylabel, title, xlabels, top, None, None, True, None, 0, None)[0]
    # 0.155
    # return plot_barchart_multi_core(bss, xlabel, ylabel, title, xlabels, top, None, None, True, -0.125, 0, None)[0]


def plot_barchart_multi_dual(bss1: List[Barseries], bss2: List[Barseries], xlabel, ylabel1, ylabel2, title, xlabels, top, show):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    fig, ax1 = plot_barchart_multi_core(
        bss1, xlabel, ylabel1, title, xlabels, top, fig, ax1, False, -0.155, 2, "upper left")

    for b in bss2:
        b.color = "red"

    ax2 = ax1.twinx()
    fig, _ = plot_barchart_multi_core(
        bss2, xlabel, ylabel2, title, xlabels, top, fig, ax2, True, 0.155, 2, "upper right")

    return fig


def plot_barchart_multi_core(bss: List[Barseries], xlabel, ylabel, title, xlabels, top, fig, ax, show, offset, bcount, legend_loc):

    # create plot
    if fig is None or ax is None:
        print("creating new figure")
        fig, ax = plt.subplots()

    n_groups = len(bss)

    if bcount != 0:
        bar_width = 1 / (bcount + 1)
    else:
        bar_width = 1/(n_groups+1)

    if offset is None:
        # offset = -1 / (n_groups * 2 * bar_width + 1)
        # offset = -bar_width/2
        offset = bar_width/2

    # if n_groups == 1:
    #     bar_width = 1

    opacity = 0.7

    low = None
    high = None

    for i in range(n_groups):

        index = np.arange(len(bss[i].data))

        # print(bss[i].data)

        low1 = min(bss[i].data)
        high1 = max(bss[i].data)

        if low is None:
            low = low1
            high = high1

        if low1 < low:
            low = low1
        if high1 > high:
            high = high1

        rb = plt.bar(
            index + offset + i * bar_width,
            bss[i].data,
            bar_width,
            alpha=opacity,
            color=bss[i].color,
            label=bss[i].label,
            zorder=3)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if n_groups == 1:
        plt.xticks(index, xlabels)
    else:
        plt.xticks(index + bar_width, xlabels)

    if not legend_loc:
        legend_loc = "upper left"

    plt.legend(loc=legend_loc)

    ax.grid(zorder=0)

    print(low)
    print(high)
    # plt.ylim([math.ceil(low-0.5*(high-low)), math.ceil(high+0.5*(high-low))])
    # plt.ylim([math.ceil(low-0.005*(high-low)), math.ceil(high+0.005*(high-low))])
    # plt.ylim([low, high])

    kscale = 0.01

    if not top:
        high = 100
    else:
        high += kscale*high

    low -= kscale*low

    plt.ylim([low, high])

    plt.tight_layout()

    if show:
        print("show")
        plt.show()

    return fig, ax


def plot_barchart(labels, values, xlabel, ylabel, title, color):

    fig = plt.figure()

    y_pos = np.arange(len(labels))

    plt.bar(y_pos, values, align='center', alpha=0.7, color=color)
    plt.xticks(y_pos, labels)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    low = min(values)
    high = max(values)

    print(low)
    print(high)
    # plt.ylim([math.ceil(low-0.5*(high-low)), math.ceil(high+0.5*(high-low))])
    # plt.ylim([math.ceil(low-0.005*(high-low)), math.ceil(high+0.005*(high-low))])
    plt.ylim([low - 0.005*low, high + 0.005*high])

    plt.title(title)

    fig = plt.gcf()

    plt.show()

    return fig


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", scale=None, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap

    # ax.figure.clim(0, 1)

    im = ax.imshow(data, **kwargs)

    # ax.figure.clim(0, 1)

    if scale is not None:
        for im in plt.gca().get_images():
            im.set_clim(scale[0], scale[1])

    # Create colorbar
    cbar = None
    if cbarlabel is not None:

        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)

        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    # ax.tick_params(top=True, bottom=False,
    #                labeltop=False, labelbottom=True)

    ax.tick_params(top=False, bottom=False, left=False,
                   labeltop=False, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #          rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)

    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_matrix_cmap(elements: List[CMapMatrixElement], xsize, ysize, title, xlabel, ylabel, xlabels, ylabels):

    min_val, max_val = elements[0].val, elements[0].val

    # intersection_matrix = np.random.randint(0, 10, size=(max_val, max_val))
    intersection_matrix = np.zeros((xsize, ysize))

    # print(intersection_matrix)
    for e in elements:
        intersection_matrix[e.i][e.j] = e.val
        if e.val < min_val:
            min_val = e.val
        if e.val > max_val:
            max_val = e.val

    fig, ax = plt.subplots()

    im, cbar = heatmap(intersection_matrix, xlabels, ylabels, ax=ax,
                       cmap="RdYlGn", cbarlabel="")

    # texts = annotate_heatmap(im, valfmt="{x:.1f} t")

    def millions(x, pos):
        'The two args are the value and tick position'
        return '%.1f' % (x)

    texts = annotate_heatmap(im, valfmt=millions)

    set_disp(title, xlabel, ylabel)

    fig.tight_layout()
    plt.show()

    return fig


def plot_matrix_cmap_plain(elements: List[CMapMatrixElement], xsize, ysize, title, xlabel, ylabel, xlabels, ylabels, scale=None):

    min_val, max_val = elements[0].val, elements[0].val

    # intersection_matrix = np.random.randint(0, 10, size=(max_val, max_val))
    intersection_matrix = np.zeros((xsize, ysize))

    # print(intersection_matrix)

    # intersection_matrix = np.zeros((xsize, ysize + 2))

    for e in elements:
        intersection_matrix[e.i][e.j] = e.val
        print(e.val)
        if e.val < min_val:
            min_val = e.val
        if e.val > max_val:
            max_val = e.val

    # intersection_matrix[0][ysize] = 0
    # intersection_matrix[0][ysize+1] = 1

    fig, ax = plt.subplots()

    im, cbar = heatmap(intersection_matrix, xlabels, ylabels, ax=ax,
                       cmap="Blues", cbarlabel=None, scale=scale)

    set_disp(title, xlabel, ylabel)

    fig.tight_layout()
    plt.show()

    return fig


def get_n_ax(n, figsize=None):
    if figsize:
        fig, ax = plt.subplots(nrows=n, ncols=1, figsize=figsize)
    else:   
        fig, ax = plt.subplots(nrows=n, ncols=1)
    return fig, ax

def plot_matrix_cmap_plain_ax(elements: List[CMapMatrixElement], xsize, ysize, title, xlabel, ylabel, xlabels, ylabels, scale, fig, ax):

    min_val, max_val = elements[0].val, elements[0].val

    intersection_matrix = np.zeros((xsize, ysize))

    for e in elements:
        intersection_matrix[e.i][e.j] = e.val
        print(e.val)
        if e.val < min_val:
            min_val = e.val
        if e.val > max_val:
            max_val = e.val


    im, cbar = heatmap(intersection_matrix, xlabels, ylabels, ax=ax,
                       cmap="Blues", cbarlabel=None, scale=scale)

    set_disp(title, xlabel, ylabel)


def show_fig(fig):
    fig.tight_layout()
    plt.show()

    return fig
