import pickle
from modules import graph

from modules.graph import Timeseries, CMapMatrixElement
import numpy as np

from os import listdir
from os.path import isfile, join
import json
from typing import List

import yaml


def get_intersection_matrix(elements: List[CMapMatrixElement], rows, cols):
    intersection_matrix = np.zeros((rows, cols))
    print(rows, cols)
    # print("check elems")
    for e in elements:
        # print(e.i, e.j)
        intersection_matrix[e.j][e.i] = e.val

    return intersection_matrix

        
def plot_intersection_matrix(elements, index, nrows, ncols, save, scale):
    fig = graph.plot_matrix_cmap_plain(
        elements, nrows, ncols, "Sequence evaluation", "sample bins", "",  xlabels, ylabels, scale, (16,7))
    if save:
        graph.save_figure(fig, "./figs/valve_sequence_random_prediction")


nvalves = 6

with open("elem.dat", "rb") as f:
    elements_combined = pickle.load(f)

with open("elem1.dat", "rb") as f:
    elements = pickle.load(f)

with open("elem2.dat", "rb") as f:
    elements_prediction = pickle.load(f)


n_bins = len(elements_combined)
rowskip = 1

vals = [e.val for e in elements_combined]
print(vals)
print("avg: ", np.mean(vals))

# quit()

imatrix = get_intersection_matrix(elements_combined, n_bins, 1)
print(imatrix)

elements_all = []

for e in elements:
    elements_all.append(e)

for e in elements_combined:
    e.i = 7
    elements_all.append(e)


xlabels = [("v" + str(i + 1)) for i in range(nvalves)]
xlabels.append("  ")
xlabels.append("p ")
ylabels = [("" + str(i+1)) for i in range(n_bins)]
# ylabels = [(" ") for i in range(n_bins)]

plot_intersection_matrix(elements_all, 3, 8, n_bins, True, (0, 1))

