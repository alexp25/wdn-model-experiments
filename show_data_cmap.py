from modules import graph

from modules.graph import Timeseries, CMapMatrixElement
import numpy as np

from os import listdir
from os.path import isfile, join
import json
from typing import List

import yaml

# import copy

with open("config.yml", "r") as f:
    config = yaml.load(f)

elements: List[CMapMatrixElement] = []

rowsdict = {}
colsdict = {}

input_file = "./data/random1/raw_buffer.csv"

nvalves = 6
nrows = 0

rowskip = 500
nrowskip = 0

with open(input_file, "r") as f:
    content = f.read().split("\n")

    for index, line in enumerate(content):
        spec = line.split(",")
        if len(spec) > 1:
            # print(spec)
            try:
                elements_buffer: List[CMapMatrixElement] = []
                for valve in range(nvalves):
                    e = CMapMatrixElement()
                    e.i = valve
                    e.j = nrows
                    e.val = float(spec[valve + 13])
                    elements_buffer.append(e)

                nrowskip += 1

                if nrowskip >= rowskip:
                    nrowskip = 0
                    nrows += 1
                    for e in elements_buffer:
                        elements.append(e)
            except:
                pass

# print(elements)

# quit()


xlabels = [("v" + str(i + 1)) for i in range(nvalves)]
ylabels = [(str(int(i * rowskip / 100))) for i in range(nrows)]
# xlabels = []
# ylabels = []

# intersection_matrix = np.random.randint(0, 10, size=(max_val, max_val))
intersection_matrix = np.zeros((nvalves, nrows))

# print(intersection_matrix)

for e in elements:
    intersection_matrix[e.i][e.j] = e.val

print(intersection_matrix)

fig = graph.plot_matrix_cmap_plain(
    elements, nvalves, nrows, "Valve Sequence", "sample x" + str(rowskip), "valves",  xlabels, ylabels, None, None)
graph.save_figure(fig, "./figs/valve_sequence")
