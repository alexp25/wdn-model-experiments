import numpy as np
import pandas as pd

# import our modules
from modules import classifiers
from modules import deep_learning
from modules import loader

import tensorflow as tf

from keras import backend as K

import time

import os
from shutil import copyfile, copy2

import yaml


with open("config.yml", "r") as f:
    config = yaml.load(f)

root_data_folder = config["root_data_folder"]
root_crt_model_folder = config["root_crt_model_folder"]
# read the data from the csv file
# input_file = "./PastHires.csv"
input_file = config["input_file"]
filenames = config["filenames"]

if config["run_clean"]:
    loader.clean(root_crt_model_folder)

n_reps = 5
save_best_model = True

if n_reps > 1:
    save_best_model = True
else:
    save_best_model = False


model_file = root_crt_model_folder + "/" + "combined"

top_acc = 0
top_model_filename = None


for (i, r) in enumerate(range(n_reps)):
    model = None
    graph = tf.Graph()
    model_filename = model_file + \
        "." + str(i+1) + "." + "full" + ".h5"
    with tf.Session(graph=graph):
        for (j, filename) in enumerate(filenames):
            # load dataset
            model_filename_incremental = model_file + \
                "." + str(i+1) + "." + str(j+1) + ".h5"

            data_file = root_data_folder + "/" + filename + ".csv"
            
            x, y, _, _ = loader.load_dataset(data_file)
            y = loader.binarize(y)

            train_percent = config["train_percent"]

            x_train, y_train = classifiers.split_dataset_train(
                x, y, train_percent)

            x_eval, y_eval = classifiers.split_dataset_test(
                x, y, train_percent)

            if model is None:
                model = deep_learning.create_model(x_train, y_train)

            deep_learning.train_model(model, x_train, y_train)
            deep_learning.dl_save_model(model, model_filename_incremental)

            deep_learning.eval_write_info(
                model, x_eval, y_eval, model_filename_incremental, 0, 0)

        deep_learning.dl_save_model(model, model_filename)
        acc = deep_learning.eval_write_info(model, x, y, model_filename, 0, 0)
        if acc > top_acc:
            top_acc = acc
            top_model_filename = model_filename.split(".h5")[0]

if save_best_model:
    copy2(top_model_filename + ".h5", top_model_filename + "_top.h5")
    copy2(top_model_filename + ".h5.txt", top_model_filename + "_top.h5.txt")
