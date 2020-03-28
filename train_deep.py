import numpy as np
import pandas as pd

# import our modules
from modules import classifiers
from modules import deep_learning, loader
import tensorflow as tf
from keras import backend as K
import time
import os
from shutil import copyfile, copy2
import yaml
from modules.preprocessing import Preprocessing
from modules import generator

with open("config.yml", "r") as f:
    config = yaml.load(f)

root_data_folder = config["root_data_folder"]
root_crt_model_folder = config["root_crt_model_folder"]

if config["run_clean"]:
    loader.clean(root_crt_model_folder)

# read the data from the csv file
# input_file = "./PastHires.csv"
input_file = config["input_file"]
filenames = config["filenames"]
bookmarks = config["bookmarks"]

# root_data_folder += "/random1"
# filenames = ["exp_179"]

# filenames = ["exp_39"]

# root_data_folder += "/control/2"
# filenames = ["exp_217"]

# filenames = [filenames[len(filenames)-1]]

n_reps = 5
append_timestamp = False
save_best_model = True

if n_reps > 1:
    append_timestamp = False
    save_best_model = True
else:
    save_best_model = False

use_rnn = False

prep = Preprocessing()

if config["one_hot_encoding"]:
    prep.create_encoder(prep.adapt_input(
        generator.generate_binary(config["n_valves"])))

# create separate models for each data file
for filename in filenames:
    data_file = root_data_folder + "/" + filename + ".csv"
    x, y, _, _ = loader.load_dataset(data_file)

    ##
    # process input data
    print("select data")

    print(y)
    # binarize the outputs
    y = loader.binarize(y)

    if config["one_hot_encoding"]:
        y = prep.encode(prep.adapt_input(y))

    # print(y[:1])
    # print(prep.decode_int_onehot(y))

    # quit()    

    train_percent = config["train_percent"]

    # train_percent = 50

    x_train, y_train = classifiers.split_dataset_train(
        x, y, train_percent)

    x_eval, y_eval = classifiers.split_dataset_test(
        x, y, train_percent)

    # bookmark_index = 1
    # x_train = x[0:bookmarks[bookmark_index], :]
    # y_train = y[0:bookmarks[bookmark_index], :]
    # x_eval = x[0:bookmarks[len(bookmarks)-1], :]
    # y_eval = y[0:bookmarks[len(bookmarks)-1], :]

    # x = loader.remove_col(x, 1)
    # y = loader.remove_col(y, 1)
    print("end select data")
    # quit(0)
    ##

    sizex = np.shape(x_train)

    # print(sizex)
    # quit()

    top_acc = 0
    top_model_filename = None

    # run multiple evaluations (each training may return different results in terms of accuracy)
    for i in range(n_reps):
        print("evaluating model rep: " + str(i) + "/" + str(n_reps))

        # session = K.get_session()
        model_file = root_crt_model_folder + "/" + filename
        model_file_raw = model_file
        model_file_raw += "_" + str(i+1)
        if append_timestamp:
            app = "_" + str(time.time())
            model_file_raw += app
        model_file = model_file_raw + ".h5"

        # create tensorflow graph session
        graph = tf.Graph()
        with tf.Session(graph=graph):

            if use_rnn:
                tstart = time.time()
                model = deep_learning.create_model_RNN(x_train, y_train, config["activation_fn"], config["loss_fn"])
                dt = time.time() - tstart

                deep_learning.dl_save_model(model, model_file)

                fsize = os.stat(model_file).st_size

                deep_learning.eval_write_info_RNN(
                    model, x_eval, y_eval, model_file, dt, fsize)

                acc = deep_learning.eval_model_RNN(
                    model, x_eval, y_eval, sizex[1])
            else:
                tstart = time.time()
                model = deep_learning.create_model(x_train, y_train, config["activation_fn"], config["loss_fn"])
                dt = time.time() - tstart

                deep_learning.dl_save_model(model, model_file)

                fsize = os.stat(model_file).st_size

                deep_learning.eval_write_info(
                    model, x_eval, y_eval, model_file, dt, fsize)

                acc = deep_learning.eval_model(
                    model, x_eval, y_eval, sizex[1], use_rnn)

            if acc > top_acc:
                top_acc = acc
                top_model_filename = model_file_raw

            if i == n_reps - 1:
                if save_best_model:
                    copy2(top_model_filename + ".h5",
                          top_model_filename + "_top.h5")
                    copy2(top_model_filename + ".h5.txt",
                          top_model_filename + "_top.h5.txt")

        # break

        # K.clear_session()
