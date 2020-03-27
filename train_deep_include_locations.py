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

with open("config.yml", "r") as f:
    config = yaml.load(f)

root_data_folder = config["root_data_folder"]
root_crt_model_folder = config["root_crt_model_folder"]
# read the data from the csv file
# input_file = "./PastHires.csv"
input_file = config["input_file"]
filenames = config["filenames"]
bookmarks = config["bookmarks"]

if config["run_clean"]:
    loader.clean(root_crt_model_folder)
    
n_reps = 1
use_saved_model = False
append_timestamp = True
save_best_model = True

# todo: real coords
coords = [(271, 134), (147, 222), (77, 420),
          (273, 420), (363, 246), (447, 131)]

if n_reps > 1:
    use_saved_model = False
    append_timestamp = True
    save_best_model = True
else:
    save_best_model = False

# create separate models for each data file
for filename in filenames:
    data_file = root_data_folder + "/" + filename + ".csv"
    x, y, _, _ = loader.load_dataset(data_file)

    ##
    # process input data
    print("select data")

    train_percent = config["train_percent"]

    # train_percent = 50

    ##
    # process input data
    # print("select data")
    # y2 = None
    # for (r, y1) in enumerate(y):
    #     y21 = y1
    #     found = False
    #     for (i, y11) in enumerate(y1):
    #         if y11 == 1:
    #             y21 = np.append(y21, [coords[i][0]], axis=0)
    #             y21 = np.append(y21, [coords[i][0]], axis=0)
    #             found = True
    #             break
    #     if not found:
    #         y21 = np.append(y21, [0,0], axis=0)
    #     # print(np.shape(y2))
    #     # print(np.shape(y21))
    #     # print(y21)
    #     if y2 is None:
    #         y2 = [y21]
    #         # print(y2)
    #         # print([y21])
    #     else:
    #         y2 = np.append(y2, [y21], axis=0)

    # y = y2

    # print(y)

    # quit(0)

    x_train, y_train = classifiers.split_dataset_train(
        x, y, train_percent)

    x_eval, y_eval = classifiers.split_dataset_test(
        x, y, train_percent)


    print("end select data")
    quit(0)
    ##   

    sizex = np.shape(x_train)
 

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

            model = deep_learning.create_model(x_train, y_train, config["activation_fn"], config["loss_fn"])
            deep_learning.dl_save_model(model, model_file)
            deep_learning.eval_write_info(
                model, x_eval, y_eval, model_file, 0, 0)

            acc = deep_learning.eval_model(model, x_eval, y_eval, sizex[1], False)
        
            if acc > top_acc:
                top_acc = acc
                top_model_filename = model_file_raw

            if i == n_reps - 1:
                if save_best_model:
                    copy2(top_model_filename + ".h5",
                          top_model_filename + "_top.h5")
                    copy2(top_model_filename + ".h5.txt",
                          top_model_filename + "_top.h5.txt")

        # K.clear_session()
