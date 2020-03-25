import numpy as np
import csv
import os
import shutil


def clean(folder):
    except_files = [".gitkeep"]
    for filename in os.listdir(folder):
        if filename in except_files:
            continue

        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                print("removing file: " + file_path)
                os.unlink(file_path)
                
            elif os.path.isdir(file_path):
                print("removing folder: " + file_path)
                shutil.rmtree(file_path)         
        
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def load_model_result_file(filename):
    accuracy = 0
    dt = 0
    fsize = 0
    with open(filename, "r") as f:
        lines = f.read().split("\n")
        for line in lines:
            if "accuracy" in line:
                accuracy = float(line.split("accuracy: ")[1])*100
            if "dt" in line:
                dt = float(line.split("dt: ")[1])
            if "fsize" in line:
                fsize = float(line.split("fsize: ")[1])
    return accuracy, dt, fsize


def load_dataset_raw_buffer(filename):
    dataset = np.genfromtxt(filename, delimiter=',')
    # split into input (X) and output (y) variables
    x = dataset[1:, 2:13]
    y = dataset[1:, 14:20]

    sizex = np.shape(x)
    sizey = np.shape(y)

    print(sizex)
    print(sizey)

    print(x)
    print(y)

    return x, y


def load_dataset_full(filename):
    # load the dataset
    dataset = np.genfromtxt(filename, delimiter=',')
    return dataset

def load_dataset_full_with_header(filename):
    # load the dataset
    header = []
    with open(filename, "r") as f:
        header = f.read().split("\n")[0].split(",")

    dataset = np.genfromtxt(filename, delimiter=',')
    dataset = dataset[1:]

    return dataset, header

def load_dataset(filename):
    # load the dataset
    with open(filename, "r") as f:
        header = f.read().split("\n")[0].split(",")

    dataset = np.genfromtxt(filename, delimiter=',')
    # split into input (X) and output (y) variables
    x = dataset[1:, 2:13]
    y = dataset[1:, 14:20]

    sizex = np.shape(x)
    sizey = np.shape(y)

    print(sizex)
    print(sizey)

    print(x)
    print(y)

    return x, y, header[2:13], header[14:20]

def remove_col(x, col):
    # numpy.delete(arr, obj, axis=None)
    return np.delete(x, col, 1)

def remove_row(x, row):
    return np.delete(x, row, 0)

def binarize(x):
    s = np.shape(x)

    rows = s[0]
    cols = s[1]

    for i in range(0, rows):
        for j in range(0, cols):
            if x[i, j] > 0:
                x[i, j] = 1
            else:
                x[i, j] = 0

    return x


if __name__ == "__main__":
    clean("./data/models/crt")
