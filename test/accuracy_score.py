import random as rnd
import numpy as np
from sklearn.metrics import accuracy_score



def compute_accuracy(y_test, y_pred):
    y_pred_norm = []

    for elem in y_pred:
        line = [ 0 ] * len(elem)

        try:
            # if an error appears here
            # get a random class
            line[elem.tolist().index(max(elem.tolist()))] = 1
        except:
            print("Error for getting predicted class")
            line[rnd.randint(0, len(l))] = 1
        y_pred_norm.append(line)

    y_p = np.argmax(np.array(y_pred_norm), 1)
    y_t = np.argmax(np.array(y_test), 1)

    print(y_p)
    print(y_t)
    accu = accuracy_score(y_t, y_p)
    return accu

if __name__ == "__main__":
    # y_test este o matrice care tine pe linii un one hot encoder cu clasele reale pentru o observatie
    # y_pred este o matrice care tine pe linii un one hot encoder predictia, i.e., procentajele de apartenenta a unei observatii la o clasa
    y_true = np.array([[0, 0, 1], [ 1, 0, 0], [0,1,0], [0,0,1], [0,1,0]])
    y_pred = np.array([[.2,.2, .8], [.3, .5, .7], [.2, .2, .2], [.2, .3, .3], [.2, .7, .2] ])
    print(compute_accuracy(y_true, y_pred))