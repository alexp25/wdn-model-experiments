# first neural network with keras tutorial
# from numpy import genfromtxt
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.neural_network import MLPRegressor
from keras.callbacks import EarlyStopping

import time


class DeepLearningControlSim:

    def __init__(self):
        self.epochs = 10
        self.batch_size = 10
        self.model = None
        self.u = 0
        self.y = 0
        self.ref = 10
        self.kp = 1
        self.ki = 1
        self.kd = 0
        self.ts = 0.1
        self.acc = 0
        self.err = 0
        self.err_vect = []
        self.params = []
        self.params_vect = []

    def create_model(self, input_size, output_size):
        # define the keras model

        self.model = Sequential()

        # We use the 'add()' function to add layers to our model. We will add two layers and an output layer.
        # 'Dense' is the layer type. Dense is a standard layer type that works for most cases.
        # In a dense layer, all nodes in the previous layer connect to the nodes in the current layer.

        self.model.add(Dense(12, input_dim=input_size, activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(output_size, activation='sigmoid'))

        # compile the keras model
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam', metrics=['accuracy'])

        return self.model

    def run_sample(self, fit, deltau):
        # simulate model output
        self.y = self.y * 0.5 + self.u * 0.5

        err = self.ref - self.y
        self.acc = self.ki * (err) * self.ts
        derr = (err - self.err) / self.ts

        derr = 0

        self.u = self.kp * err + self.acc * self.ki + derr * self.kd + deltau
        self.err = err
        self.err_vect.append(err)
        # self.err = err

        self.params = [self.kp, self.ki, self.kd]
        self.params_vect.append(self.params)

        # for current error fit params
        if fit:
            npx = np.array(self.err_vect)
            npy = np.array(self.params_vect)

            print("fit")
            # print(npx, npy)

            acc = self.fit_model_sample(npx, npy)

            # we want 0 error, predict new params
            print("predict")
            params = self.predict_model(np.array([0]))
            params = params[0]
            self.kp = params[0]
            self.ki = params[1]
            self.kd = params[2]
            print(params)


        # self.err_vect = self.err_vect[-10:]
        # self.params_vect = self.params_vect[-10:]
        print(self.ref, self.u, self.y, self.err)

    # def run_sample(self, fit):
    #     # simulate model output
    #     self.y = self.y * 0.5 + self.u * 0.5

    #     err = self.ref - self.y

    #     self.err_vect.append([self.y, err])
    #     self.params_vect.append(self.u)

    #     # for current error fit params
    #     if fit:
    #         npx = np.array(self.err_vect)
    #         npy = np.array(self.params_vect)

    #         print("fit")
    #         print(npx, npy)

    #         acc = self.fit_model_sample(npx, npy)

    #         # we want 0 error, predict new params
    #         print("\npredict")
    #         params = self.predict_model(np.array([[self.ref, 0]]))
    #         params = params[0]
    #         self.u = params[0]
    #         print(params)

    #     self.err_vect = self.err_vect[-10:]
    #     self.params_vect = self.params_vect[-10:]
    #     print(self.y)

    def fit_model_sample(self, x, y):

         # set early stopping monitor so the model stops training when it won't improve anymore
        early_stopping_monitor = EarlyStopping(patience=3)

        # fit the keras model on the dataset
        h = self.model.fit(x, y, epochs=self.epochs,
                           batch_size=self.batch_size, verbose=0, callbacks=[early_stopping_monitor])

        # evaluate the keras model
        print("eval")
        _, accuracy = self.model.evaluate(x, y, verbose=0)
        print('Accuracy: %.2f' % (accuracy * 100))

        return accuracy * 100

    def predict_model(self, x):
         # make probability predictions with the model
        predictions = self.model.predict(x)
        return predictions


if __name__ == "__main__":
    dlcs = DeepLearningControlSim()

    dlcs.create_model(1, 3)

    t0 = time.time()

    c = 0
    while True:
        time.sleep(0.01)
        t1 = time.time()
        if t1 - t0 >= dlcs.ts:
            t0 = t1

            fit = c % 10 == 0
            # fit = True
            # if fit:
            #     dlcs.ref += 1

            if fit:
                deltau = 5
            else:
                deltau = -5

            dlcs.run_sample(fit, deltau)
            c += 1
            if c >= 1000:
                break

    # print(dlcs.params_vect)
    # with open("params_evolution.txt", "w") as f:
    #     for p in dlcs.params_vect:
    #         f.write(",".join([str(p1) for p1 in p]) + "\n")
