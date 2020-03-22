# first neural network with keras tutorial
# from numpy import genfromtxt
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, SimpleRNN, LSTM
from sklearn.neural_network import MLPRegressor
from keras.callbacks import EarlyStopping

epochs = 500
epochs_RNN = 5
batch_size = 10


def reshape_RNN1(X):
    print("reshape RNN")
    sizex = np.shape(X)

    # batch_size = 1
    # time_step = sizex[0]
    # data_dim = sizex[1]

    batch_size = sizex[0]
    time_step = 1
    data_dim = sizex[1]

    # X_train = np.reshape(X, (batch_size, time_step, data_dim))
    X_train = np.reshape(X, (batch_size, time_step, data_dim))

    # [samples, timesteps, features]

    return X_train, batch_size, time_step, data_dim


def reshape_RNN(X, y):

    X_train, batch_size, time_step, data_dim = reshape_RNN1(X)
    y_train, _, _, output_dim = reshape_RNN1(y)

    return X_train, y_train, time_step, data_dim, output_dim


def create_model_RNN(X, y):

    # https://stackoverflow.com/questions/48978609/valueerror-error-when-checking-input-expected-lstm-1-input-to-have-3-dimension
    # Shape of input and output arrays for an LSTM layer should be (batch_size, time_step, dim).

    # X = np.array([[1, 2, 3], [4, 5, 6]])
    # y = np.array([[2, 4, 6], [8, 10, 12]])

    # X = X.reshape(1,2,3)
    # y = y.reshape(1,2,3)

    # data_dim = 3
    # timesteps = 2

    # model = Sequential()
    # model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim)))
    # model.add(LSTM(32, return_sequences=True))
    # model.add(Dense(3, activation='linear'))

    # print(model.summary())

    # model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

    # model.fit(X,y, batch_size=1, epochs=1000)

    # define the keras model

    X_train, y_train, time_step, data_dim, output_dim = reshape_RNN(X, y)

    # model = Sequential()

    model = Sequential()
    model.add(LSTM(32, return_sequences=True,
                   input_shape=(time_step, data_dim)))
    model.add(LSTM(32, return_sequences=True))
    # model.add(Dense(output_dim, activation='linear'))
    model.add(Dense(output_dim, activation='sigmoid'))

    # model.add(SimpleRNN(units=output_dim, input_shape=sizex))
    # model.add(Dense(24, input_dim=input_dim, activation='relu'))

    # Recurrent layer
    # model.add(LSTM(64, return_sequences=False,
    #             dropout=0.1, recurrent_dropout=0.1))

    # Fully connected layer
    # model.add(Dense(64, activation='relu'))

    # model.add(Dense(1))
    # model.add(Dense(output_dim, activation='sigmoid'))
    # model.compile(loss='mean_squared_error', optimizer='adam')

    # model.compile(loss='binary_crossentropy',
    #               optimizer='adam', metrics=['accuracy'])


    # There are three kinds of classification tasks:

    # Binary classification: two exclusive classes
    # Multi-class classification: more than two exclusive classes
    # Multi-label classification: just non-exclusive classes
    # Here, we can say

    # In the case of (1), you need to use binary cross entropy.
    # In the case of (2), you need to use categorical cross entropy.
    # In the case of (3), you need to use binary cross entropy.

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.summary()

    # model.fit(X_train, y_train, epochs=20, batch_size=1)

    train_model_RNN(model, X_train, y_train, X)

    return model


def create_model(X, y):
    # define the keras model

    sizex = np.shape(X)
    sizey = np.shape(y)

    input_dim = sizex[1]
    output_dim = sizey[1]

    model = Sequential()
    # model.add(Dense(12, input_dim=input_dim, activation='relu'))
    # model.add(Dense(8, activation='relu'))

    # We use the 'add()' function to add layers to our model. We will add two layers and an output layer.
    # 'Dense' is the layer type. Dense is a standard layer type that works for most cases.
    # In a dense layer, all nodes in the previous layer connect to the nodes in the current layer.

    # model.add(Dense(12, input_dim=input_dim, activation='relu'))
    # model.add(Dense(8, activation='relu'))

    # model.add(Dense(12, input_dim=input_dim, activation='sigmoid'))
    # model.add(Dense(8, activation='sigmoid'))

    # ReLU is used usually for hidden layers. it avoids vanishing gradient problem. Try this. For output layer, softmax to get probabilities for possible outputs.
    model.add(Dense(24, input_dim=input_dim, activation='relu'))
    model.add(Dense(16, activation='relu'))

    # Regression Loss Functions
    #     Mean Squared Error Loss
    #     Mean Squared Logarithmic Error Loss
    #     Mean Absolute Error Loss
    # Binary Classification Loss Functions
    #     Binary Cross-Entropy
    #     Hinge Loss
    #     Squared Hinge Loss
    # Multi-Class Classification Loss Functions
    #     Multi-Class Cross-Entropy Loss
    #     Sparse Multiclass Cross-Entropy Loss
    #     Kullback Leibler Divergence Loss

    # compile the keras model

    # model.add(Dense(output_dim, activation='sigmoid'))
    model.add(Dense(output_dim, activation='sigmoid'))
    # model.add(Dense(output_dim, activation='softmax'))

    # https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # It is recommended that the output layer has one node for the target variable and the linear activation function is used.

    # model.add(Dense(output_dim, activation='linear'))
    #
    # model.compile(loss='mean_squared_logarithmic_error',
    #               optimizer='adam', metrics=['accuracy'])

    # model.add(Dense(output_dim, activation='softmax'))
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam', metrics=['accuracy'])

    model.summary()

    train_model(model, X, y)

    return model


def train_model_RNN(model, X_train, y_train, X_orig):
    # set early stopping monitor so the model stops training when it won't improve anymore
    early_stopping_monitor = EarlyStopping(monitor='loss', patience=5)

    # fit the keras model on the dataset
    # model.fit(X, y, epochs=150, batch_size=10, verbose=1)
    h = model.fit(X_train, y_train,
                  epochs=epochs_RNN, batch_size=1, callbacks=[early_stopping_monitor])

    accuracy = eval_model_acc_RNN(model, X_train, y_train)
    print("\ntrain model accuracy: " + str(accuracy * 100) + "\n")
    res = reshape_RNN1([X_orig[0]])[0]
    print(res)
    predictions = model.predict(res)
    print(predictions)

    return h


def train_model(model, X, y):

    # set early stopping monitor so the model stops training when it won't improve anymore
    early_stopping_monitor = EarlyStopping(patience=10)

    # fit the keras model on the dataset
    # model.fit(X, y, epochs=150, batch_size=10, verbose=1)
    h = model.fit(X, y, validation_split=0.2, epochs=epochs,
                  batch_size=batch_size, verbose=0, callbacks=[early_stopping_monitor])

    accuracy = eval_model_acc(model, X, y)
    print("\ntrain model accuracy: " + str(accuracy * 100) + "\n")

    return h


def eval_write_info(model, X, y, model_file, dt, fsize):
    accuracy = eval_model_acc(model, X, y)
    print("\neval model accuracy: " + str(accuracy * 100) + "\n")
    write_info(model_file, accuracy, dt, fsize)
    return accuracy


def eval_write_info_RNN(model, X, y, model_file, dt, fsize):
    res = reshape_RNN(X, y)
    X = res[0]
    y = res[1]
    accuracy = eval_model_acc_RNN(model, X, y)
    print("\neval model accuracy: " + str(accuracy * 100) + "\n")
    write_info(model_file, accuracy, dt, fsize)
    return accuracy


def eval_model_acc(model, X, y):
    _, accuracy = model.evaluate(X, y, batch_size=batch_size, verbose=1)
    return accuracy


def eval_model_acc_RNN(model, X, y):
    _, accuracy = model.evaluate(X, y, batch_size=1, verbose=0)
    return accuracy


def write_file(model_file, data):
    with open(model_file + ".txt", "w") as f:
        f.write(data)


def write_info(model_file, accuracy, dt, fsize):
    with open(model_file + ".txt", "w") as f:
        acc = "accuracy: " + str(accuracy)
        f.write(acc + "\n")
        f.write("dt: " + str(dt) + "\n")
        f.write("fsize: " + str(fsize) + "\n")


def dl_save_model(model, model_file):
     # Save the model
    model.save(model_file)


def dl_load_model(model_file):
    # Recreate the exact same model purely from the file
    return load_model(model_file)


def eval_model(model, X, y, ndata, use_rnn):

    if use_rnn:
        return eval_model_RNN(model, X, y, ndata)

    # evaluate the keras model
    accuracy = eval_model_acc(model, X, y)
    print('Accuracy: %.2f' % (accuracy*100))

    # make probability predictions with the model
    predictions = model.predict(X)

    # alternately make class predictions with the model
    # predictions = model.predict_classes(X)
    sizex = np.shape(X)
    sizex = sizex[0]

    predictions = binarize_predictions(predictions, 0.2, 0.8)

    if ndata is not None:
        sizex = ndata

    # summarize the first 5 cases
    for i in range(sizex):
        print('%s => %s (expected %s)' %
              (X[i].tolist(), predictions[i].tolist(), y[i].tolist()))

    return accuracy

def predict_model_RNN(model, X):
    XR = reshape_RNN1(X)[0]
    # make probability predictions with the model
    predictions = model.predict(XR)
    return predictions

def eval_model_RNN(model, X, y, ndata):
    res = reshape_RNN(X, y)
    X = res[0]
    y = res[1]

    # evaluate the keras model
    accuracy = eval_model_acc_RNN(model, X, y)
    print('Accuracy: %.2f' % (accuracy*100))

    return accuracy

def binarize_predictions_mean(x):
    s = np.shape(x)
    rows = s[0]
    cols = s[1]

    for i in range(0, rows):
        threshold = np.mean(x[i])
        for j in range(0, cols):
            if x[i, j] > threshold:
                x[i, j] = 1
            else:
                x[i, j] = 0
                
    return x

def binarize_predictions_max(x):
    s = np.shape(x)
    rows = s[0]
    cols = s[1]

    for i in range(0, rows):
        threshold = np.max(x[i])
        for j in range(0, cols):
            if x[i, j] == threshold:
                x[i, j] = 1
            else:
                x[i, j] = 0
                
    return x

def binarize_predictions_1(x, threshold):
    s = np.shape(x)
    rows = s[0]
    cols = s[1]

    for i in range(0, rows):
        for j in range(0, cols):
            if x[i, j] > threshold:
                x[i, j] = 1
            else:
                x[i, j] = 0
                
    return x

def binarize_predictions(x, low, high):
    s = np.shape(x)

    rows = s[0]
    cols = s[1]

    low1 = 0
    high1 = 1

    for i in range(0, rows):
        for j in range(0, cols):
            done = False
            if x[i, j] > high:
                x[i, j] = 1
                done = True
            if x[i, j] < low:
                x[i, j] = 0
                done = True
            if not done:
                dist_low = abs(x[i, j] - low1)
                dist_high = abs(x[i, j] - high1)
                if dist_low < dist_high:
                    x[i, j] = 0
                else:
                    x[i, j] = 1
    return x


def create_MLP_regressor(X, y):

    # clf = MLPRegressor(solver='lbfgs', alpha=1e-5,
    #                 hidden_layer_sizes=(5, 2), random_state=1)

    clf = MLPRegressor(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
                       beta_2=0.999, early_stopping=False, epsilon=1e-08,
                       hidden_layer_sizes=(12, 8), learning_rate='constant',
                       learning_rate_init=0.001, max_iter=200, momentum=0.9,
                       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                       warm_start=False)

    clf.fit(X, y)

    print(clf.predict(X))

# def test_MLP_regressor(X, y):

#     # clf = MLPRegressor(solver='lbfgs', alpha=1e-5,
#     #                 hidden_layer_sizes=(5, 2), random_state=1)

#     clf = MLPRegressor(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(5, 2), learning_rate='constant',
#        learning_rate_init=0.001, max_iter=200, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
#        solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
#        warm_start=False)

#     X=[[-61, 25, 0.62, 0.64, 2, -35, 0.7, 0.65], [2,-5,0.58,0.7,-3,-15,0.65,0.52] ]
#     y=[ [0.63, 0.64], [0.58,0.61] ]
#     clf.fit(X,y)

#     print(clf.predict([[-61, 20, 0.62, 0.50, 2, -35, 0.5, 0.6]]))
#     print(clf.predict([[2,-5,0.58,0.7,-3,-15,0.65,0.52]]))
