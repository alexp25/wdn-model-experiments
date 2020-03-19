
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class Preprocessing:
    def __init__(self):
        self.encoder = None

    ## adapt input, from list of list of binary values to list of string encoded binary values
    def adapt_input(self, data):
        data_str = []
        for row in data:
            row_str = self.adapt_input_core(row)
            data_str.append([row_str])
        # print(data_str)
        return data_str

    ## adapt input from list of binary values to a string encoded binary value
    def adapt_input_core(self, data):
        return "".join([str(int(e)) for e in data])
      
    ## create one hot encoder
    def create_encoder(self, data):
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        # enc = OrdinalEncoder()
        self.encoder.fit(data)

        # print(self.encoder.categories_)
        # td = enc.transform(data).toarray()
        td = self.encoder.transform(data)
        # print("encoded:")
        # print(td)
        return td

    ## encode via one hot encoder
    def encode(self, data):
        encoded = self.encoder.transform(data)
        # print(encoded)
        return encoded

    ## decode via one hot encoder
    def decode(self, data):
        decoded = self.encoder.inverse_transform(data)
        # print(decoded)
        return decoded

    ## decode from string encoded binary value into an int
    def binary_to_int(self, val_s):
        int_b = 0
        val_s = val_s[::-1]
        p = 0
        for c in val_s:
            int_b1 = int(c)
            if int_b1 == 1:
                int_b += pow(2, p)
            p += 1
        return int_b

    ## decode from list of lists of binary values into a list of ints
    def decode_int(self, data):
        ints = []
        for b in data:
            b = b[0]
            # print(b)
            int_b = self.binary_to_int(b)
            ints.append(int_b)
        return ints

    ## decode from list of lists of one hot encoded binary values into a list of ints
    def decode_int_onehot(self, data):
        ints = []
        for b in data:
            # self.adapt_input_core(b)
            # print("data: ", b)
            decoded_binary = self.decode([b])[0]
            # print(decoded_binary)
            int_b = self.binary_to_int(decoded_binary[0])
            ints.append(int_b)
        return ints

if __name__ == "__main__":
    preprocessing = Preprocessing()
    # data = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
    # data = [["A"], ["B"], ["C"]]
    data = [["000"], ["001"], ["010"], ["011"],
            ["100"], ["101"], ["110"], ["111"]]
    encoded = preprocessing.create_encoder(data)

    print("encoded: ")
    print(encoded)

    encoded1 = preprocessing.encode([["011"]])

    print("binary to int")
    print(preprocessing.binary_to_int("110"))

    # print("data to int")
    print(preprocessing.decode_int(data))

    print(encoded1)
    print("one hot encoded to list")
    print(preprocessing.decode_int_onehot(encoded1))

    decoded = preprocessing.decode(encoded)

    
