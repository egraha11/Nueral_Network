import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class Nueral_Network():

    def __init__(self):

        self.learning_rate = .0001
   

    def feed_forward(self, x):

        #muliply each input by each of its respective weights for each nueron in the first hidden layer
        self.layer1 = np.dot(x, self.w1.T)

        #add the bias term to weighted outputs
        self.layer1 = self.layer1 + self.b1
        

        #pass outputs through the relu activation function
        self.layer1 = np.maximum(0, self.layer1)

        output_layer = np.dot(self.layer1, self.w2[0].T)
        

        self.output = output_layer + self.b2

        #Linear activation function (no output function needed)
        

    def back_prop(self, y, x):

        #derivitive of residual sum of squares
        loss_function_deriv = 2 * (y - self.output)

        #derivitive of linear activation function is one for each term (no action needed) (for regressive nueral networks)
        #derivitive of relu activation function is 0 when 0 and 1 when greater than 0

        #update bias term of output
        self.b2 += (loss_function_deriv * self.learning_rate)
        self.b1 += (loss_function_deriv * self.learning_rate * self.layer1)

        #derivitives of the output layer weights
        d_w2 = self.layer1 * loss_function_deriv

        #update hidden layer weights
        #outer = np.multiply.outer(x, self.w2[0])
        d_w1 = x.T * self.w2[0] * self.layer1 * loss_function_deriv

        self.w2 += d_w2 * self.learning_rate
        self.w1 += d_w1 * self.learning_rate



    def fit(self, x, y):

        self.x = x
        self.y = y
        #initialize the number of layers 
        self.layers = 2

        #initialize the 1st hidden layer's weights
        self.w1 = np.random.random((x.shape[1], x.shape[1]))

        #inititalize the 1st hidden layers bias vector
        self.b1 = np.random.random((x.shape[1]))

        #initialize the output layer's weights
        self.w2 = np.random.random((1, self.w1.shape[0]))

        #initialize the output layer's bias vector
        self.b2 = np.random.random(1)

        for sample in range(np.shape(x)[0]):
            self.feed_forward(x[sample])
            self.back_prop(y[sample], x[sample])

    def predict(self, x): 

        preds = []

        for sample in range(np.shape(x)[0]):
            layer1 = np.dot(sample, self.w1.T) + self.b1
            preds.append(np.sum(np.dot(layer1, self.w2.T) + self.b2))

        return preds


def main():

    df = fetch_california_housing(as_frame=True)

    x = np.array(df.data)
    y = np.array(df.target)

    x_train, x_test, y_train, y_test = train_test_split(x, y)


    model = Nueral_Network()

    model.fit(x_train, y_train)

    y_preds = model.predict(x_test)

    print(list(zip(y_preds, y_test)))

    print(mean_squared_error(y_test, y_preds))

if __name__ == "__main__":
    main()