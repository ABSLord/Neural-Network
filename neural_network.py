from activation_functions import *


class NeuralNetwork:
    def __init__(self, structure):

        self.act = sigmoid
        self.act_deriv = sigmoid_deriv

        self.output_act = softmax
        self.output_act_deriv = softmax_deriv

        self.count_of_layers = len(structure)

        self.W = []
        self.Upd = []
        self.H = [np.zeros((0, 0)) for _ in range(self.count_of_layers - 1)]

        for i in range(self.count_of_layers - 1):
            self.W.append(np.random.randn(structure[i], structure[i + 1]) / np.sqrt(structure[i]))
            self.Upd.append(0)

    def forward_propagation(self, X):
        self.H[0] = X
        for i in range(self.count_of_layers - 2):
            curr_h = self.act(np.dot(self.H[i], self.W[i]))
            self.H[i + 1] = curr_h
        y = self.output_act(np.dot(self.H[self.count_of_layers - 2], self.W[self.count_of_layers - 2]))
        return y

    def back_propagation(self, X, Y, epochs=10, learning_rate=0.2):
        for k in range(epochs):
            print(k)
            for i in range(X.shape[0]):
                y_out = self.forward_propagation(X[i])
                delta_output = np.multiply(self.output_act_deriv(y_out), Y[i] - y_out)

                deltas = [np.zeros((0, 0)) for _ in range(self.count_of_layers - 1)]

                deltas[self.count_of_layers - 2] = delta_output

                for j in range(self.count_of_layers - 3, -1, -1):
                    delta = np.multiply(self.act_deriv(self.H[j + 1]), np.dot(self.W[j + 1], deltas[j + 1]))
                    deltas[j] = delta

                for t in range(self.count_of_layers - 1):
                    self.Upd[t] = np.multiply(learning_rate, np.outer(self.H[t], deltas[t]))

                for m in range(self.count_of_layers - 1):
                    self.W[m] += self.Upd[m]

    def predict(self, X):
        Y = np.zeros([X.shape[0], self.W[self.count_of_layers - 2].shape[1]])
        for i in range(0, X.shape[0]):
            Y[i] = self.forward_propagation(X[i])
        return Y
