import numpy as np

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(x, 0)

#write the derivative of relu for np array
def d_relu(x):
    return 1 * (x > 0)

def preprocess_split(path1, path2):
    #split data into target and test data
    train_data = np.genfromtxt(path1, delimiter=" ")
    test_data = np.genfromtxt(path2, delimiter=" ")
    #split data into X and y for both train and test
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    #take the y values and turn them into 1s and 0s
    for i in range(y_train.shape[0]):
        if y_train[i] == 1:
            y_train[i] = 1
        else:
            y_train[i] = 0

    for i in range(y_test.shape[0]):
        if y_test[i] == 1:
            y_test[i] = 1
        else:
            y_test[i] = 0
    
    #normalize X values
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

    return X_train.T, X_test.T, y_train.T, y_test.T

class MLP: 
    def __init__(self, path_train, path_test, num_hidden=64, epochs = 1000, lr = 1e-1):
        #Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = preprocess_split(path_train, path_test)
        self.epochs = epochs
        self.lr = lr
        self.num_samples_train = self.X_train.shape[1]
        self.num_samples_test = self.X_test.shape[1]

        #create weights sampled from a gaussian and 0s for bias's 
        W1 = np.random.normal(0, .1, size=(num_hidden, self.X_train.shape[0]))
        W2 = np.random.normal(0, .1, size=(1, num_hidden))
        b1 = np.zeros((num_hidden, 1))
        b2 = np.zeros((1, 1))   

        #shove it into a dictionary
        self.parameters = {"W1": W1, "W2": W2, "b1": b1, "b2": b2}

        #shove all gradients into a dict and set to 0
        self.grads = {"dW1": np.zeros(W1.shape), "dW2": np.zeros(W2.shape), "db1": np.zeros(b1.shape), "db2": np.zeros(b2.shape)}

        #create dict set to none for forward pass
        self.cache = {"Z1": None, "A1": None, "Z2": None, "A2": None}
    
    #crossentropy fucntion for loss
    def loss(self, y, target):
        loss = -np.sum((target * np.log(y)) + ((1 - target) * np.log(1 - y)))
        cost = loss / y.shape[1]
        return cost
    
    #compute the forward pass for the network while cacheing values
    def forward(self, X): 
        self.cache['Z1'] = self.parameters['W1'] @ X + self.parameters["b1"]
        self.cache['A1'] = relu(self.cache['Z1'])
        self.cache['Z2'] = self.parameters["W2"] @ self.cache['A1'] + self.parameters["b2"]
        self.cache['A2'] = sigmoid(self.cache['Z2'])
        return self.cache['A2']
    
    #compute forward pass without caching and saving the vals for backprop
    #to be used for prediction and test data
    def forward_nocache(self, X): 
        Z1 = self.parameters['W1'] @ X + self.parameters["b1"]
        A1 = relu(Z1)
        Z2 = self.parameters["W2"] @ A1 + self.parameters["b2"]
        A2 = sigmoid(Z2)
        return A2
    
    #backprop 
    def backprop(self):
        A1 = self.cache["A1"]
        A2 = self.cache["A2"]
        Z1 = self.cache["Z1"]
        Z2 = self.cache["Z2"]
        #calculate error
        error = (A2 - self.y_train)
        #calculate gradients
        dw2 = (1/ self.num_samples_train) * (error @ A1.T)
        dz1 = (self.parameters["W2"].T @ error) * d_relu(A1)
        dw1 = (1 / self.num_samples_train) * (dz1 @ self.X_train.T)
        db2 = (1 / self.num_samples_train) * np.sum(error, axis = 1, keepdims=True)
        db1 = (1 / self.num_samples_train) * np.sum(dz1, axis = 1, keepdims=True)
        self.grads['dW1'] = dw1
        self.grads['dW2'] = dw2
        self.grads['db1'] = db1
        self.grads['db2'] = db2
        #update weights
        self.parameters["W1"] -= self.lr * dw1
        self.parameters["W2"] -= self.lr * dw2
        self.parameters["b1"] -= self.lr * db1
        self.parameters["b2"] -= self.lr * db2

    def train(self):
        for i in range(self.epochs):
            self.forward(self.X_train)
            self.backprop()
            if i % 100 == 0:
                #compute forward pass on test data 
                output_test = self.forward_nocache(self.X_test)
                print("Epoch: ", i, "Loss Train: ", self.loss(self.cache["A2"], self.y_train), ", Loss Test: ", self.loss(output_test, self.y_test))
    def predict(self, X):
        predict = self.forward(X)
        predictions = predict > .5
        return predictions.astype(int)
    

def main():
    #input two paths for training and testing data 
    train_path = input("Enter path to training data: ")
    test_path = input("Enter path to testing data: ")
    mlp = MLP(train_path, test_path)
    mlp.train()
    predictions_train = mlp.predict(mlp.X_train)
    predictions_test = mlp.predict(mlp.X_test)
    #print accuracy percentage
    print("Accuracy_Train: ", np.mean(predictions_train == mlp.y_train) * 100, "%", 
          "Accuracy_Test: ", np.mean(predictions_test == mlp.y_test) * 100, "%")


if __name__ == "__main__":
    main()


