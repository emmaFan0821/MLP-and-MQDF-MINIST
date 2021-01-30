from load_data import load_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

def main():
    a = 60000
    b = 10000
    '''load data'''
    train_data, train_label, x_test, y_test = load_data(a, b)
    train_data = (train_data / 255)
    x_test = (x_test / 255).T
    x_train = train_data[0:55000, :].T
    x_valid = train_data[55000:60000, :].T
    y_train = train_label[0:55000, 0].reshape(55000, 1)
    y_valid = train_label[55000:60000, 0].reshape(5000, 1)
    '''onehot'''
    enc = OneHotEncoder(sparse=False, categories='auto')
    y_train = enc.fit_transform(y_train.reshape(len(y_train), -1)).T
    y_valid = enc.fit_transform(y_valid.reshape(len(y_valid), -1)).T
    y_test = enc.fit_transform(y_test.reshape(len(y_test), -1)).T
    '''Start training'''
    # i_list = []
    # i = 0
    # for n in range(9):
    #     i += 100
    #     i_list.append(i)
    # for iterations in i_list:
    h_nodes=50
    learning_rate=4.5
    iterations=500
    cost_list, W = MLP(x_train, y_train, h_nodes, learning_rate, iterations)
    train_accuracy = predict(x_train, y_train, W)
    valid_accuracy = predict(x_valid, y_valid, W)
    test_accuracy = predict(x_test, y_test, W)
    print('h nodes=%d, learning rate=%f, iterations=%d' % (h_nodes, learning_rate, iterations))
    print("The accuracy of training set is：%.2f %%" % (train_accuracy))
    print("The accuracy of validation set is：%.2f %%" % (valid_accuracy))
    print("The accuracy of testing set is：%.2f %%" % (test_accuracy))

    plt.figure()
    plt.plot(np.arange(len(cost_list)), cost_list)
    # plt.plot(cost_list)
    plt.xlabel("iterations")
    plt.ylabel("Cost")
    plt.show()

def MLP(X, Y, h_nodes, learning_rate, iterations):
    a = X.shape[1]  # 55000
    cost_list = []
    W = initialize_W(h_nodes)
    for epoch in range(iterations):
        A2, ZA = forward_propagation(X, W)
        cost = 1 / 2 * np.sum((A2 - Y)**2)
        # cost = -np.mean(Y * np.log(A2 + 1e-8))
        dW = backward_propagation(a, X, Y, ZA)
        W = update_W(W, dW, learning_rate)

        if epoch % 50 == 0:
            print(f"Epoch: %d, Cost: %f" % (epoch, cost))
            cost_list.append(cost)
    return cost_list, W

def initialize_W(h_nodes):
    np.random.seed(1)
    W1 = np.random.randn(h_nodes, 784) / np.sqrt(784)
    b1 = np.zeros((h_nodes, 1))
    W2 = np.random.randn(10, h_nodes) / np.sqrt(h_nodes)
    b2 = np.zeros((10, 1))
    W = {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}
    return W

def forward_propagation(X, W):
    Z1 = np.dot(W['W1'], X) + W['b1']
    A1 = active(Z1)
    Z2 = np.dot(W['W2'], A1) + W['b2']
    A2 = softmax(Z2)
    ZA = {'W1': W['W1'], 'W2': W['W2'], 'A1': A1, 'A2': A2, 'Z1': Z1, 'Z2': Z2}
    return A2, ZA

def backward_propagation(a, X, Y, ZA):
    A1 = ZA['A1']
    A2 = ZA['A2']
    W2 = ZA['W2']
    Z1 = ZA['Z1']

    dW2 = np.dot(A2-Y, A1.T) / a
    db2 = np.sum(A2-Y, axis=1, keepdims=True) / a
    dZ1 = np.dot(W2.T, A2-Y) * derivative(Z1)
    dW1 = np.dot(dZ1, X.T) / a
    db1 = np.sum(dZ1, axis=1, keepdims=True) / a
    dW = {'dW1': dW1, 'dW2': dW2, 'db1': db1, 'db2': db2}
    return dW

def update_W(W, dW, learning_rate):
    W1 = W['W1'] - learning_rate * dW['dW1']
    W2 = W['W2'] - learning_rate * dW['dW2']
    b1 = W['b1'] - learning_rate * dW['db1']
    b2 = W['b2'] - learning_rate * dW['db2']
    W = {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}
    return W

def predict(X, Y, W):
    A, ZA = forward_propagation(X, W)
    output = np.argmax(A, axis=0)
    Y = np.argmax(Y, axis=0)
    accuracy = (output == Y).mean() * 100
    return accuracy

def active(x):
    return 1 / (1 + np.exp(-x))
def derivative(x):
    y = 1 / (1 + np.exp(-x))
    return y * (1 - y)
def softmax(x):
    y = np.exp(x - np.max(x))
    return y / np.sum(y, axis=0, keepdims=True)
def relu(x):
    for m in range (x.shape[0]):
        for n in range (x.shape[1]):
            x[m, n] = np.max(x[m, n],0)
    return x
def relu_derivative(x):
    for m in range(x.shape[0]):
        for n in range(x.shape[1]):
            if x[m, n] > 0:
                x[m, n] = 1
            else:
                x[m, n] = 0
    return x

if __name__ == '__main__':
    main()