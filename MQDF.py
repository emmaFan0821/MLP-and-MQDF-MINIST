import numpy as np
import struct
import matplotlib.pyplot as plt
from load_data import load_data


def shuffle(Z):
    """shuffle data"""
    index = [i for i in range(len(Z))]
    np.random.seed(20)
    np.random.shuffle(index)  # use Index to shuffle
    Z = Z[index, :]
    return Z


def mqdf(x_j, e_value_i, e_vector_i, mean_i, k):
    delta = np.sum(e_value_i[k: 784]) / (784 - k)
    # delta = e_value_i[k]
    # delta = e_value_i[k+1]
    epsilon = (np.sum((x_j - mean_i) ** 2) - np.sum(np.dot((x_j - mean_i), np.array(e_vector_i[:, 0:k])) ** 2))
    first = np.sum((np.dot((x_j - mean_i), np.array(e_vector_i[:, 0:k])) ** 2) / e_value_i[0:k])
    second = epsilon / delta
    third = np.sum(np.log(e_value_i[0:k].real))
    fourth = (784 - k) * np.log(delta)
    g2 = -first - second - third - fourth
    return g2


if __name__ == '__main__':
    a = 60000
    b = 10000
    train_data, train_label, x_test, y_test = load_data(a, b)
    '''normalization and split data'''
    train_data = train_data / 255
    x_test = x_test / 255
    x_train = train_data[0:55000, :]
    x_valid = train_data[55000:60000, :]
    y_train = train_label[0:55000, 0].reshape(55000,1)
    y_valid = train_label[55000:60000, 0].reshape(5000,1)

    '''classify X_train by the value of label'''
    XY = np.hstack((x_train, y_train))  # 55000*785
    XY = XY[np.lexsort(XY.T)]
    '''calculate covariance, eigenvalue, eigenvector, mean'''
    x = [[], [], [], [], [], [], [], [], [], []]
    cov = [[], [], [], [], [], [], [], [], [], []]
    e_value = [[], [], [], [], [], [], [], [], [], []]
    e_vector = [[], [], [], [], [], [], [], [], [], []]
    mean = [[], [], [], [], [], [], [], [], [], []]
    for k in range(55000):
        for i in range(10):
            if XY[k, 784] == i:
                x[i].append(XY[k, 0:784])
    for i in range(10):
        cov[i] = np.cov(np.mat(x[i]).T) / 784
        e_value[i], e_vector[i] = np.linalg.eig(cov[i])
        mean[i] = np.mean(x[i], axis=0)
    print('--------validation----------')
    k_list = []
    k = 20
    for n in range(10):
        k += 1
        k_list.append(k)
    overall_accuracy = []
    for k in (k_list):
        prediction = []
        for j in range(len(x_valid)):
            x_j = x_valid[j]  # (1x784) vector to be classified
            P = []
            # The probabilities are calculated separately for each category
            for i in range(10):
                e_value_i = e_value[i].real
                e_vector_i = e_vector[i].real
                mean_i = mean[i]
                g = mqdf(x_j, e_value_i, e_vector_i, mean_i, k)
                P.append(g)
            likely_class = P.index(max(P))
            prediction.append(likely_class)

        count = 0
        prediction = np.array(prediction)
        prediction = prediction.reshape(5000, 1)
        for i in range(0, 5000):
            if prediction[i, 0] == y_valid[i, 0]:
                count += 1
        accuracy = 100 * (count / 5000)
        overall_accuracy.append(accuracy)
        print("k = %f, The validation accuracy rate is：%.2f %%" % (k, accuracy))
    best_k = k_list[overall_accuracy.index(max(overall_accuracy))]
    print("The optimized k is", best_k)
    plt.plot(k_list, overall_accuracy)
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.show()
    print('--------test----------')
    prediction = []
    k=best_k
    for j in range(len(x_test)):
        x_j = x_test[j]  # (1x784) vector to be classified
        P = []
        # The probabilities are calculated separately for each category
        for i in range(10):
            e_value_i = e_value[i].real
            e_vector_i = e_vector[i].real
            mean_i = mean[i]
            g = mqdf(x_j, e_value_i, e_vector_i, mean_i, k)
            P.append(g)
        likely_class = P.index(max(P))
        prediction.append(likely_class)

    count = 0
    prediction = np.array(prediction)
    prediction = prediction.reshape(10000, 1)
    for i in range(0, 10000):
        if prediction[i, 0] == y_test[i, 0]:
            count += 1
    accuracy = 100 * (count / 10000)
    print("k = %f, The validation accuracy rate is：%.2f %%" % (k, accuracy))
















