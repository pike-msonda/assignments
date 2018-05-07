import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

def init_weights(shape):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_

def get_data():
    pic_data = pd.read_csv("damar.csv", sep=';', decimal=',')
    classes =  pic_data['sinif']
    data = pic_data.drop('sinif', axis=1)
    
    n, m = data.shape
    all_x = np.ones ((n, m + 1))
    all_x[:, 1:] = data

    num_labels = len(np.unique(classes))
    le = preprocessing.LabelEncoder()
    encoded_classes =  le.fit_transform(classes)
    all_y =  np.eye(num_labels)[encoded_classes]
    return train_test_split(all_x, all_y, test_size=0.33, random_state=RANDOM_SEED)
def main():
    train_X, test_X, train_Y, test_Y =  get_data()

    x_size = train_X.shape[1]
    h_size = 226
    y_size = train_Y.shape[1]

    X = tf.placeholder("float", shape=[None, x_size])
    Y = tf.placeholder("float", shape=[None, y_size])

    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.02).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(100):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], Y: train_Y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_Y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X, Y: train_Y}))
        test_accuracy  = np.mean(np.argmax(test_Y, axis=1) ==
                                 sess.run(predict, feed_dict={X: test_X, Y: test_Y}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    sess.close()

if __name__ == '__main__':
    main()

    