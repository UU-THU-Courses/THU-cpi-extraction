#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Define global path variables for dataset files                                              #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
train_ft = "data/features/training_features.csv"
devel_ft = "data/features/development_features.csv"
tests_ft = "data/features/testing_features.csv"

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Column headers for different types of data files                                            #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
column_name = ["shape", "chem_loc", "gene_loc", "int_loc", "chem_len", "gene_len", "sent_len",
               "chem_gene_dist", "chem_int_dist", "gene_int_dist", "label"]

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   main function to train the multi-layered neural network.                                    #
#                                                                                               #
#***********************************************************************************************#
def train_mnn():
    # read the csve files containing features
    train_data = pd.read_csv(train_ft, sep=',', lineterminator='\n', header=None, names=column_name, keep_default_na=False)
    devel_data = pd.read_csv(devel_ft, sep=',', lineterminator='\n', header=None, names=column_name, keep_default_na=False)
    tests_data = pd.read_csv(tests_ft, sep=',', lineterminator='\n', header=None, names=column_name, keep_default_na=False)
    # create a dictionary of labels to be used for rest of this module
    lbl_dictionary = create_dictionary(train_data["label"].to_numpy())
    # get features and labels as separate lists
    tr_features, tr_labels = split_features(train_data, lbl_dictionary)
    dv_features, dv_labels = split_features(devel_data, lbl_dictionary)
    ts_features, ts_labels = split_features(tests_data, lbl_dictionary)
    #(tr_mnn_features, tr_mnn_labels, ts_mnn_features, tr_cnn_features, tr_cnn_labels, ts_cnn_features, n_classes):
    # begin training process
    train(tr_features, tr_labels, dv_features, dv_labels, len(lbl_dictionary))

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   function to read the features from data files and split them into features/labels.          #
#                                                                                               #
#***********************************************************************************************#
def split_features(features_data, lbl_dictionary):
    features = features_data[["shape", "chem_loc", "gene_loc", "int_loc", "chem_len", "gene_len",
                              "sent_len", "chem_gene_dist", "chem_int_dist", "gene_int_dist"]].to_numpy()
    named_labels = features_data["label"].to_numpy()
    labels = np.empty(0)
    # create a numeric label list
    for name in named_labels:
        labels = np.append(labels, lbl_dictionary[name])
    # return the split values for training
    return features, labels

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   function to create a label dictionary.                                                      #
#                                                                                               #
#***********************************************************************************************#
def create_dictionary(labels):
    labelList ={label for label in labels}
    return {label: i for i, label in enumerate(labelList)}

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   train()                                                                                     #
#                                                                                               #
#   Description:                                                                                #
#   The training module of the project. Responsible for training the parameters for provided    #
#   features and selected options.                                                              #
#                                                                                               #
#***********************************************************************************************#
def train(tr_mnn_features, tr_mnn_labels, ts_mnn_features, ts_mnn_labels, n_classes):
    # call the multi-layer neural network to get results
    mnn_y_pred, mnn_probs, mnn_pred = tensor_multilayer_neural_network(tr_mnn_features, tr_mnn_labels, ts_mnn_features, n_classes, training_epochs=1000)
    # ensemble the results to get combined prediction
    #return ensemble_results(mnn_probs, mnn_pred, cnn_1d_probs, cnn_2d_probs)

#***********************************************************************************************#
#                                                                                               #
#   Module:                                                                                     #
#   tensor_multilayer_neural_network()                                                          #
#                                                                                               #
#   Description:                                                                                #
#   Using tensorflow library to build a simple multi layer neural network.                      #
#                                                                                               #
#***********************************************************************************************#
def tensor_multilayer_neural_network(tr_features, tr_labels, ts_features, n_classes, training_epochs):
    # initialize the beginning paramters.
    n_dim = tr_features.shape[1]
    n_hidden_units_1 =  50   #280
    n_hidden_units_2 =  50   #300
    n_hidden_units_3 =  50   #300
    n_hidden_units_4 =  50   #300
    n_hidden_units_5 =  50   #300

    sd = 1 / np.sqrt(n_dim)

    # one hot encode from training labels
    tr_labels = to_categorical(tr_labels)

    X = tf.placeholder(tf.float32,[None,n_dim])
    Y = tf.placeholder(tf.float32,[None,n_classes])

    # initializing starting learning rate - will use decaying technique
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.1, global_step, 50, 0.50, staircase=True)

    # initialize layer 1 parameters
    W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_1], mean = 0, stddev=sd))
    b_1 = tf.Variable(tf.random_normal([n_hidden_units_1], mean = 0, stddev=sd))
    h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)

    # initialize layer 2 parameters
    W_2 = tf.Variable(tf.random_normal([n_hidden_units_1,n_hidden_units_2], mean = 0, stddev=sd))
    b_2 = tf.Variable(tf.random_normal([n_hidden_units_2], mean = 0, stddev=sd))
    h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)

    # initialize layer 3 parameters
    W_3 = tf.Variable(tf.random_normal([n_hidden_units_2,n_hidden_units_3], mean = 0, stddev=sd))
    b_3 = tf.Variable(tf.random_normal([n_hidden_units_3], mean = 0, stddev=sd))
    h_3 = tf.nn.sigmoid(tf.matmul(h_2,W_3) + b_3)

    # initialize layer 4 parameters
    W_4 = tf.Variable(tf.random_normal([n_hidden_units_3,n_hidden_units_4], mean = 0, stddev=sd))
    b_4 = tf.Variable(tf.random_normal([n_hidden_units_4], mean = 0, stddev=sd))
    h_4 = tf.nn.sigmoid(tf.matmul(h_3,W_4) + b_4)

    # initialize layer 5 parameters
    W_5 = tf.Variable(tf.random_normal([n_hidden_units_4,n_hidden_units_5], mean = 0, stddev=sd))
    b_5 = tf.Variable(tf.random_normal([n_hidden_units_5], mean = 0, stddev=sd))
    h_5 = tf.nn.sigmoid(tf.matmul(h_4,W_5) + b_5)

    W = tf.Variable(tf.random_normal([n_hidden_units_5,n_classes], mean = 0, stddev=sd))
    b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
    y_ = tf.nn.softmax(tf.matmul(h_5,W) + b)

    cost_function = -tf.reduce_sum(Y * tf.log(y_))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function, global_step=global_step)

    init = tf.global_variables_initializer()

    cost_history = np.empty(shape=[1],dtype=float)
    y_pred = None
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            # running the training_epoch numbered epoch
            _,cost = sess.run([optimizer,cost_function],feed_dict={X:tr_features,Y:tr_labels})
            cost_history = np.append(cost_history,cost)
        # predict results based on the trained model
        y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: ts_features})
        y_k_probs, y_k_pred = sess.run(tf.nn.top_k(y_, k=n_classes), feed_dict={X: ts_features})

    # plot cost history
    df = pd.DataFrame(cost_history)
    df.to_csv("../data/cost_history_mnn.csv")

    # return the predicted values back to the calling program
    return y_pred, y_k_probs, y_k_pred


