
# Text classifier with two classes
# Data is some policy. So two policies are ingested.
#

import tensorflow as tf
import pandas as pd
import _pickle as pickle
import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
# >>>

# Network param
n_hidden1 = 128     # neurons in first hidden layer
n_hidden2 = 128    # neurons in second hidden layer

# Training Param
learning_rate = 0.001
batch_size = 128    # Batch size defines number of samples that going to be propagated through the network.
num_step = 500      # Epochs
display_step = 100  #
model_path = './../models/nn/'


def feature_extractor(data):
    """ Convert text into numerical features"""
    print("Feature extraction started ...")
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(data)
    # joblib.dump(count_vect.vocabulary_, 'feature.pkl')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    pickle.dump(count_vect.vocabulary_, open("./../models/nn/feature.pkl","wb"))
    return X_train_counts.toarray()


def neural_net(x ,n_input, num_classes):
    # Hidden fully connected layer with 256 neurons
    weight_h1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))
    weight_h2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
    weight_out = tf.Variable(tf.random_normal([n_hidden2, num_classes]))

    bias_h1 = tf.Variable(tf.random_normal([n_hidden1]))
    bias_h2 = tf.Variable(tf.random_normal([n_hidden2]))
    bias_out = tf.Variable(tf.random_normal([num_classes]))

    layer_1 = tf.add(tf.matmul(x, weight_h1), bias_h1)
    layer_2 = tf.add(tf.matmul(layer_1, weight_h2), bias_h2)
    # Output fully connected layer with a neuron for each class
    layer_out = tf.add(tf.matmul(layer_2, weight_out), bias_out)
    return layer_out


def train_test_split(data, intent):
    train_x = data.iloc[:1500,:]
    train_y = intent.iloc[:1500, :]
    test_x = data.iloc[1500:, :]
    test_y = intent.iloc[1500:, :]
    return train_x, test_x, train_y, test_y


def neural_engine(train_x, train_y):
    # train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size = 0.33, random_state = 42)
    print("Network building start ...")
    train_x = pd.DataFrame(train_x)
    train_y = pd.DataFrame(train_y)
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y)
    n_input = len(train_x.columns) # Number of features
    num_classes = len(train_y.columns)
    X = tf.placeholder("float", [None, n_input],name='X')
    Y = tf.placeholder("float", [None, num_classes])
    # Construct model
    logits = neural_net(X, n_input, num_classes)
    prediction = tf.nn.sigmoid(logits, name="prediction") # It will return same number of output neurons as defined

    loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(loss_op)

    correct_pred = tf.equal(prediction, Y)  # Check for correct results

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # nothing but average kind of thing

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)

        for step in range(1, num_step + 1):
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={X: train_x, Y: train_y})
            if step % display_step == 0 or step == 1:
                # print(sess.run(prediction, feed_dict={X: train_x, Y: train_y}))
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: train_x,
                                                                     Y: train_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

        print("Optimization Finished!")
        print("Test Accuracy", sess.run(accuracy, feed_dict={X:test_x, Y: test_y}))
        res = sess.run(prediction, feed_dict={X: train_x})
        res = pd.DataFrame(res)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        save_path = model_path + '/model.ckpt'
        tf_save_path = saver.save(sess, save_path)
        print("Model saved in file: %s" % tf_save_path)

    return res


def dec_to_bin(x):
    return int(bin(x)[2:])

# def modify_intent_column(data):
#     intent_list = data['Intent'].unique().tolist()
#
#
#     print(dec_to_bin(total_intent_count))
#     # Add coulmns
#
#     print(type(total_intent_count))
#     print(total_intent_count)
#     pass


if __name__ == "__main__":

    # Pre-process Data and conver it into training data
    shuffled_data = pd.read_csv('./../Data/text_classification_data2.csv')
    # shuffled_data = shuffle(data) # Shuffling data set
    # modify_intent_column(shuffled_data)
    text_data = shuffled_data['Que']    # Column name which has sentences
    intent = shuffled_data.iloc[:,2:4]   # Column which has corresponding intent to a sentence
    train_features = feature_extractor(text_data)
    print("Feature extraction Done !")

    output_frame = neural_engine(train_features, intent)
    print("Training done !!!")
    # shuffled_data['Predicted Class'] = output_frame
    # output_file_name = "output/text_classfier_output.csv"
    # shuffled_data.to_csv(output_file_name)
    # print("Output file dumped !!!")