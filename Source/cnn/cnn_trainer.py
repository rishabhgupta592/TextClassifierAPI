import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
import _pickle as pickle
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split

# CNN for text data
tf.logging.set_verbosity(tf.logging.INFO)

num_step = 100
display_step = 10
learning_rate = 0.01


def feature_extractor(data):
    """ Convert text into numerical features"""
    print("Feature extraction started ...")
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(data)
    # joblib.dump(count_vect.vocabulary_, 'feature.pkl')
    pickle.dump(count_vect.vocabulary_, open("./../models/cnn/feature.pkl", "wb"))
    print("Feature vector size = ", np.shape(X_train_counts))
    print("Feature extraction Done !")
    return X_train_counts.toarray()


def main(unused_argv):
    # Pre-process Data and conver it into training data
    shuffled_data = pd.read_csv('./../Data/text_classification_data3.csv')
    # shuffled_data = shuffle(data) # Shuffling data set

    text_data = shuffled_data['Que']
    intent = shuffled_data['I1']
    # intent = shuffled_data.iloc[:, 2:4]
    # intent = shuffled_data.iloc[:,1:]
    train_features = feature_extractor(text_data)

    X_train, X_test, y_train, y_test = train_test_split(train_features, intent, test_size=0.33)

    # import sys
    # sys.exit(0)
    # zero_numpy = np.zeros(30*30)
    # zero_numpy[:]
    train_features = np.pad(X_train, ((0, 0), (0, 73)), 'constant')
    train_features = train_features.astype('float16')

    X_test = np.pad(X_test, ((0, 0), (0, 73)), 'constant')
    X_test = X_test.astype('float16')
    # y_train = y_train.astype('float16')
    print("check1")
    # Create the Estimator
    model_path = './../models/cnn/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    print("check2")
    print("check3")
    # Train the model
    train_data = train_features
    # print(type(train_data))
    train_labels = y_train.as_matrix()

    labels = train_labels
    # From here start convolution

    X = tf.placeholder(tf.float32, shape=[None, 30 * 30], name='input_X')
    y_true = tf.placeholder(tf.float32, shape=[None, 2], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)
    input_layer = tf.reshape(X, [-1, 30, 30, 1])
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    # Todo add dropout
    # dropout = tf.layers.dropout(
    #     inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    """ Skipping dropout for now """

    # Todo unit should be equal to number of classes
    logits = tf.layers.dense(inputs=dense, units=10)

    # with tf.variable_scope("Softmax"):
    y_pred = tf.nn.softmax(logits, name="pred_prob")
    y_pred_cls = tf.argmax(y_pred, dimension=1, name="prediction")


    # sparse_softmax_cross_entropy => method takes input as 1, 2, 3 i.e. number of classes

    loss_op = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(loss_op)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for step in range(1, num_step + 1):
            # Feeding all sample in once
            # TODO batch size introduction.
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={X: train_data, y_true_cls: y_train})
            if step % display_step == 0 or step == 1:
                # print(sess.run(prediction, feed_dict={X: train_x, Y: train_y}))
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: train_data, y_true_cls: y_train})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

        print("Optimization Finished!")
        print("Test Accuracy", sess.run(accuracy, feed_dict={X: X_test , y_true_cls: y_test}))
        res = sess.run(y_pred_cls, feed_dict={X: train_data})
        print(res)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        save_path = model_path + '/model.ckpt'
        tf_save_path = saver.save(sess, save_path)
        print("Model saved in file: %s" % tf_save_path)


if __name__ == "__main__":
    # main('unused_argv')
    tf.app.run()