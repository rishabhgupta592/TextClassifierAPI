import tensorflow as tf
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
import _pickle as pickle

# model_path = './../models/nn/'
model_path = './../models/cnn/'
import numpy as np


def load_models(feature):
    # saver = tf.train.import_meta_graph('./models/text_classifier/model.meta')
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(model_path + '/model.ckpt.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(model_path))
        graph = tf.get_default_graph()
        train_features = np.pad(feature, ((0, 0), (0, 73)), 'constant')
        train_features = train_features.astype('float16')
        # X = tf.placeholder("float", [None, 861], name='X')
        X = graph.get_tensor_by_name("input_X:0")
        prediction = graph.get_tensor_by_name("prediction:0")
        prediction_prob = graph.get_tensor_by_name("pred_prob:0")
        print(sess.run(prediction_prob, feed_dict={X:train_features}))
        print("Class: ",sess.run(prediction, feed_dict={X: train_features}))
        # return sess.run(prediction, feed_dict={X:feature})


def get_class(query):
    query = [query]
    # from sklearn.feature_extraction.text import CountVectorizer
    # count_vect = joblib.load('feature.pkl')
    count_vect = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(model_path+"feature.pkl", "rb")))
    X_train_counts = count_vect.fit_transform(query)
    feature = X_train_counts.toarray()
    # print(feature)
    return load_models(feature)


if __name__ == "__main__":
    # query = "Work Schedule Discover employees"
    query = " Benefits Enrollment Deductions in Default Benefits: Day One"
    print(get_class(query))