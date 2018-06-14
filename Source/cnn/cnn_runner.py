import tensorflow as tf
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
import _pickle as pickle

# model_path = './../models/nn/'
model_path = './../models/cnn/mnist_convnet_model/'


# def load_models(feature):
#     # saver = tf.train.import_meta_graph('./models/text_classifier/model.meta')
#     with tf.Session() as sess:
#         new_saver = tf.train.import_meta_graph(model_path + '/model.ckpt-54.meta')
#         new_saver.restore(sess, tf.train.latest_checkpoint(model_path))
#         graph = tf.get_default_graph()
#         # X = tf.placeholder("float", [None, 861], name='X')
#         X = graph.get_tensor_by_name("probabilities:0")
#         prediction = graph.get_tensor_by_name("probabilities:0")
#         print(sess.run(prediction, feed_dict={X:feature}))
#         return sess.run(prediction, feed_dict={X:feature})
#
#         full_model_dir = model_path
#         tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], full_model_dir)
#         predictor = tf.contrib.predictor.from_saved_model(full_model_dir)
#         model_input = tf.train.Example(features=tf.train.Features(
#             feature={"words": tf.train.Feature(int64_list=tf.train.Int64List(value=features_test_set))}))
#         model_input = model_input.SerializeToString()
#         output_dict = predictor({"predictor_inputs": [model_input]})
#         y_predicted = output_dict["pred_output_classes"][0]
#         return y_predicted
import numpy as np

def load_models(feature):
    # saver = tf.train.import_meta_graph('./models/text_classifier/model.meta')

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(model_path + '/model.ckpt-1.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(model_path))
        graph = tf.get_default_graph()
        # X = tf.placeholder("float", [None, 861], name='X')
        train_features = np.pad(feature, ((0, 0), (0, 73)), 'constant')
        train_features = train_features.astype('float16')
        train_features  = np.reshape(train_features, [1, 30, 30, 1])
        # batch_size = 100 set at the time of training
        # train_features = tf.reshape(train_features, [1, 30, 30, 1])
        X = graph.get_tensor_by_name("input_x:0")
        prediction_prob = graph.get_tensor_by_name("softmax_tensor:0")
        prediction_class = graph.get_tensor_by_name("output_class:0")
        #
        print(sess.run(prediction_prob, feed_dict={X:train_features}))
        print(sess.run(prediction_class, feed_dict={X: train_features}))
        # return sess.run(prediction_class, feed_dict={X: train_features})


def get_class(query):
    query = [query]
    # from sklearn.feature_extraction.text import CountVectorizer
    # count_vect = joblib.load('feature.pkl')
    count_vect = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(model_path+"cnn_feature.pkl", "rb")))
    X_train_counts = count_vect.fit_transform(query)
    feature = X_train_counts.toarray()
    print(np.shape(feature))
    return load_models(feature)


if __name__ == "__main__":
    # query = "Work Schedule Discover employees"
    query = " Benefits Enrollment Deductions in Default Benefits: Day One"
    get_class(query)