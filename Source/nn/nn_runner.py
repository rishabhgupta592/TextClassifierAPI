import tensorflow as tf
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
import _pickle as pickle

model_path = './../models/nn/'

def load_models(feature):
    # saver = tf.train.import_meta_graph('./models/text_classifier/model.meta')
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(model_path + '/model.ckpt.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(model_path))
        graph = tf.get_default_graph()
        # X = tf.placeholder("float", [None, 861], name='X')
        X = graph.get_tensor_by_name("X:0")
        prediction = graph.get_tensor_by_name("prediction:0")
        print(sess.run(prediction, feed_dict={X:feature}))
        return sess.run(prediction, feed_dict={X:feature})


def get_class(query):
    query = [query]
    # from sklearn.feature_extraction.text import CountVectorizer
    # count_vect = joblib.load('feature.pkl')
    count_vect = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(model_path+"/feature.pkl", "rb")))
    X_train_counts = count_vect.fit_transform(query)
    feature = X_train_counts.toarray()
    # print(feature)
    return load_models(feature)


if __name__ == "__main__":
    query = "Work Schedule Discover employees"
    get_class(query)