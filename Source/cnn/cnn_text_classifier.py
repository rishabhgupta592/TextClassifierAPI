import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
import _pickle as pickle
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
# CNN for text data
tf.logging.set_verbosity(tf.logging.INFO)


def feature_extractor(data):
    """ Convert text into numerical features"""
    print("Feature extraction started ...")
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(data)
    # joblib.dump(count_vect.vocabulary_, 'feature.pkl')
    pickle.dump(count_vect.vocabulary_, open("cnn_feature.pkl","wb"))
    print("Feature vector size = ",np.shape(X_train_counts))
    print("Feature extraction Done !")
    return X_train_counts.toarray()


def cnn_model_fn(features, labels, mode):
    print("Inside CNN")
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel

    # -1 for batch size, which specifies that this dimension should be dynamically
    # computed based on the number of input values in
    input_layer = tf.reshape(features["x"], [-1, 30, 30, 1], name="input_x")

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1,name="output_class"),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {'predict_output': tf.estimator.export.PredictOutput(predictions)}
        predictions_dict = {"predicted": predictions}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,export_outputs=export_outputs)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Pre-process Data and conver it into training data
    shuffled_data = pd.read_csv('./../Data/text_classification_data3.csv')
    # shuffled_data = shuffle(data) # Shuffling data set

    text_data = shuffled_data['Que']
    intent = shuffled_data['I1']
    # intent = shuffled_data.iloc[:, 2:4]
    # intent = shuffled_data.iloc[:,1:]
    train_features = feature_extractor(text_data)


    X_train, X_test, y_train, y_test = train_test_split(train_features, intent, test_size = 0.33)

    # import sys
    # sys.exit(0)
    # zero_numpy = np.zeros(30*30)
    # zero_numpy[:]
    train_features = np.pad(X_train, ((0,0), (0,73)), 'constant')
    train_features = train_features.astype('float16')

    X_test = np.pad(X_test, ((0,0), (0,73)), 'constant')
    X_test = X_test.astype('float16')
    # y_train = y_train.astype('float16')
    print("check1")
    # Create the Estimator
    model_path = './../models/cnn/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_path+"mnist_convnet_model")
    print("check2")
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1)
    print("check3")
    # Train the model
    train_data = train_features
    # print(type(train_data))
    train_labels = y_train.as_matrix()
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1,
        hooks=[logging_hook])
    print("check4")
    eval_data = X_test
    y_test = y_test.as_matrix()
    eval_labels = y_test
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    # save_model(mnist_classifier, model_path)
    print(eval_results)


# saving models for later/runtime use.
# https://www.tensorflow.org/programmers_guide/saved_model
# Using SavedModel with Estimators

def save_model(classifier, model_path ):
    print("Saving models ...")
    classifier.export_savedmodel(model_path,serving_input_receiver_fn=serving_input_receiver_fn)
    pass


""" During training, an input_fn() ingests data and prepares it for use by the model. At serving time, similarly, a serving_input_receiver_fn() accepts
 inference requests and prepares them for the model.
 
 This function has the following purposes:

 $ To add placeholders to the graph that the serving system will feed with inference requests.
 $ To add any additional ops needed to convert data from the input format into the feature Tensors expected by the model.
"""

def serving_input_receiver_fn():
    serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[None], name='input_tensors')
    receiver_tensors = {"predictor_inputs": serialized_tf_example}
    feature_spec = {"x": tf.FixedLenFeature([827], tf.float32)}
    # feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    features = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

if __name__ == "__main__":
    # main('unused_argv')
    tf.app.run()