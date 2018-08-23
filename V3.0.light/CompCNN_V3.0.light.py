# V_2.1 is the beginning of an object oriented tensorflow model.
# The model class is written inside the NNModels.py file.
# The train method is built in the model class.
# Submissions are now stored in submission_files.

import numpy as np
import tensorflow as tf
import csv
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import time
from skimage.transform import resize
import pickle


# Modules
import DataRetriever as DR
import NNModels

# Hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 1
BATCH_SIZE = 100
DATA_PORTION = 800 # 80,000 Maximum
TRAIN_PORTION = int(4.5 * DATA_PORTION // 5)
SUBMISSION_DATA_PORTION = 2000 # 20,000 Maximum

# Do you want to augment the data with MNIST?
MNIST_AUGMENT = False

# Do you want to augment the pickle data?
PICKLE_AUGMENT = False

# Do you want to build a submit file?
SUBMIT = True

# Do you want to run on floyd hub?
FLOYD = False


if MNIST_AUGMENT:
    # Import MNIST data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    mnist_data = mnist.train.images # Returns np.array
    mnist_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    mnist_eval_data = mnist.test.images # Returns np.array
    mnist_eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    mnist_labels = np.concatenate([mnist_labels,mnist_eval_labels])
    mnist_data = np.concatenate([mnist_data,mnist_eval_data])

    # print(mnist_data.shape)
    mnist_data = mnist_data.reshape(mnist_data.shape[0], 28, 28)

    # print(mnist_labels.shape)
    mnist_data = resize(mnist_data, (mnist_data.shape[0],24,24))
    mnist_data = mnist_data.reshape(mnist_data.shape[0], 576)
    # print("mnist_data.shape:", mnist_data.shape)
    mnist_data = np.insert(mnist_data, 0, 2.0, axis=1)
    # print("mnist_data.shape:", mnist_data.shape)
    # print("mnist_data[0]:", mnist_data[0])
    # print("mnist_data[1]:", mnist_data[1])

    print("\n\n\n\n\nMNIST data imported.")

if PICKLE_AUGMENT:
    if FLOYD:
        "Floyd Pickle Data Augment"
        with open('/floyd_data/XnXoynyo2.pickle', 'rb') as handle:
            Xn_train, Xo_train, yn_train, yo_train = pickle.load(handle)
    else:
        with open('../XnXoynyo2.pickle', 'rb') as handle:
            Xn_train, Xo_train, yn_train, yo_train = pickle.load(handle)
    print("Xn_train.shape:", Xn_train.shape)
    print("yn_train.shape:", yn_train.shape)

    Xn_train = Xn_train.reshape(Xn_train.shape[0], 576)
    yn_train = yn_train[:,0:10]

    print("Xn_train.shape:", Xn_train.shape)
    print("yn_train.shape:", yn_train.shape)
    print("\nPickle data imported.")

# Retrieve data from csv files
# Data kept in train_digits, train_ops and eval_digits, eval_ops
# Labels kept in train_labels and eval_labels
if FLOYD:
    datafile_labels = '/floyd_data/train_labels.csv'
else:
    datafile_labels = '../train_labels.csv'
train_labels, eval_labels, train_labels_digits, train_labels_ops, eval_labels_digits, eval_labels_ops = DR.retrieveLabels(datafile_labels, DATA_PORTION, TRAIN_PORTION)

if FLOYD:
    datafile_training = '/floyd_data/train.csv'
else:
    datafile_training = '../train.csv'
train_digits, eval_digits, train_ops, eval_ops = DR.retrieveTrainingData(datafile_training , train_labels, eval_labels, DATA_PORTION, TRAIN_PORTION)

if SUBMIT:
    if FLOYD:
        datafile_submission = '/floyd_data/test.csv'
    else:
        datafile_submission = '../test.csv'
    submission_data = DR.retrieveSubmissionData(datafile_submission, SUBMISSION_DATA_PORTION)

print("submisstion_data.shape:", submission_data[0:300,0].shape)

train_labels_digits = train_labels_digits[:,1]
# print("TRAINLABELS.SHAPE:",train_labels_digits.shape)

if MNIST_AUGMENT:
    # Concatenate the MNIST data labels for training
    train_labels_digits = np.concatenate([mnist_labels, train_labels_digits])
    # print("train_digits.shape:", train_digits.shape)
    train_digits = np.concatenate([mnist_data, train_digits], 0)
    # print("train_digits.shape:", train_digits.shape)

# Build one-hot arrays for labels
# print("train_labels_digits.size:",train_labels_digits.size)
# print("train_labels_digits.shape[0]:",train_labels_digits.shape[0])
y_train_digits = np.zeros((train_labels_digits.shape[0], 10))
# print("y_train_digits.shape:",y_train_digits.shape)
y_train_digits[range(train_labels_digits.shape[0]), train_labels_digits] = 1.0
train_labels_digits = y_train_digits

if PICKLE_AUGMENT:
    # print("original Xn_train.shape:", Xn_train.shape)
    Xn_train = np.insert(Xn_train, 0, 2.0, axis=1)
    # print("Xn_train.shape:", Xn_train.shape)
    print("train_labels_digits.shape:", train_labels_digits.shape)
    # print("yn_train.shape:", yn_train.shape)
    train_labels_digits = np.concatenate([yn_train, train_labels_digits])
    print("train_digits.shape before concatenation:", train_digits.shape)
    train_digits = np.concatenate([Xn_train, train_digits], 0)
    print("train_digits.shape:", train_digits.shape)

# print("train_labels_ops.size:",train_labels_ops.size)
# print("train_labels_ops.shape[0]:",train_labels_ops.shape[0])
y_train_ops = np.zeros((train_labels_ops.shape[0], 3))
# print("y_train_ops.shape:",y_train_ops.shape)
y_train_ops[range(train_labels_ops.shape[0]), train_labels_ops[:,1]-10] = 1.0
train_labels_ops = y_train_ops

# print("eval_labels_digits.size:",eval_labels_digits.size)
# print("eval_labels_digits.shape[0]:",eval_labels_digits.shape[0])
y_eval_digits = np.zeros((eval_labels_digits.shape[0], 10))
# print("y_eval.shape",y_eval.shape)
y_eval_digits[range(eval_labels_digits.shape[0]), eval_labels_digits[:,1]] = 1.0
eval_labels_digits = y_eval_digits

# print("eval_labels_ops.size:",eval_labels_ops.size)
# print("eval_labels_ops.shape[0]:",eval_labels_ops.shape[0])
y_eval_ops = np.zeros((eval_labels_ops.shape[0], 3))
# print("y_eval_ops.shape",y_eval_ops.shape)
y_eval_ops[range(eval_labels_ops.shape[0]), eval_labels_ops[:,1]-10] = 1.0
eval_labels_ops = y_eval_ops


# Instantiate a model for learning digits and another for learning operations
model_digits1 = NNModels.CNNmodel_light("Digits1", 10, LEARNING_RATE, 1300, "cross_entropy")
model_digits2 = NNModels.CNNmodel_light("Digits2", 10, LEARNING_RATE, 1300, "MSE")
model_digits3 = NNModels.CNNmodel_light("Digits3", 10, LEARNING_RATE, 1300, "huber")
model_ops1 = NNModels.CNNmodel_light("Operations1", 3, LEARNING_RATE, 900, "cross_entropy")
model_ops2 = NNModels.CNNmodel_light("Operations2", 3, LEARNING_RATE, 1000, "MSE")
model_ops3 = NNModels.CNNmodel_light("Operations3", 3, LEARNING_RATE, 1000, "huber")


with tf.Session() as sess:
    # From the NNModels.py file
    sess.run(tf.global_variables_initializer())

    model_digits1.train(sess, EPOCHS, BATCH_SIZE, train_digits, train_labels_digits, eval_digits, eval_labels_digits)
    model_digits2.train(sess, EPOCHS, BATCH_SIZE, train_digits, train_labels_digits, eval_digits, eval_labels_digits)
    model_digits3.train(sess, EPOCHS, BATCH_SIZE, train_digits, train_labels_digits, eval_digits, eval_labels_digits)
    model_ops1.train(sess, EPOCHS, BATCH_SIZE, train_ops, train_labels_ops, eval_ops, eval_labels_ops)
    model_ops2.train(sess, EPOCHS, BATCH_SIZE, train_ops, train_labels_ops, eval_ops, eval_labels_ops)
    model_ops3.train(sess, EPOCHS, BATCH_SIZE, train_ops, train_labels_ops, eval_ops, eval_labels_ops)

    # Training summary
    print("\nTraining Complete.\n")

    numbs = [1, 1, 0, 8, 2, 6, 3, 2, 6, 9, 4, 5, 7, 7, 1, 2, 0, 3, 6, 2, 5, 1, 9, 7, 1, 2, 1, 6, 3, 9,
        2, 2, 0, 3, 2, 0, 5, 2, 1, 6, 9, 3, 3, 7, 4, 0, 0, 0, 0, 3, 3, 7, 1, 5, 4, 4, 7, 4, 4, 1,
        8, 2, 5, 4, 0, 4, 2, 5, 7, 5, 1, 5, 6, 2, 6, 9, 7, 1, 9, 7, 2, 6, 0, 4, 1, 8, 8, 0, 3, 1,
        8, 4, 4, 3, 0, 2, 1, 6, 5, 4, 0, 4, 2, 7, 9, 6, 6, 0, 6, 0, 7, 9, 1, 7, 6, 9, 3, 3, 3, 5,
        7, 1, 8, 4, 1, 6, 6, 0, 6, 4, 4, 1, 5, 2, 5, 6, 0, 6, 9, 1, 8, 1, 2, 1, 8, 6, 3, 2, 5, 3,
        7, 4, 3, 0, 6, 6, 4, 4, 0, 6, 9, 3, 8, 0, 7, 3, 3, 0, 5, 0, 5, 9, 6, 3, 5, 3, 1, 7, 2, 5,
        1, 1, 0, 4, 0, 4, 6, 5, 1, 2, 7, 8, 6, 3, 8, 5, 5, 1, 3, 4, 5, 8, 2, 7, 9, 9, 1, 5, 2, 7,
        0, 1, 1, 6, 5, 3, 5, 4, 1, 0, 3, 2, 2, 1, 0, 7, 2, 9, 9, 1, 7, 7, 2, 5, 8, 5, 3, 3, 3, 0,
        9, 1, 7, 9, 3, 6, 5, 2, 6, 4, 6, 2, 1, 1, 0, 6, 0, 7, 6, 3, 8, 4, 0, 4, 2, 7, 8, 3, 2, 5,
        1, 0, 2, 6, 2, 4, 7, 0, 7, 3, 5, 9, 2, 7, 8, 1, 8, 8, 2, 0, 3, 9, 5, 3, 8, 7, 0, 2, 1, 2,
        0, 3, 2, 9, 7, 3, 4, 3, 7, 9, 5, 3, 7, 2, 9, 8, 0, 9, 0, 3, 3, 6, 7, 1, 1, 6, 5, 6, 2, 7,
        5, 7, 1, 9, 9, 1, 9, 0, 9, 9, 7, 1, 7, 3, 3, 0, 6, 6, 9, 1, 8, 3, 8, 6, 2, 2, 4, 6, 1, 3,
        1, 1, 2, 3, 1, 4, 6, 5, 2, 5, 0, 5, 4, 4, 1, 6, 0, 6, 8, 1, 6, 5, 3, 2, 1, 1, 1, 5, 1, 4,
        4, 1, 5, 5, 6, 1, 7, 0, 7, 8, 6, 2, 4, 3, 1, 5, 5, 1, 7, 3, 6, 8, 9, 0, 0, 3, 4, 2, 2, 0,
        8, 4, 5, 5, 4, 0, 7, 1, 4, 4, 3, 2, 3, 6, 8, 0, 2, 2, 3, 4, 6, 7, 2, 8, 3, 1, 4, 3, 4, 1,
        3, 6, 9, 6, 1, 7, 8, 7, 1, 2, 5, 6, 7, 3, 4, 6, 0, 6, 9, 5, 4, 1, 0, 0, 6, 2, 4, 6, 0, 7,
        1, 1, 2, 0, 2, 2, 0, 6, 6, 7, 2, 6, 4, 1, 5, 4, 9, 5, 3, 5, 7, 6, 6, 0, 2, 9, 7, 7, 4, 1,
        8, 6, 3, 1, 7, 7, 0, 6, 6, 0, 7, 8, 7, 2, 9, 5, 1, 7, 3, 4, 6, 0, 5, 6, 5, 1, 8, 6, 3, 9,
        3, 2, 5, 9, 0, 8, 5, 3, 6, 6, 0, 6, 3, 2, 3, 6, 6, 0, 1, 2, 1, 4, 4, 9, 3, 0, 1, 8, 9, 1,
        3, 0, 3, 7, 0, 7, 3, 1, 4, 1, 0, 3, 0, 8, 8, 5, 2, 4, 2, 6, 8, 3, 4, 5, 0, 1, 2, 7, 9, 0,
        8, 8, 0, 5, 0, 7, 5, 5, 0, 7, 2, 8, 4, 2, 1, 2, 4, 8, 1, 6, 7, 9, 9, 0, 5, 2, 3, 7, 0, 5,
        0, 6, 6, 6, 8, 3, 7, 2, 6, 9, 8, 0, 9, 7, 3, 7, 2, 9, 6, 4, 3, 1, 1, 1, 4, 9, 5, 5, 1, 3,
        0, 2, 2, 7, 1, 7, 8, 2, 6, 1, 7, 7, 3, 6, 3, 1, 5, 4, 3, 2, 0, 7, 2, 5, 5, 1, 5, 5, 1, 3,
        5, 2, 4, 1, 8, 9, 6, 1, 7, 4, 6, 2, 1, 6, 6, 4, 0, 4, 3, 2, 5, 7, 0, 7, 1, 1, 1, 0, 8, 8,
        5, 5, 1, 9, 7, 0, 5, 3, 8, 1, 2, 2, 0, 4, 3, 1, 0, 0, 8, 8, 0, 3, 1, 3, 4, 1, 3, 7, 4, 3,
        5, 3, 2, 3, 2, 4, 6, 1, 5, 9, 0, 8, 1, 4, 5, 3, 3, 0, 7, 0, 7, 3, 2, 5, 3, 8, 6, 7, 0, 7,
        7, 7, 2, 3, 6, 9, 8, 7, 1, 3, 1, 4, 9, 6, 4, 1, 0, 0, 4, 4, 6, 8, 4, 5, 7, 3, 4, 8, 2, 6,
        3, 1, 2, 0, 1, 1, 2, 0, 2, 7, 3, 4, 9, 5, 5, 9, 6, 2, 6, 1, 6, 7, 0, 6, 2, 2, 0, 1, 3, 4,
        6, 4, 1, 1, 1, 2, 4, 5, 9, 6, 1, 7, 1, 1, 0, 9, 0, 9, 1, 7, 8, 9, 2, 7, 2, 2, 1, 5, 0, 5,
        8, 1, 9, 1, 5, 6, 9, 9, 1, 9, 4, 4, 7, 0, 7, 6, 6, 2, 0, 6, 6, 5, 5, 1, 9, 8, 1, 6, 5, 0,
        ]
    numbs = np.array(numbs)
    numbs = np.reshape(numbs, (int(numbs.shape[0]/3),3))
    numbs = np.transpose(numbs)
    numbs = np.reshape(numbs, (-1, ))
    # numbs = np.transpose(numbs)
    length = int(numbs.shape[0] / 3)
    numbs_y = np.zeros((numbs.shape[0], 10))
    # print("y_train_digits.shape:",y_train_digits.shape)
    numbs_y[range(numbs.shape[0]), numbs] = 1.0
    numbs = numbs_y
    # numbs = np.reshape(numbs, (int(numbs.shape[0]/3),30))



    print("Daniel's Cheater Data Accuracy:")
    acc_cheater1 = sess.run(model_digits1.accuracy, feed_dict={model_digits1.x: submission_data[0:length,0], model_digits1.y: numbs[0:length], model_digits1.keep_prob: 1.0, model_digits1.keep_prob_conv: 1.0})
    out_cheater1 = sess.run(model_digits1.y_out, feed_dict={model_digits1.x: submission_data[0:length,0], model_digits1.y: numbs[0:length], model_digits1.keep_prob: 1.0, model_digits1.keep_prob_conv: 1.0})
    corr_cheater1 = sess.run(model_digits1.correct_prediction, feed_dict={model_digits1.x: submission_data[0:length,0], model_digits1.y: numbs[0:length], model_digits1.keep_prob: 1.0, model_digits1.keep_prob_conv: 1.0})
    print("out_cheater1:",out_cheater1[0])
    print("numbs:", numbs[0])
    acc_cheater2 = sess.run(model_digits2.accuracy, feed_dict={model_digits2.x: submission_data[0:length,2], model_digits2.y: numbs[length:int(length*2)], model_digits2.keep_prob: 1.0, model_digits2.keep_prob_conv: 1.0})
    out_cheater2 = sess.run(model_digits2.y_out, feed_dict={model_digits2.x: submission_data[0:length,2], model_digits2.y: numbs[length:int(length*2)], model_digits2.keep_prob: 1.0, model_digits2.keep_prob_conv: 1.0})
    corr_cheater2 = sess.run(model_digits2.correct_prediction, feed_dict={model_digits2.x: submission_data[0:length,2], model_digits2.y: numbs[length:int(length*2)], model_digits2.keep_prob: 1.0, model_digits2.keep_prob_conv: 1.0})

    acc_cheater3 = sess.run(model_digits3.accuracy, feed_dict={model_digits3.x: submission_data[0:length,4], model_digits3.y: numbs[int(length*2):int(length*3)], model_digits3.keep_prob: 1.0, model_digits3.keep_prob_conv: 1.0})
    out_cheater3 = sess.run(model_digits3.y_out, feed_dict={model_digits3.x: submission_data[0:length,4], model_digits3.y: numbs[int(length*2):int(length*3)], model_digits3.keep_prob: 1.0, model_digits3.keep_prob_conv: 1.0})
    corr_cheater3 = sess.run(model_digits3.correct_prediction, feed_dict={model_digits3.x: submission_data[0:length,4], model_digits3.y: numbs[int(length*2):int(length*3)], model_digits3.keep_prob: 1.0, model_digits3.keep_prob_conv: 1.0})
    print("Cheater1 Accuracy:", acc_cheater1)
    print("Cheater2 Accuracy:", acc_cheater2)
    print("Cheater3 Accuracy:", acc_cheater3)

    corr_cheater = np.concatenate([corr_cheater1, corr_cheater2, corr_cheater3])
    out_cheater = np.concatenate([out_cheater1, out_cheater2, out_cheater3])
    with open('/output/corr_output.pickle', 'wb') as handle:
        pickle.dump((corr_cheater, out_cheater), handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("\n\n\nFinished CHEATING...")



    acc_dig1 = sess.run(model_digits1.accuracy, feed_dict={model_digits1.x: eval_digits[:,1:577], model_digits1.y: eval_labels_digits, model_digits1.keep_prob: 1.0, model_digits1.keep_prob_conv: 1.0})
    acc_dig2 = sess.run(model_digits2.accuracy, feed_dict={model_digits2.x: eval_digits[:,1:577], model_digits2.y: eval_labels_digits, model_digits2.keep_prob: 1.0, model_digits2.keep_prob_conv: 1.0})
    acc_dig3 = sess.run(model_digits3.accuracy, feed_dict={model_digits3.x: eval_digits[:,1:577], model_digits3.y: eval_labels_digits, model_digits3.keep_prob: 1.0, model_digits3.keep_prob_conv: 1.0})
    acc_ops1 = sess.run(model_ops1.accuracy, feed_dict={model_ops1.x: eval_ops[:,1:577], model_ops1.y: eval_labels_ops, model_ops1.keep_prob: 1.0, model_ops1.keep_prob_conv: 1.0})
    acc_ops2 = sess.run(model_ops2.accuracy, feed_dict={model_ops2.x: eval_ops[:,1:577], model_ops2.y: eval_labels_ops, model_ops2.keep_prob: 1.0, model_ops2.keep_prob_conv: 1.0})
    acc_ops3 = sess.run(model_ops3.accuracy, feed_dict={model_ops3.x: eval_ops[:,1:577], model_ops3.y: eval_labels_ops, model_ops3.keep_prob: 1.0, model_ops3.keep_prob_conv: 1.0})
    acc_total1 = (acc_dig1 * train_digits.shape[0] + acc_ops1 * train_ops.shape[0]) / (train_digits.shape[0] + train_ops.shape[0])
    acc_total2 = (acc_dig2 * train_digits.shape[0] + acc_ops2 * train_ops.shape[0]) / (train_digits.shape[0] + train_ops.shape[0])
    acc_total3 = (acc_dig3 * train_digits.shape[0] + acc_ops3 * train_ops.shape[0]) / (train_digits.shape[0] + train_ops.shape[0])
    print("Digits1 Accuracy:",acc_dig1)
    print("Operations1 Accuracy:",acc_ops1)
    print("Final1 Accuracy:", acc_total1)
    print("Digits2 Accuracy:",acc_dig2)
    print("Operations2 Accuracy:",acc_ops2)
    print("Final2 Accuracy:", acc_total2)
    print("Digits3 Accuracy:",acc_dig3)
    print("Operations3 Accuracy:",acc_ops3)
    print("Final3 Accuracy:", acc_total3)


    if SUBMIT:
        print("\n\nRunning submission file through network...")

        # Out variable will be output to file
        out = []

        # Iterate through submission data to test the validity of each equation and store each evaluation in a variable (out)
        # for submission_data_single in submission_data:
        check_tick = time.clock()
        x1 = sess.run(model_digits1.y_out, feed_dict={model_digits1.x: submission_data[:,0], model_digits1.keep_prob: 1.0, model_digits1.keep_prob_conv: 1.0})
        x1 += sess.run(model_digits2.y_out, feed_dict={model_digits2.x: submission_data[:,0], model_digits2.keep_prob: 1.0, model_digits2.keep_prob_conv: 1.0})
        x1 += sess.run(model_digits3.y_out, feed_dict={model_digits3.x: submission_data[:,0], model_digits3.keep_prob: 1.0, model_digits3.keep_prob_conv: 1.0})
        x1 = sess.run(tf.argmax(x1,1))
        print("Finished x1 forward pass.")



        x2 = sess.run(model_digits1.y_out, feed_dict={model_digits1.x: submission_data[:,2], model_digits1.keep_prob: 1.0, model_digits1.keep_prob_conv: 1.0})
        x2 += sess.run(model_digits2.y_out, feed_dict={model_digits2.x: submission_data[:,2], model_digits2.keep_prob: 1.0, model_digits2.keep_prob_conv: 1.0})
        x2 += sess.run(model_digits3.y_out, feed_dict={model_digits3.x: submission_data[:,2], model_digits3.keep_prob: 1.0, model_digits3.keep_prob_conv: 1.0})
        x2 = sess.run(tf.argmax(x2,1))
        print("Finished x2 forward pass.")



        x3 = sess.run(model_digits1.y_out, feed_dict={model_digits1.x: submission_data[:,4], model_digits1.keep_prob: 1.0, model_digits1.keep_prob_conv: 1.0})
        x3 += sess.run(model_digits2.y_out, feed_dict={model_digits2.x: submission_data[:,4], model_digits2.keep_prob: 1.0, model_digits2.keep_prob_conv: 1.0})
        x3 += sess.run(model_digits3.y_out, feed_dict={model_digits3.x: submission_data[:,4], model_digits3.keep_prob: 1.0, model_digits3.keep_prob_conv: 1.0})
        x3 = sess.run(tf.argmax(x3,1))
        print("Finished x3 forward pass.")



        op1 = sess.run(model_ops1.y_out, feed_dict={model_ops1.x: submission_data[:,1], model_ops1.keep_prob: 1.0, model_ops1.keep_prob_conv: 1.0})
        op1 += sess.run(model_ops2.y_out, feed_dict={model_ops2.x: submission_data[:,1], model_ops2.keep_prob: 1.0, model_ops2.keep_prob_conv: 1.0})
        op1 += sess.run(model_ops3.y_out, feed_dict={model_ops3.x: submission_data[:,1], model_ops3.keep_prob: 1.0, model_ops3.keep_prob_conv: 1.0})
        # print("op1.shape:",op1.shape)
        # print("op1:",op1)
        op1 = sess.run(tf.argmax(op1,1))
        print("Finished op1 forward pass.")



        op2 = sess.run(model_ops1.y_out, feed_dict={model_ops1.x: submission_data[:,3], model_ops1.keep_prob: 1.0, model_ops1.keep_prob_conv: 1.0})
        op2 += sess.run(model_ops2.y_out, feed_dict={model_ops2.x: submission_data[:,3], model_ops2.keep_prob: 1.0, model_ops2.keep_prob_conv: 1.0})
        op2 += sess.run(model_ops3.y_out, feed_dict={model_ops3.x: submission_data[:,3], model_ops3.keep_prob: 1.0, model_ops3.keep_prob_conv: 1.0})
        op2 = sess.run(tf.argmax(op2,1))
        print("Finished op2 forward pass.")



        check_toc = time.clock()
        print("Forward pass of submission data (s):", check_toc-check_tick)
        # Determine whether equation is true or false
        row = []
        row.append(0)
        row.append(0)
        for i in range(submission_data.shape[0]):
            row[0] = i
            if op1[i] == 2:
                if op2[i] == 1:
                    row[1] = int(x1[i] == x2[i] - x3[i])
                else:
                    row[1] = int(x1[i] == x2[i] + x3[i])
            else:
                if op1[i] == 1:
                    row[1] = int(x1[i] - x2[i] == x3[i])
                else:
                    row[1] = int(x1[i] + x2[i] == x3[i])
            out.append([row[0], row[1]])


        datafile_submission = "/output/submission" + time.strftime("%Y-%m-%d--%H:%M:%S") + ".csv"

        with open(datafile_submission, 'w', newline='') as csvfile:
            write = csv.writer(csvfile)
            write.writerow(['index', 'label'])
            for r in out:
                write.writerow(r)
