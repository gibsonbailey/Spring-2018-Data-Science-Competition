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


# Modules
import DataRetriever as DR
import NNModels

# Hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 2
BATCH_SIZE = 100
DATA_PORTION = 80000 # 80,000 Maximum
TRAIN_PORTION = int(4.0 * DATA_PORTION // 5)
SUBMISSION_DATA_PORTION = 20000 # 20,000 Maximum

# Do you want to build a submit file?
SUBMIT = True


# Retrieve data from csv files
# Data kept in train_digits, train_ops and eval_digits, eval_ops
# Labels kept in train_labels and eval_labels
datafile_labels = '../train_labels.csv'
train_labels, eval_labels, train_labels_digits, train_labels_ops, eval_labels_digits, eval_labels_ops = DR.retrieveLabels(datafile_labels, DATA_PORTION, TRAIN_PORTION)

datafile_training = '../train.csv'
train_digits, eval_digits, train_ops, eval_ops = DR.retrieveTrainingData(datafile_training , train_labels, eval_labels, DATA_PORTION, TRAIN_PORTION)

if SUBMIT:
    datafile_submission = '../test.csv'
    submission_data = DR.retrieveSubmissionData(datafile_submission, SUBMISSION_DATA_PORTION)




# Build one-hot arrays for labels
# print("train_labels_digits.size:",train_labels_digits.size)
# print("train_labels_digits.shape[0]:",train_labels_digits.shape[0])
y_train_digits = np.zeros((train_labels_digits.shape[0], 10))
# print("y_train_digits.shape:",y_train_digits.shape)
y_train_digits[range(train_labels_digits.shape[0]), train_labels_digits[:,1]] = 1.0
train_labels_digits = y_train_digits

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
model_digits = NNModels.CNNmodel("Digits", 10, LEARNING_RATE)
model_ops = NNModels.CNNmodel("Operations", 3, LEARNING_RATE)


with tf.Session() as sess:
    # From the NNModels.py file
    sess.run(tf.global_variables_initializer())

    model_digits.train(sess, EPOCHS, BATCH_SIZE, train_digits, train_labels_digits, eval_digits, eval_labels_digits)
    model_ops.train(sess, EPOCHS, BATCH_SIZE, train_ops, train_labels_ops, eval_ops, eval_labels_ops)

    # Training summary
    print("\nTraining Complete.\n")
    acc_dig = sess.run(model_digits.accuracy, feed_dict={model_digits.x: eval_digits[:,1:577], model_digits.y: eval_labels_digits})
    acc_ops = sess.run(model_ops.accuracy, feed_dict={model_ops.x: eval_ops[:,1:577], model_ops.y: eval_labels_ops})
    acc_total = (acc_dig * train_digits.shape[0] + acc_ops * train_ops.shape[0]) / (train_digits.shape[0] + train_ops.shape[0])
    print("Digits Accuracy:",acc_dig)
    print("Operations Accuracy:",acc_ops)
    print("Final Accuracy:", acc_total)



    if SUBMIT:
        print("\n\nRunning submission file through network...")

        # Out variable will be output to file
        out = []

        # Iterate through submission data to test the validity of each equation and store each evaluation in a variable (out)
        # for submission_data_single in submission_data:
        check_tick = time.clock()
        x1 = sess.run(tf.argmax(model_digits.y_out,1), feed_dict={model_digits.x: submission_data[:,0]})
        x2 = sess.run(tf.argmax(model_digits.y_out,1), feed_dict={model_digits.x: submission_data[:,2]})
        x3 = sess.run(tf.argmax(model_digits.y_out,1), feed_dict={model_digits.x: submission_data[:,4]})
        op1 = sess.run(tf.argmax(model_ops.y_out,1), feed_dict={model_ops.x: submission_data[:,1]})
        op2 = sess.run(tf.argmax(model_ops.y_out,1), feed_dict={model_ops.x: submission_data[:,3]})
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

        datafile_submission = "./submission_files/submission" + time.strftime("%Y-%m-%d--%H:%M:%S") + ".csv"

        with open(datafile_submission, 'w', newline='') as csvfile:
            write = csv.writer(csvfile)
            write.writerow(['index', 'label'])
            for r in out:
                write.writerow(r)
