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
DATA_PORTION = 8000 # 80,000 Maximum
TRAIN_PORTION = int(4.0 * DATA_PORTION // 5)
SUBMISSION_DATA_PORTION = 200 # 20,000 Maximum

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
model_digits1 = NNModels.CNNmodel("Digits1", 10, LEARNING_RATE, 1200, "Adam")
model_digits2 = NNModels.CNNmodel("Digits2", 10, LEARNING_RATE, 600, "SGD")
model_digits3 = NNModels.CNNmodel("Digits3", 10, LEARNING_RATE, 1300, "RMS")
model_ops1 = NNModels.CNNmodel("Operations1", 3, LEARNING_RATE, 900, "Adam")
model_ops2 = NNModels.CNNmodel("Operations2", 3, LEARNING_RATE, 1000, "SGD")
model_ops3 = NNModels.CNNmodel("Operations3", 3, LEARNING_RATE, 700, "RMS")


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
    out_dig = sess.run(model_digits1.y_out, feed_dict={model_digits1.x: eval_digits[:,1:577], model_digits1.y: eval_labels_digits, model_digits1.keep_prob: 1.0, model_digits1.keep_prob_conv: 1.0})
    print("out_dig:", out_dig[0])
    out_dig2 = sess.run(model_digits2.y_out, feed_dict={model_digits2.x: eval_digits[:,1:577], model_digits2.y: eval_labels_digits, model_digits2.keep_prob: 1.0, model_digits2.keep_prob_conv: 1.0})
    print("out_dig2:", out_dig2[0])
    out_dig += sess.run(model_digits2.y_out, feed_dict={model_digits2.x: eval_digits[:,1:577], model_digits2.y: eval_labels_digits, model_digits2.keep_prob: 1.0, model_digits2.keep_prob_conv: 1.0})
    print("out_dig:", out_dig[0])
    out_dig += sess.run(model_digits3.y_out, feed_dict={model_digits3.x: eval_digits[:,1:577], model_digits3.y: eval_labels_digits, model_digits3.keep_prob: 1.0, model_digits3.keep_prob_conv: 1.0})
    print("out_dig:", out_dig[0])
    out_dig = sess.run(tf.argmax(out_dig, 1))
    print("out_dig:", out_dig)

    out_ops = sess.run(model_ops1.y_out, feed_dict={model_ops1.x: eval_ops[:,1:577], model_ops1.y: eval_labels_ops, model_ops1.keep_prob: 1.0, model_ops1.keep_prob_conv: 1.0})
    out_ops += sess.run(model_ops2.y_out, feed_dict={model_ops2.x: eval_ops[:,1:577], model_ops2.y: eval_labels_ops, model_ops2.keep_prob: 1.0, model_ops2.keep_prob_conv: 1.0})
    out_ops += sess.run(model_ops3.y_out, feed_dict={model_ops3.x: eval_ops[:,1:577], model_ops3.y: eval_labels_ops, model_ops3.keep_prob: 1.0, model_ops3.keep_prob_conv: 1.0})
    out_ops = sess.run(tf.argmax(out_ops, 1))


    labels = sess.run(tf.argmax(eval_labels_digits, 1))
    correct_prediction_digits = sess.run(tf.equal(out_dig, tf.argmax(eval_labels_digits,1)))
    print("accuracy_digits:", correct_prediction_digits)

    accuracy_digits = sess.run(tf.reduce_mean(tf.cast(correct_prediction_digits, tf.float32)))

    correct_prediction_ops = sess.run(tf.equal(out_ops, tf.argmax(eval_labels_ops,1)))
    accuracy_ops = sess.run(tf.reduce_mean(tf.cast(correct_prediction_ops, tf.float32)))
    accuracy_total = (accuracy_digits * train_digits.shape[0] + accuracy_ops * train_ops.shape[0]) / (train_digits.shape[0] + train_ops.shape[0])
    print("Digits Accuracy:",accuracy_digits)
    print("Operations Accuracy:",accuracy_ops)
    print("Final Accuracy:", accuracy_total)


    # acc_dig1 = sess.run(model_digits1.accuracy, feed_dict={model_digits1.x: eval_digits[:,1:577], model_digits1.y: eval_labels_digits, model_digits1.keep_prob: 1.0})
    # acc_dig2 = sess.run(model_digits2.accuracy, feed_dict={model_digits2.x: eval_digits[:,1:577], model_digits2.y: eval_labels_digits, model_digits2.keep_prob: 1.0})
    # acc_dig3 = sess.run(model_digits3.accuracy, feed_dict={model_digits3.x: eval_digits[:,1:577], model_digits3.y: eval_labels_digits, model_digits3.keep_prob: 1.0})
    # acc_ops1 = sess.run(model_ops1.accuracy, feed_dict={model_ops1.x: eval_ops[:,1:577], model_ops1.y: eval_labels_ops, model_ops1.keep_prob: 1.0})
    # acc_ops2 = sess.run(model_ops2.accuracy, feed_dict={model_ops2.x: eval_ops[:,1:577], model_ops2.y: eval_labels_ops, model_ops2.keep_prob: 1.0})
    # acc_ops3 = sess.run(model_ops3.accuracy, feed_dict={model_ops3.x: eval_ops[:,1:577], model_ops3.y: eval_labels_ops, model_ops3.keep_prob: 1.0})
    # acc_total1 = (acc_dig1 * train_digits.shape[0] + acc_ops1 * train_ops.shape[0]) / (train_digits.shape[0] + train_ops.shape[0])
    # acc_total2 = (acc_dig2 * train_digits.shape[0] + acc_ops2 * train_ops.shape[0]) / (train_digits.shape[0] + train_ops.shape[0])
    # acc_total3 = (acc_dig3 * train_digits.shape[0] + acc_ops3 * train_ops.shape[0]) / (train_digits.shape[0] + train_ops.shape[0])
    # print("Digits1 Accuracy:",acc_dig1)
    # print("Operations1 Accuracy:",acc_ops1)
    # print("Final1 Accuracy:", acc_total1)
    # print("Digits2 Accuracy:",acc_dig2)
    # print("Operations2 Accuracy:",acc_ops2)
    # print("Final2 Accuracy:", acc_total2)
    # print("Digits3 Accuracy:",acc_dig3)
    # print("Operations3 Accuracy:",acc_ops3)
    # print("Final3 Accuracy:", acc_total3)


    if SUBMIT:
        print("\n\nRunning submission file through network...")

        # Out variable will be output to file
        out = []

        # Iterate through submission data to test the validity of each equation and store each evaluation in a variable (out)
        # for submission_data_single in submission_data:
        check_tick = time.clock()
        x1 = sess.run(model_digits1.y_out, feed_dict={model_digits1.x: submission_data[:,0], model_digits1.keep_prob: 1.0})
        # print("x1:",x1)
        x1 += sess.run(model_digits2.y_out, feed_dict={model_digits2.x: submission_data[:,0], model_digits2.keep_prob: 1.0})
        # print("x1:",x1)
        x1 += sess.run(model_digits3.y_out, feed_dict={model_digits3.x: submission_data[:,0], model_digits3.keep_prob: 1.0})
        # print("x1:",x1)
        # print("x1.shape:",x1.shape)
        x1 = sess.run(tf.argmax(x1,1))
        # print("x1:",x1)
        print("Finished x1 forward pass.")



        x2 = sess.run(model_digits1.y_out, feed_dict={model_digits1.x: submission_data[:,2], model_digits1.keep_prob: 1.0})
        x2 += sess.run(model_digits2.y_out, feed_dict={model_digits2.x: submission_data[:,2], model_digits2.keep_prob: 1.0})
        x2 += sess.run(model_digits3.y_out, feed_dict={model_digits3.x: submission_data[:,2], model_digits3.keep_prob: 1.0})
        x2 = sess.run(tf.argmax(x2,1))
        print("Finished x2 forward pass.")



        x3 = sess.run(model_digits1.y_out, feed_dict={model_digits1.x: submission_data[:,4], model_digits1.keep_prob: 1.0})
        x3 += sess.run(model_digits2.y_out, feed_dict={model_digits2.x: submission_data[:,4], model_digits2.keep_prob: 1.0})
        x3 += sess.run(model_digits3.y_out, feed_dict={model_digits3.x: submission_data[:,4], model_digits3.keep_prob: 1.0})
        x3 = sess.run(tf.argmax(x3,1))
        print("Finished x3 forward pass.")



        op1 = sess.run(model_ops1.y_out, feed_dict={model_ops1.x: submission_data[:,1], model_ops1.keep_prob: 1.0})
        op1 += sess.run(model_ops2.y_out, feed_dict={model_ops2.x: submission_data[:,1], model_ops2.keep_prob: 1.0})
        op1 += sess.run(model_ops3.y_out, feed_dict={model_ops3.x: submission_data[:,1], model_ops3.keep_prob: 1.0})
        # print("op1.shape:",op1.shape)
        # print("op1:",op1)
        op1 = sess.run(tf.argmax(op1,1))
        print("Finished op1 forward pass.")



        op2 = sess.run(model_ops1.y_out, feed_dict={model_ops1.x: submission_data[:,3], model_ops1.keep_prob: 1.0})
        op2 += sess.run(model_ops2.y_out, feed_dict={model_ops2.x: submission_data[:,3], model_ops2.keep_prob: 1.0})
        op2 += sess.run(model_ops3.y_out, feed_dict={model_ops3.x: submission_data[:,3], model_ops3.keep_prob: 1.0})
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

        datafile_submission = "./submission_files/submission" + time.strftime("%Y-%m-%d--%H:%M:%S") + ".csv"

        with open(datafile_submission, 'w', newline='') as csvfile:
            write = csv.writer(csvfile)
            write.writerow(['index', 'label'])
            for r in out:
                write.writerow(r)
