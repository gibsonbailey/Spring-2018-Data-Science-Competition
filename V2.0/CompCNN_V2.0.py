# V_2 includes a separation of the training data into digits and operators.
# This data collection happens in new file called DataRetriever.py
# The network is copied, so the original can train on digits only, and the copy
# can train on the operators only.

import numpy as np
import tensorflow as tf
import csv
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import time


# Modules
import DataRetriever as DR

# Hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 2
BATCH_SIZE = 100
DATA_PORTION = 80000 # 80,000 Maximum
TRAIN_PORTION = int(4.99 * DATA_PORTION // 5)
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
# print("y_eval.shape",y_eval.shape)
y_eval_ops[range(eval_labels_ops.shape[0]), eval_labels_ops[:,1]-10] = 1.0
eval_labels_ops = y_eval_ops


# Input placeholder
x = tf.placeholder(tf.float32, [None,576])
# Reshape for convolution
x_shaped = tf.reshape(x, [-1,24,24,1])

# Output placeholder
y = tf.placeholder(tf.float32, [None, 10])
y_ops = tf.placeholder(tf.float32, [None, 3])

def convolution_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    conv_filter_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

    # Initilization of filter weights and biases
    weights = tf.Variable(tf.truncated_normal(conv_filter_shape, stddev=0.03), name=name+'_W')
    biases = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    out_layer = tf.nn.conv2d(input_data, weights, [1,1,1,1], padding='SAME')
    out_layer += biases
    out_layer = tf.nn.relu(out_layer)

    kernal_size = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1,2,2,1]
    out_layer = tf.nn.max_pool(out_layer, ksize=kernal_size, strides=strides, padding='SAME')

    return out_layer


# Model definition (digits)
layer1 = convolution_layer(x_shaped, 1, 32, [5,5], [2,2], name='layer1')
layer2 = convolution_layer(layer1, 32, 64, [5,5], [2,2], name='layer2')
# Ouput shape is 6 x 6

# Reshaping for dense layers
flattened = tf.reshape(layer2, [-1,6*6*64])

# Dense layer 1 definition
wd1 = tf.Variable(tf.truncated_normal([6*6*64, 1000], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)
# Output shape: 1 x 1000

# Dense layer 2 definition
wd2 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([10], stddev=0.01), name='bd2')
dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
y_ = tf.nn.softmax(dense_layer2)

# Cost function declaration (cross-entropy)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense_layer2, labels=y))

# Set up optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)

# Define accuracy assessment
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# Model definition (ops)
layer1op = convolution_layer(x_shaped, 1, 32, [5,5], [2,2], name='layer1')
layer2op = convolution_layer(layer1op, 32, 64, [5,5], [2,2], name='layer2')
# Ouput shape is 6 x 6

# Reshaping for dense layers
flattenedop = tf.reshape(layer2op, [-1,6*6*64])

# Dense layer 1 definition
wd1op = tf.Variable(tf.truncated_normal([6*6*64, 1000], stddev=0.03), name='wd1')
bd1op = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
dense_layer1op = tf.matmul(flattenedop, wd1op) + bd1op
dense_layer1op = tf.nn.relu(dense_layer1op)
# Output shape: 1 x 1000

# Dense layer 2 definition
wd2op = tf.Variable(tf.truncated_normal([1000, 3], stddev=0.03), name='wd2')
bd2op = tf.Variable(tf.truncated_normal([3], stddev=0.01), name='bd2')
dense_layer2op = tf.matmul(dense_layer1op, wd2op) + bd2op
y_op = tf.nn.softmax(dense_layer2op)

# Cost function declaration (cross-entropy)
cross_entropyop = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense_layer2op, labels=y_ops))

# Set up optimizer
optimizerop = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropyop)

# Define accuracy assessment
correct_predictionop = tf.equal(tf.argmax(y_ops,1), tf.argmax(y_op,1))
accuracyop = tf.reduce_mean(tf.cast(correct_predictionop, tf.float32))

# Initialization operator
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    total_batch_num_digits = int(train_digits.shape[0] / BATCH_SIZE)
    total_batch_num_ops = int(train_ops.shape[0] / BATCH_SIZE)

    for epoch in range(EPOCHS):
        print("\n\n\nStarting epoch", epoch+1)

        avg_cost = 0
        batch_low_ind = 0
        batch_high_ind = batch_low_ind + BATCH_SIZE

        # Shuffle training data
        shuff_digits = list(range(train_digits.shape[0]))
        np.random.shuffle(shuff_digits)
        train_digits = train_digits[shuff_digits]
        train_labels_digits = train_labels_digits[shuff_digits]

        shuff_ops = list(range(train_ops.shape[0]))
        np.random.shuffle(shuff_ops)
        train_ops = train_ops[shuff_ops]
        train_labels_ops = train_labels_ops[shuff_ops]


        print("\n")
        print("Beginning digits training...")
        print("\nTotal number of digits batches to be run:", total_batch_num_digits)
        print("\n")
        for i in range(total_batch_num_digits):
            if((i + 1) % 60 == 0 or i == 0):
                print("Current Digits Batch Number:", i+1)
            # Extract batch interval from total dataset
            batch_x = train_digits[batch_low_ind:batch_high_ind,1:train_digits.shape[1]]
            batch_y = train_labels_digits[batch_low_ind:batch_high_ind]
            batch_low_ind += BATCH_SIZE
            batch_high_ind += BATCH_SIZE
            _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch_num_digits
            # Uncomment below for output example on first image
            # if i % (total_batch_num // 5) == 0:
            #     network_out = sess.run(y_, feed_dict={x: eval_digits[0:1], y: eval_labels_digits[0:1]})
            #     label_ex = eval_labels_digits[0]
            #     print("Ex. network output:", network_out)
            #     print("Corresponding label:", label_ex)

        print("Testing accuracy of digits network...")
        test_acc = sess.run(accuracy, feed_dict={x: eval_digits[:,1:577], y: eval_labels_digits})

        print("Epoch", (epoch+1), "cost =", "{:.3f}".format(avg_cost), " digits test accuracy: ", "{:.3f}".format(test_acc))
        print("\n\n")


        print("Beginning operations training...")
        print("\nTotal number of ops batches to be run:", total_batch_num_ops)
        print("\n")
        avg_cost_ops = 0
        batch_low_ind = 0
        batch_high_ind = batch_low_ind + BATCH_SIZE
        for i in range(total_batch_num_ops):
            if((i + 1) % 20 == 0 or i == 0):
                print("Current Operations Batch Number:", i+1)
            # Extract batch interval from total dataset
            batch_x = train_ops[batch_low_ind:batch_high_ind,1:train_ops.shape[1]]
            batch_y = train_labels_ops[batch_low_ind:batch_high_ind]
            batch_low_ind += BATCH_SIZE
            batch_high_ind += BATCH_SIZE

            _, c_ops = sess.run([optimizerop, cross_entropyop], feed_dict={x: batch_x, y_ops: batch_y})
            avg_cost_ops += c_ops / total_batch_num_ops
            # Uncomment below for output example on first image
            # if i % (total_batch_num // 5) == 0:
            #     network_out = sess.run(y_op, feed_dict={x: eval_ops[0:1], y: eval_labels_ops[0:1]})
            #     label_ex = eval_labels_ops[0]
            #     print("Ex. network output:", network_out)
            #     print("Corresponding label:", label_ex)

        print("Testing accuracy of operations network...")
        test_acc_ops = sess.run(accuracyop, feed_dict={x: eval_ops[:,1:577], y_ops: eval_labels_ops})

        print("Epoch", (epoch+1), "cost =", "{:.3f}".format(avg_cost_ops), "operations test accuracy: ", "{:.3f}".format(test_acc_ops))
        print("\n")

    print("\nTraining Complete.\n")
    acc_dig = sess.run(accuracy, feed_dict={x: eval_digits[:,1:577], y: eval_labels_digits})
    acc_ops = sess.run(accuracyop, feed_dict={x: eval_ops[:,1:577], y_ops: eval_labels_ops})
    acc_total = (acc_dig * train_digits.shape[0] + acc_ops * train_ops.shape[0]) / (train_digits.shape[0] + train_ops.shape[0])
    print("Final Accuracy:", acc_total)



    if SUBMIT:
        print("\n\nRunning submission file through network...")

        # Out variable will be output to file
        out = []

        # Iterate through submission data to test the validity of each equation and store each evaluation in a variable (out)
        # for submission_data_single in submission_data:
        check_tick = time.clock()
        x1 = sess.run(tf.argmax(y_,1), feed_dict={x: submission_data[:,0]})
        x2 = sess.run(tf.argmax(y_,1), feed_dict={x: submission_data[:,2]})
        x3 = sess.run(tf.argmax(y_,1), feed_dict={x: submission_data[:,4]})
        op1 = sess.run(tf.argmax(y_op,1), feed_dict={x: submission_data[:,1]})
        op2 = sess.run(tf.argmax(y_op,1), feed_dict={x: submission_data[:,3]})
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

        datafile_submission = "submission" + time.strftime("%Y-%m-%d--%H:%M:%S") + ".csv"

        with open(datafile_submission, 'w', newline='') as csvfile:
            write = csv.writer(csvfile)
            write.writerow(['index', 'label'])
            for r in out:
                write.writerow(r)
