import numpy as np
import tensorflow as tf
import csv
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import time

# Hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 20
BATCH_SIZE = 100
DATA_PORTION = 800 # 80,000 Maximum
TRAIN_PORTION = int(5 * DATA_PORTION // 5)
TEST_PORTION = DATA_PORTION - TRAIN_PORTION
submission_data_portion = 20000 # 20,000 Maximum

# Do you want to build a submit file?
SUBMIT = False


# Retrieve data from csv files
# Data kept in train_data and eval_data
# Labels kept in train_labels and eval_labels
print("\n\n\nFetching data...\n\n")
with open('../train.csv', newline='') as csvfile:
    read = csv.reader(csvfile)
    i = 0
    train_data = []
    test_data = []
    for row in read:
        del row[0]
        if(i > 0 and i <= TRAIN_PORTION):
            train_data.append(row)
        if(i > 0 and i > TRAIN_PORTION):
            test_data.append(row)
        if i % DATA_PORTION == 0:
            if i > 0:
                break
        i += 1
    print("Finished reading training images.")


    # Change type to numpy array for use with tensorflow
    train_data = np.array(train_data, dtype='f4')
    eval_data = np.array(test_data, dtype='f4')
    print("Training image shape:", train_data.shape)
    print("Testing image shape:", eval_data.shape)


with open('../train_labels.csv', newline='') as csvfile:
    read = csv.reader(csvfile)

    i = 0
    train_lab = []
    test_lab = []
    for row in read:
        if(i > 0 and i <= TRAIN_PORTION):
            train_lab.append(row[1])
        if(i > 0 and i > TRAIN_PORTION):
            test_lab.append(row[1])
        if i % DATA_PORTION == 0:
            if i > 0:
                break
        i += 1
    print("Finished reading training labels.")

    # Change type to numpy array for use with tensorflow
    train_labels = np.asarray(train_lab, dtype=np.int32)
    eval_labels = np.asarray(test_lab, dtype=np.int32)
    # print(train_labels.shape)

    # Uncomment to view training image at index of your choice
    # train_data = train_data.reshape(train_data.shape[0], 24, 24)
    # plt.imshow(train_data[20], cmap='gray')
    # plt.title(train_labels[20])
    # plt.show()

if SUBMIT:
    with open('../test.csv', newline='') as csvfile:
        read = csv.reader(csvfile)

        i = 0
        submission_data = []
        test_data = []
        for row in read:
            del row[0]
            if(i > 0 and i <= submission_data_portion):
                submission_data.append(row)
            if i % submission_data_portion == 0:
                if i > 0:
                    break
            i += 1
        print("Finished reading submission images.")


        submission_data = np.array(submission_data, dtype='f4')
        # print("Submission image shape:", submission_data.shape)

    submission_data = submission_data.reshape(submission_data_portion, 24, 120)
    submission_data = submission_data.transpose([0,2,1])
    submission_data = submission_data.reshape(submission_data_portion, 5, 24, 24)
    submission_data = submission_data.transpose([0,1,3,2])
    submission_data = submission_data.reshape(submission_data_portion, 5, 576)
    # print("Submission image shape:", submission_data[0].shape)


# Build one-hot arrays for labels
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nTrain_labels.size:",train_labels.size)
print("train_labels.shape:",train_labels.shape)
y_train = np.zeros((train_labels.size, 13))
print("y_train.shape:",y_train.shape)
y_train[range(train_labels.size), train_labels] = 1.0
train_labels = y_train

y_eval = np.zeros((eval_labels.size, 13))
y_eval[range(eval_labels.size), eval_labels] = 1.0
eval_labels = y_eval

# Input placeholder
x = tf.placeholder(tf.float32, [None,576])
# Reshape for convolution
x_shaped = tf.reshape(x, [-1,24,24,1])

# Output placeholder
y = tf.placeholder(tf.float32, [None, 13])

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


# Model definition
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
wd2 = tf.Variable(tf.truncated_normal([1000, 13], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([13], stddev=0.01), name='bd2')
dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
y_ = tf.nn.softmax(dense_layer2)

# Cost function declaration (cross-entropy)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense_layer2, labels=y))

# Set up optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)

# Define accuracy assessment
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialization operator
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    total_batch_num = int(TRAIN_PORTION / BATCH_SIZE)


    for epoch in range(EPOCHS):
        print("Starting epoch", epoch)
        print("\nTotal number of batches to be run:", total_batch_num)
        print("\n")
        avg_cost = 0
        batch_low_ind = 0
        batch_high_ind = batch_low_ind + BATCH_SIZE

        # Shuffle training data
        shuff = list(range(TRAIN_PORTION))
        np.random.shuffle(shuff)
        train_data = train_data[shuff]
        train_labels = train_labels[shuff]

        for i in range(total_batch_num):
            if((i + 1) % 20 == 0 or i == 0):
                print("Current Batch Number:", i+1)
            # Extract batch interval from total dataset
            batch_x = train_data[batch_low_ind:batch_high_ind]
            batch_y = train_labels[batch_low_ind:batch_high_ind]
            batch_low_ind += BATCH_SIZE
            batch_high_ind += BATCH_SIZE

            _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch_num
            # Uncomment below for example on first image
            # if i % (total_batch_num // 5) == 0:
            #     network_out = sess.run(y_, feed_dict={x: eval_data[0:1], y: eval_labels[0:1]})
            #     label_ex = eval_labels[0]
            #     print("Ex. network output:", network_out)
            #     print("Corresponding label:", label_ex)
        print("Testing accuracy...")
        test_acc = sess.run(accuracy, feed_dict={x: eval_data, y: eval_labels})


        print("Epoch", (epoch+1), "cost =", "{:.3f}".format(avg_cost), "test accuracy: ", "{:.3f}".format(test_acc))
        print("\n\n\n\n")

    print("\nTraining Complete.")
    print("Final Accuracy:",sess.run(accuracy, feed_dict={x: eval_data, y: eval_labels}))



    if SUBMIT:
        print("Running submission file through network...")

        # Out variable will be output to file
        out = []

        # Iterate through submission data to test the validity of each equation and store each evaluation in a variable (out)
        # for submission_data_single in submission_data:
        check_tick = time.clock()
        x1 = sess.run(tf.argmax(y_,1), feed_dict={x: submission_data[:,0]})
        x2 = sess.run(tf.argmax(y_,1), feed_dict={x: submission_data[:,2]})
        x3 = sess.run(tf.argmax(y_,1), feed_dict={x: submission_data[:,4]})
        op1 = sess.run(tf.argmax(y_,1), feed_dict={x: submission_data[:,1]})
        op2 = sess.run(tf.argmax(y_,1), feed_dict={x: submission_data[:,3]})
        check_toc = time.clock()
        print("Forward pass of submission data (s):", check_toc-check_tick)
        # Determine whether equation is true or false
        row = []
        row.append(0)
        row.append(0)
        for i in range(submission_data.shape[0]):
            row[0] = i
            if op1[i] == 12:
                if op2[i] == 11:
                    row[1] = int(x1[i] == x2[i] - x3[i])
                else:
                    row[1] = int(x1[i] == x2[i] + x3[i])
            else:
                if op1[i] == 11:
                    row[1] = int(x1[i] - x2[i] == x3[i])
                else:
                    row[1] = int(x1[i] + x2[i] == x3[i])
            out.append([row[0], row[1]])


with open('submission.csv', 'w', newline='') as csvfile:
    write = csv.writer(csvfile)
    write.writerow(['index', 'label'])
    for r in out:
        write.writerow(r)
