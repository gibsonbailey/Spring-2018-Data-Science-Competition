import tensorflow as tf
import numpy as np


# Batch Normalization
def batch_norm(layer_in, convolution=False):
        epsilon = 0.0001
        if convolution:
            scale = tf.Variable(tf.ones([int(layer_in.shape[3])])) #, int(layer_in.shape[2])]))
            Beta = tf.Variable(tf.zeros([int(layer_in.shape[3])])) #, int(layer_in.shape[2])]))
            mean, var = tf.nn.moments(layer_in, [0,1,2])
        else:
            scale = tf.Variable(tf.ones([int(layer_in.shape[1])]))
            Beta = tf.Variable(tf.zeros([int(layer_in.shape[1])]))
            mean, var = tf.nn.moments(layer_in, [0])
        return tf.nn.batch_normalization(layer_in, mean, var, Beta, scale, epsilon)



# Build a generalized convolution layer
def convolution_layer(input_data, num_input_channels, num_filters, filter_shape, pooling=True, pool_shape=[2,2], stride=2, name="conv_layer"):
    conv_filter_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

    # Initilization of filter weights and biases
    weights = tf.Variable(tf.truncated_normal(conv_filter_shape, stddev=0.03), name=name+'_W')
    # biases = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    out_layer = tf.nn.conv2d(input_data, weights, [1,1,1,1], padding='SAME')
    out_layer = batch_norm(out_layer, convolution=True)
    # out_layer += biases
    out_layer = tf.nn.leaky_relu(out_layer)

    if pooling:
        # print("pool_shape[0]:",pool_shape[0])
        kernal_size = [1, 2, 2, 1]
        strides = [1,stride,stride,1]
        out_layer = tf.nn.max_pool(out_layer, ksize=kernal_size, strides=strides, padding='SAME')

    return out_layer



# Model definition
class CNNmodel():
    # Output_length is the number of classification options that the network has
    def __init__(self, model_name, output_length, LEARNING_RATE, dense_layer1_width):
        self.model_name = model_name

        # Proportion of data not dropped out for regularization (0.6 for train and 1.0 for eval)
        self.keep_prob = tf.placeholder(tf.float32)

        # Input placeholder
        self.x = tf.placeholder(tf.float32, [None,576])


        # Reshape for convolution
        self.x_shaped = tf.reshape(self.x, [-1,24,24,1])


        # Output placeholder
        self.y = tf.placeholder(tf.float32, [None, output_length])


        self.layer1 = convolution_layer(self.x_shaped, 1, 32, [5,5], pooling=False, name='layer1')
        kernal_size = [1, 2, 2, 1]
        strides = [1,2,2,1]
        self.layer1_pool = tf.nn.max_pool(self.layer1, ksize=kernal_size, strides=strides, padding='SAME')
        self.layer2 = convolution_layer(self.layer1_pool, 32, 64, [5,5], [2,2], 2, name='layer2')
        # Ouput shape is 6 x 6

        # Pooling bypasses the convolution layers
        self.pool_bypass = tf.nn.max_pool(self.x_shaped, ksize=kernal_size, strides=strides, padding='SAME')


        # Reshaping for dense layers
        self.flattened = tf.reshape(self.layer2, [-1,6*6*64])
        self.flattened1 = tf.reshape(self.layer1, [-1,24*24*32])
        self.flattened2 = tf.reshape(self.pool_bypass, [-1,12*12*1])

        # Inception
        self.flattened = tf.concat([self.flattened, self.flattened1, self.flattened2], 1)
        print("self.flattened.shape[1]:",self.flattened.shape[1])

        # Batch Normalization
        self.x_scale = tf.ones([int(self.flattened.shape[1])])
        self.x_Beta = tf.zeros([int(self.flattened.shape[1])])
        epsilon = 0.001
        self.x_mean, self.x_var = tf.nn.moments(self.flattened, [0])
        self.x_norm = tf.nn.batch_normalization(self.flattened, self.x_mean, self.x_var, self.x_Beta, self.x_scale, epsilon)


        # Dense layer 1 definition
        self.wd1 = tf.Variable(tf.truncated_normal([int(self.flattened.shape[1]), dense_layer1_width], stddev=0.03), name='wd1')
        # self.bd1 = tf.Variable(tf.truncated_normal([dense_layer1_width], stddev=0.01), name='bd1')
        self.dense_layer1 = tf.matmul(self.x_norm, self.wd1) #+ self.bd1
        self.dense_layer1 = tf.nn.relu(self.dense_layer1)
        self.drop_out = tf.nn.dropout(self.dense_layer1, self.keep_prob)
        # Output shape: 1 x dense_layer1_width

        # Batch Normalization
        self.x_scale1 = tf.ones([dense_layer1_width])
        self.x_Beta1 = tf.zeros([dense_layer1_width])
        # epsilon = 0.001
        self.x_mean1, self.x_var1 = tf.nn.moments(self.drop_out, [0])
        self.drop_out_norm = tf.nn.batch_normalization(self.drop_out, self.x_mean1, self.x_var1, self.x_Beta1, self.x_scale1, epsilon)


        # Dense layer 2 definition
        self.wd2 = tf.Variable(tf.truncated_normal([dense_layer1_width, output_length], stddev=0.03), name='wd2')
        # self.bd2 = tf.Variable(tf.truncated_normal([output_length], stddev=0.01), name='bd2')
        self.dense_layer2 = tf.matmul(self.drop_out_norm, self.wd2)# + self.bd2
        self.y_out = tf.nn.softmax(self.dense_layer2)


        # Define multiple cost functions
        # Cost function declaration (cross-entropy)
        self.cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.dense_layer2, labels=self.y))

        # Set up optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.cost_function)

        # Define accuracy assessment
        self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_out,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


    def train(self, sess, EPOCHS, BATCH_SIZE, train_data, train_labels, eval_data, eval_labels):
        # sess.run(self.init_vars)

        total_batch_num = int(train_data.shape[0] / BATCH_SIZE)

        for epoch in range(EPOCHS):
             print("\n\n\nStarting epoch", epoch+1)

             avg_cost = 0
             batch_low_ind = 0
             batch_high_ind = batch_low_ind + BATCH_SIZE

             # Shuffle training data
             shuff_data = list(range(train_data.shape[0]))
             np.random.shuffle(shuff_data)
             train_data = train_data[shuff_data]
             train_labels = train_labels[shuff_data]

             print("\n")
             print("Beginning training of " + self.model_name + "...")
             print("\nTotal number of " + self.model_name + " batches to be run:", total_batch_num)
             print("\n")
             for i in range(total_batch_num):
                 if((i + 1) % 50 == 0 or i == 0):
                     print("Current " + self.model_name + " batch number:", i+1)
                 # Extract batch interval from total dataset
                 batch_x = train_data[batch_low_ind:batch_high_ind,1:train_data.shape[1]]
                 batch_y = train_labels[batch_low_ind:batch_high_ind]
                 batch_low_ind += BATCH_SIZE
                 batch_high_ind += BATCH_SIZE
                 _, c = sess.run([self.optimizer, self.cost_function], feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 0.6})
                 avg_cost += c / total_batch_num


             print("Testing accuracy of " + self.model_name + " network...")
             test_acc = sess.run(self.accuracy, feed_dict={self.x: eval_data[:,1:577], self.y: eval_labels, self.keep_prob: 1.0})

             print("Epoch", (epoch+1), "cost =", "{:.3f}".format(avg_cost), " " + self.model_name + " test accuracy: ", "{:.3f}".format(test_acc))
             print("\n\n")


# Model definition
class CNNmodel_light():
    # Output_length is the number of classification options that the network has
    def __init__(self, model_name, output_length, LEARNING_RATE, dense_layer1_width, cost_function):
        self.model_name = model_name

        # Proportion of data not dropped out for regularization (0.6 for train and 1.0 for eval)
        self.keep_prob = tf.placeholder(tf.float32)
        self.keep_prob_conv = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool)

        # Input placeholder
        self.x = tf.placeholder(tf.float32, [None,576])

        # self.x = batch_norm(self.x)

        # Reshape for convolution
        self.x_shaped = tf.reshape(self.x, [-1,24,24,1])



        # Output placeholder
        self.y = tf.placeholder(tf.float32, [None, output_length])


        self.layer1 = convolution_layer(self.x_shaped, 1, 32, [5,5], name='layer1')
        # kernal_size = [1, 2, 2, 1]
        # strides = [1,2,2,1]
        # self.layer1_pool = tf.nn.max_pool(self.layer1, ksize=kernal_size, strides=strides, padding='SAME')
        self.layer1 = tf.nn.dropout(self.layer1, self.keep_prob_conv)
        self.layer2 = convolution_layer(self.layer1, 32, 64, [5,5], name='layer2')
        # Ouput shape is 6 x 6

        self.layer3 = convolution_layer(self.layer2, 64, 128, [3,3], name='layer3')

        # Pooling bypasses the convolution layers
        # self.pool_bypass = tf.nn.max_pool(self.x_shaped, ksize=kernal_size, strides=strides, padding='SAME')


        # Reshaping for dense layers
        self.flattened = tf.reshape(self.layer3, [-1,3*3*128])
        # self.flattened1 = tf.reshape(self.layer1, [-1,24*24*32])
        # self.flattened2 = tf.reshape(self.pool_bypass, [-1,12*12*1])
        #
        # Inception
        # self.flattened = tf.concat([self.flattened, self.flattened1, self.flattened2], 1)
        # print("self.flattened.shape[1]:",self.flattened.shape[1])

        # Batch Normalization
        # self.flattened = batch_norm(self.flattened)
        # self.x_scale = tf.ones([int(self.flattened.shape[1])])
        # self.x_Beta = tf.zeros([int(self.flattened.shape[1])])
        # epsilon = 0.001
        # self.x_mean, self.x_var = tf.nn.moments(self.flattened, [0])
        # self.x_norm = tf.nn.batch_normalization(self.flattened, self.x_mean, self.x_var, self.x_Beta, self.x_scale, epsilon)


        # Dense layer 1 definition
        self.wd1 = tf.Variable(tf.truncated_normal([int(self.flattened.shape[1]), dense_layer1_width], stddev=0.03), name='wd1')
        # self.bd1 = tf.Variable(tf.truncated_normal([dense_layer1_width], stddev=0.01), name='bd1')
        self.dense_layer1 = tf.matmul(self.flattened, self.wd1)# + self.bd1
        self.dense_layer1 = batch_norm(self.dense_layer1)
        self.dense_layer1 = tf.nn.relu(self.dense_layer1)
        self.drop_out = tf.nn.dropout(self.dense_layer1, self.keep_prob)
        # Output shape: 1 x dense_layer1_width

        # Batch Normalization
        # self.x_scale1 = tf.ones([dense_layer1_width])
        # self.x_Beta1 = tf.zeros([dense_layer1_width])
        # # epsilon = 0.001
        # self.x_mean1, self.x_var1 = tf.nn.moments(self.drop_out, [0])
        # self.drop_out_norm = tf.nn.batch_normalization(self.drop_out, self.x_mean1, self.x_var1, self.x_Beta1, self.x_scale1, epsilon)


        # Dense layer 2 definition
        self.wd2 = tf.Variable(tf.truncated_normal([dense_layer1_width, output_length], stddev=0.03), name='wd2')
        self.bd2 = tf.Variable(tf.truncated_normal([output_length], stddev=0.01), name='bd2')
        self.dense_layer2 = tf.matmul(self.drop_out, self.wd2) + self.bd2
        # self.dense_layer2 = batch_norm(self.dense_layer2)
        self.y_out = tf.nn.softmax(self.dense_layer2)


        # Define multiple cost functions
        # Cost function declaration (cross-entropy)
        if cost_function == "cross_entropy":
            self.cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.dense_layer2, labels=self.y))
        elif cost_function == "MSE":
            self.cost_function = tf.losses.mean_squared_error(predictions=self.dense_layer2, labels=self.y)
        elif cost_function == "huber":
            self.cost_function = tf.losses.huber_loss(predictions=self.dense_layer2, labels=self.y)

        # Set up optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.cost_function)

        # Define accuracy assessment
        self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_out,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


    def train(self, sess, EPOCHS, BATCH_SIZE, train_data, train_labels, eval_data, eval_labels):
        # sess.run(self.init_vars)

        total_batch_num = int(train_data.shape[0] / BATCH_SIZE)

        for epoch in range(EPOCHS):
             print("\n\n\nStarting epoch", epoch+1)

             avg_cost = 0
             batch_low_ind = 0
             batch_high_ind = batch_low_ind + BATCH_SIZE

             # Shuffle training data
             shuff_data = list(range(train_data.shape[0]))
             np.random.shuffle(shuff_data)
             train_data = train_data[shuff_data]
             train_labels = train_labels[shuff_data]

             print("\n")
             print("Beginning training of " + self.model_name + "...")
             print("\nTotal number of " + self.model_name + " batches to be run:", total_batch_num)
             print("\n")
             for i in range(total_batch_num):
                 if((i + 1) % 50 == 0 or i == 0):
                     print("Current " + self.model_name + " batch number:", i+1)
                 # Extract batch interval from total dataset
                 batch_x = train_data[batch_low_ind:batch_high_ind,1:train_data.shape[1]]
                 batch_y = train_labels[batch_low_ind:batch_high_ind]
                 batch_low_ind += BATCH_SIZE
                 batch_high_ind += BATCH_SIZE
                 _, c = sess.run([self.optimizer, self.cost_function], feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 0.6, self.keep_prob_conv: 0.6})
                 avg_cost += c / total_batch_num


             print("Testing accuracy of " + self.model_name + " network...")
             test_acc = sess.run(self.accuracy, feed_dict={self.x: eval_data[:,1:577], self.y: eval_labels, self.keep_prob: 1.0, self.keep_prob_conv: 1.0})
             print("Finished Testing.")
             print("Epoch", (epoch+1), "cost =", "{:.3f}".format(avg_cost), " " + self.model_name + " test accuracy: ", "{:.3f}".format(test_acc))
             print("\n\n")
