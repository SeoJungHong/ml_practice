import time

import numpy as np
import pandas as pd
import tensorflow as tf

N_INPUT = 10
N_LAYER_1 = 256
N_LAYER_2 = 256
N_OUTPUT = 1
N_TRAIN_DATA = 40000

# Parameters
LEARNING_RATE = 0.001
TRAINING_EPOCHS = 5000
BATCH_SIZE = 5000
DISPLAY_STEP = 50

INPUT_DATA = 'data/simple_interpolate.csv'


def multilayer_perceptron(x, weights, biases, keep_prob):
    with tf.variable_scope("LAYER_1"):
        layer_1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1'])), keep_prob)
    with tf.variable_scope("LAYER_2"):
        layer_2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])), keep_prob)
    return tf.matmul(layer_2, weights['out']) + biases['out']


def normalize(data):
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    data -= data_mean
    data /= data_std


# Data Loading
data = pd.read_csv(INPUT_DATA, index_col=0)
train_data = data.iloc[:N_TRAIN_DATA]
test_data = data.iloc[N_TRAIN_DATA:]
columns = train_data.columns.tolist()
label_column = columns.pop()  # Last column is label

train_data_label = train_data[label_column]
test_data_label = test_data[label_column]
train_data = train_data[columns]
test_data = test_data[columns]
print('* Raw Data')
print('train data')
print(train_data.head())
print('train data label')
print(train_data_label.head())
print('test data')
print(test_data.head())
print('test data label')
print(test_data_label.head())
print('----------------------')

# Data conversion & preprocessing
train_data = train_data.as_matrix()
test_data = test_data.as_matrix()
train_data_label = train_data_label.as_matrix().reshape(-1, 1)  # inversion
test_data_label = test_data_label.as_matrix().reshape(-1, 1)  # inversion

# Normalizing
for data in [train_data, train_data_label, test_data, test_data_label]:
    normalize(data)

print('* Normalized Data')
print('train data')
print(train_data[0:5])
print('train data label')
print(train_data_label[0:5])
print('test data')
print(test_data[0:5])
print('test data label')
print(test_data_label[0:5])
print('----------------------')

# tf Graph input
x = tf.placeholder("float", [None, N_INPUT], name="Input_x")
y = tf.placeholder("float", [None, N_OUTPUT], name="target_y")
dropout_keep_prob = tf.placeholder("float", name="dropout")

# Store layers weight & bias
stddev = 0.1
with tf.variable_scope("WEIGHTS"):
    weights = {
        'h1': tf.Variable(tf.random_normal([N_INPUT, N_LAYER_1], stddev=stddev)),
        'h2': tf.Variable(tf.random_normal([N_LAYER_1, N_LAYER_2], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([N_LAYER_2, N_OUTPUT], stddev=stddev))
    }

with tf.variable_scope("BIASES"):
    biases = {
        'b1': tf.Variable(tf.random_normal([N_LAYER_1])),
        'b2': tf.Variable(tf.random_normal([N_LAYER_2])),
        'out': tf.Variable(tf.random_normal([N_OUTPUT]))
    }

# Construct model
pred = multilayer_perceptron(x, weights, biases, dropout_keep_prob)

# Define loss and optimizer
cost = tf.reduce_sum(tf.square(tf.subtract(y, pred)))  # RMS
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)  # Adam Optimizer

# Accuracy
ymean = tf.reduce_mean(y)
SSE = tf.reduce_sum(tf.square(tf.subtract(y, pred)))
SSR = tf.reduce_sum(tf.square(tf.subtract(pred, ymean)))
r_squared = SSR / (SSE + SSR)
accuracy = r_squared

# Initializing the variables
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())

# Summary writer
tf.summary.scalar('RMS', cost)
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()
directory_name = 'logs'
summary_writer = tf.summary.FileWriter(directory_name, graph=sess.graph)

num_data = train_data.shape[0]

start = time.time()
# Training cycle
for epoch in range(TRAINING_EPOCHS):
    avg_cost = 0.
    total_batch = int(num_data / BATCH_SIZE)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = train_data[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]
        batch_ys = train_data_label[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]
        # Fit training using batch data
        summary, _ = sess.run([merged, optimizer], feed_dict={x: batch_xs, y: batch_ys, dropout_keep_prob: 0.7})
        # Compute average loss
        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, dropout_keep_prob: 1.}) / total_batch
        summary_writer.add_summary(summary, epoch * total_batch + i)

    # Display logs per epoch step
    if epoch % DISPLAY_STEP == 0:
        print("Epoch: %04d/%04d cost: %.6f" % epoch, TRAINING_EPOCHS, avg_cost)
        train_acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, dropout_keep_prob: 1.})
        print("Training accuracy: %.3f" % train_acc)
        test_acc = sess.run(accuracy, feed_dict={x: test_data, y: test_data_label, dropout_keep_prob: 1.})
        print("Test R-Squared : %.3f" % test_acc)

    # Early Stopper
    """if i >= 50:
        if validation_accuracy - np.mean(accuracy_list[len(accuracy_list)/2:]) <= 0.01 :
            break"""

end = time.time() - start
print("Optimization Finished")
print("training time: %.2f sec." % end)

test_acc = sess.run(accuracy, feed_dict={x: test_data, y: test_data_label, dropout_keep_prob: 1.})
print("Test R-Squared : %.3f" % test_acc)
