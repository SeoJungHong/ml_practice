import tensorflow as tf
import numpy as np

# load data
xy = np.loadtxt('data/06.train.txt', unpack=True, dtype='float32')

# pre-process data
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])
print('x_data')
print(x_data)
print('y_data')
print(y_data)

X = tf.placeholder("float", [None, 3])  # x1, x2, 1(bias)
Y = tf.placeholder("float", [None, 3])  # A, B, C

# Set model weights
W = tf.Variable(tf.zeros([3, 3]))

# Construct model
hypothesis = tf.nn.softmax(tf.matmul(X, W))

learning_rate = 0.01
# Cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss=cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# Training
for step in range(5001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 200 == 0:
        print(step)
        print(sess.run(cost, feed_dict={X: x_data, Y: y_data}))
        print(sess.run(W))
        print('----------------------------------------------')

# Testing
print('Test!')
a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7]]})
print(a, sess.run(tf.argmax(a, 1)))

b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4]]})
print(b, sess.run(tf.argmax(b, 1)))

c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0]]})
print(c, sess.run(tf.argmax(c, 1)))

all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7], [1, 3, 4], [1, 1, 0]]})
print(all, sess.run(tf.argmax(all, 1)))
