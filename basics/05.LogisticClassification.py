import numpy as np
import tensorflow as tf

xy = np.loadtxt('data/05.train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]  # 마지막줄 제외한 나머지를 x_data로
y_data = xy[-1]  # 마지막줄을 y_data로

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

# hypothesis = tf.div(1., 1. + tf.exp(-h))    # 1/(1 + e^(-W*X))
# tensorflow 의 sigmoid function을 이용해도 된다.
hypothesis = tf.nn.sigmoid(tf.matmul(W, X))

# cost function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

# Minimize
a = tf.Variable(0.1)  # learning rate
optimizer = tf.train.GradientDescentOptimizer(learning_rate=a)
train = optimizer.minimize(loss=cost)  # loss : A `Tensor` containing the value to minimize.

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(4001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

print('------------ Training Done ------------')
# testing
print(sess.run(hypothesis, feed_dict={X: [[1], [2], [2]]}))  # bias(= 1), study_hour, attendance
print(sess.run(hypothesis, feed_dict={X: [[1], [5], [5]]}))
print(sess.run(hypothesis, feed_dict={X: [[1, 1], [4, 3], [3, 5]]}))
print(sess.run(hypothesis, feed_dict={X: x_data}) > 0.5)
