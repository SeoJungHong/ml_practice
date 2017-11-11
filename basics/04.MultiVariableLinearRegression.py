import tensorflow as tf
import numpy as np

# Multiple variable without bias term (H = WX)

xy = np.loadtxt('data/04.train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]  # 마지막줄 제외한 나머지를 x_data로
y_data = xy[-1]  # 마지막줄을 y_data로

print('x = {0}'.format(x_data))
print('y = {0}'.format(y_data))

W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))  # W = 1 X 3 Matrix

hypothesis = tf.matmul(W, x_data)  # H(X) = WX

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)  # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W))
