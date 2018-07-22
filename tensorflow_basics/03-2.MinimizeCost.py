import tensorflow as tf

x_data = (1, 2, 3, 4, 5, 6, 7, 8)
y_data = (1, 2, 3, 4, 5, 6, 7, 8)

W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = tf.multiply(W, X)

cost = tf.reduce_mean(tf.square(hypothesis - Y))

learning_rate = tf.Variable(0.01)
descent = W - tf.multiply(learning_rate, tf.reduce_mean(tf.multiply((tf.multiply(W, X) - Y), X)))
update = W.assign(descent)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(1000):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
