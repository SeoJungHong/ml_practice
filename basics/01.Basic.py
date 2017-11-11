import tensorflow as tf

sess = tf.Session()

a = tf.constant(2)
b = tf.constant(3)

c = a + b

print(sess.run(c))

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    print("Add : %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("Mul : %i" % sess.run(mul, feed_dict={a: 2, b: 3}))
