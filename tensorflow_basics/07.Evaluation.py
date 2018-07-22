import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Constants
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

learning_rate = 0.01
training_epochs = 10
batch_size = 100

data = input_data.read_data_sets('data', one_hot=True)

X = tf.placeholder('float', [None, IMAGE_PIXELS], name='images')
Y = tf.placeholder('float', [None, 10], name='labels')

# Define Model
W = tf.get_variable('weights', shape=[IMAGE_PIXELS, 10])
b = tf.Variable(tf.zeros([10]), name='biases')
activation = tf.nn.softmax(tf.add(tf.matmul(X, W), b))
cost_operation = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(activation), reduction_indices=1))  # Cross entropy

# Set Training
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss=cost_operation)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    avg_cost = 0.
    batch_count = int(data.train.num_examples / batch_size)
    for i in range(batch_count):
        batch_xs, batch_ys = data.train.next_batch(batch_size)
        _, cost = sess.run(
            [training_operation, cost_operation],
            feed_dict={X: batch_xs, Y: batch_ys}
        )
        avg_cost += cost / batch_count
    print('Epoch:', epoch, 'Cost:', avg_cost)
print('-----------------------------')

correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Accuracy:', accuracy.eval(session=sess, feed_dict={X: data.test.images, Y: data.test.labels}))
