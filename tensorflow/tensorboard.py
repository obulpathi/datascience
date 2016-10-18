"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

# start the interactive session
session = tf.InteractiveSession()

# create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

w_h = tf.histogram_summary("weights", W)
b_h = tf.histogram_summary("biases", b)

# define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.scalar_summary("cost_function", cross_entropy)

# define logging
merge = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('/tmp/tensorflow/logs', graph=tf.get_default_graph())

tf.initialize_all_variables().run()

# train
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  _, summary = session.run([train, merge], feed_dict={x: batch_xs, y_: batch_ys})
  if i % 10 == 0:
      summary_writer.add_summary(summary, i)

# flush all buffers and cluse files
summary_writer.flush()
summary_writer.close()

# test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("\nClassification Accuracy: {0:.2f}%".format(100 * accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})))
