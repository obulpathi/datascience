import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
xi = np.random.rand(100).astype(np.float32)
yi = xi * 0.1 + 0.3

# Try to find values for W and b that compute yi = W * xi + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * xi + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - yi))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
session = tf.Session()
session.run(init)

# Fit the line.
for step in range(101):
    session.run(train)
    if step % 10 == 0:
        print(step, session.run(W), session.run(b))

print("\nFinal Leanrnings:")
print("\tb: {0:.2f}".format(float(session.run(b))))
print("\tW: {0:.2f}".format(float(session.run(W))))
