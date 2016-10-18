import tensorflow as tf

x = tf.Variable(tf.constant(2))
y = tf.Variable(tf.constant(3))
z = x * y

init = tf.initialize_all_variables()

session = tf.Session()

session.run(init)
print(session.run(z))
