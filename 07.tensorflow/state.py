import tensorflow as tf

state = tf.Variable(0, name="counter")
new_value = tf.add(state, tf.constant(1))
update = tf.assign(state, new_value)

session = tf.Session()
session.run(tf.initialize_all_variables())

print(session.run(state))
for _ in range(3):
    session.run(update)
    print(session.run(state))
