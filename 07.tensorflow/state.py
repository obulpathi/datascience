import tensorflow as tf

state = tf.Variable(0, name="counter")
new_value = tf.add(state, tf.constant(1))
update = tf.assign(state, new_value)

init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)

print(session.run(state))
for _ in range(3):
    session.run(update)
    print(session.run(state))
