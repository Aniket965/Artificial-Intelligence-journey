import tensorflow as tf 
W = tf.Variable([.3],dtype = tf.float32) 
b = tf.Variable([-.3],dtype = tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
# for instializing variables in Tensorflow

init = tf.global_variables_initializer()

# variables only initailzed by sess.run()
sess.run(init)

print()