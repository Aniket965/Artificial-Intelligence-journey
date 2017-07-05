import tensorflow as tf 
W = tf.Variable([.3],dtype = tf.float32) 
b = tf.Variable([-.3],dtype = tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)

loss = tf.reduce_sum(squared_deltas)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# for instializing variables in Tensorflow
init = tf.global_variables_initializer()
sess = tf.Session()
# variables only initailzed by sess.run()
sess.run(init)
for i in range(1000):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})

# # for fixing value
# fixW = tf.assign(W,[-1.])
# fixb = tf.assign(b,[1.])
# sess.run([fixW,fixb])
# print(sess.run(linear_model,{x:[1,2,3,4]}))
# print(sess.run(loss,))

print(sess.run([W,b]))