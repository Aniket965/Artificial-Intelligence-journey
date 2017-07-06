import tensorflow as tf
# creates a list with 100 elemenet
# equally spaced b/w -3 to 3
x = tf.linspace(-3.0,3.0,100)

print(x)

g = tf.get_default_graph()
print([op.name for op in g.get_operations()])

print(g.get_tensor_by_name('LinSpace:0'))

sess = tf.Session()

print(sess.run(x))

# for closing session
# sess.close()
# for telling the session Explicitly manage the desired
# graph
# sess = tf.Session(graph = graph)

# in tesnorflow unlike numpy we have to compute shapes
print(x.get_shape())

 
