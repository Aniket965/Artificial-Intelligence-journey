import tensorflow as tf

node1 = tf.constant(3.0,dtype=tf.float32)
node2 = tf.constant(4.0,dtype = tf.float32)
# print(node2)
node3 = tf.add(node1,node2)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
add_and_twice = adder_node * 2
sess = tf.Session()

print(node3,sess.run(node3))
print(sess.run([node1,node2]))

print(sess.run(adder_node, {a:3,b:4.5}))
print(sess.run(add_and_twice,{a:[2,3],b:[5.6,7.7]}))