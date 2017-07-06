from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32,[None,784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


# softmax - function for forcing to make probablity to 1
# activator are used to provide non linearity
# some activator functions - 
# identity function 
# binary step function postive input is 1
## where as negative input is 0
