
import tensorflow as tf


# Setting the placeholders
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

init = tf.initialize_all_variables()


# Setting the model, Y from Y = WX + b
# And we're using softmax
Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) + b)
# And these are the answers
Y_ = tf.placeholder(tf.float32, [None, 10])

# We use cross entropy for our loss function
# Reduce sum just adds all the elements of the vector
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# Number of correct answers found in thebatch
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
# Accuracy in the model for the iteration
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Gradient descent
# Partial derivatives are calculated automatically
optimizer = tf.train.GradientDescentOptimizer(0.003)
# Size of the gradient step for each iteration - it's currently consistent
train_step = optimizer.minimize(cross_entropy)

# We need to start the session so all the Lazy setup gets executed
sess = tf.Session()
sess.run(init)


# Get the images for the actual test
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)


# We begin the actual computation
for i in range(1000):

	# We load the images in the placeholders
	batch_X, batch_Y = mnist.train.next_batch(100)
	# And we create our results dictionary

	# Updating training data
	train_data = {X: batch_X, Y_: batch_Y}
	a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
	print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))

	# Updating test data
	test_data={X: mnist.test.images, Y_: mnist.test.labels}
	a,c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
	print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

	# Backpropagation training step
	sess.run(train_step, feed_dict=train_data)






