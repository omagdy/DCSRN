import tensorflow as tf

def supervised_loss(real, fake):
	# loss = tf.reduce_mean(tf.square(real - fake)) # L2 Loss
	loss = tf.losses.mean_squared_error(real, fake) # L2 Loss
    # loss = tf.reduce_mean(tf.abs(real - fake))  # L1 Loss
	return loss
