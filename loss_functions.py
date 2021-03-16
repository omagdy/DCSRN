import tensorflow as tf

def supervised_loss(real, fake, loss_type="l2_loss"):
	if loss_type == "l2_loss":
		loss = tf.reduce_mean(tf.square(real - fake))
	else: #L1 Loss
	    loss = tf.reduce_mean(tf.abs(real - fake))
	return loss
