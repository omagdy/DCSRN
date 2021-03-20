import tensorflow as tf

mse = tf.keras.losses.MeanSquaredError()

def supervised_loss(real, fake):
#     loss = tf.reduce_mean(tf.square(real - fake)) # L2 Loss
    loss = mse(real, fake) # L2 Loss
#     loss = tf.reduce_mean(tf.abs(real - fake))  # L1 Loss
    return loss
