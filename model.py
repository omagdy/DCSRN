import tensorflow as tf

k = 24
filter_size = 3

first_conv_filter_number = 2*k
NUMBER_OF_UNITS_PER_BLOCK = 4

utilize_bias = True
# w_init = tf.keras.initializers.VarianceScaling()
w_init = tf.keras.initializers.HeUniform()

def dense_unit(no_of_filters=k, f_size=filter_size):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ELU())
#     result.add(Swish(dtype='float64', trainable=False))    
    result.add(tf.keras.layers.Conv3D(no_of_filters, f_size, kernel_initializer=w_init,
                                      use_bias = utilize_bias, padding='same', dtype='float64'))
    return result


def Generator(patch_size=64):
        
    dense_units_output = []
    
    inputs = tf.keras.layers.Input(shape=[patch_size,patch_size,patch_size,1], dtype='float64')
    conv1 = tf.keras.layers.Conv3D(first_conv_filter_number, filter_size, kernel_initializer=w_init, 
                                   use_bias = utilize_bias, padding='same', dtype='float64')(inputs)
    dense_units_output.append(conv1)

    dense_unit_output_0 = dense_unit(k, filter_size)(conv1)
    dense_units_output.append(dense_unit_output_0)
    dense_unit_output = tf.keras.layers.Concatenate(dtype='float64')([dense_unit_output_0, conv1])
    
    for i in range(NUMBER_OF_UNITS_PER_BLOCK-1):
        dense_unit_output = dense_unit(k, filter_size)(dense_unit_output)
        dense_units_output.append(dense_unit_output)
        dense_unit_output = tf.keras.layers.Concatenate(dtype='float64')(dense_units_output[:-1]+[dense_unit_output])

    reconstruction_output = tf.keras.layers.Conv3D(1, 1, kernel_initializer=w_init, 
                                   use_bias = utilize_bias, padding='same', dtype='float64')(dense_unit_output)
            
    return tf.keras.Model(inputs=inputs, outputs=reconstruction_output)