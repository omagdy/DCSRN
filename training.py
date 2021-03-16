import os
import time
import numpy as np
import tensorflow as tf
from model import Generator
from loss_functions import supervised_loss
from plotting import generate_images, plot_losses
from data_preparing import get_batch_data
from model_checkpoints import get_generator, save_generator

@tf.function
def train_step(real_x, real_y, generator_g, generator_optimizer):
    
    with tf.GradientTape(persistent=True) as tape:
        
        fake_y = generator_g(real_x, training=True)

        gen_g_super_loss = supervised_loss(real_y, fake_y)
                
    gradients_of_generator = tape.gradient(gen_g_super_loss, generator_g.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_g.trainable_variables))
    
    return gen_g_super_loss


def training_loop(LR_G, EPOCHS, BATCH_SIZE, N_TRAINING_DATA):

    print("Began training at "+time.ctime())

    lr_data = np.load('data/3d_lr_data.npy') # (N, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1)
    hr_data = np.load('data/3d_hr_data.npy') # (N, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1)
    
    PATCH_SIZES = hr_data.shape[1:4]         # (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
    assert(PATCH_SIZES==lr_data.shape[1:4])
    PATCH_SIZE  = PATCH_SIZES[0]
    assert(PATCH_SIZE==PATCH_SIZES[1]==PATCH_SIZES[2])
    print("Patch Size is "+str(PATCH_SIZE))

    total  = N_TRAINING_DATA

    generator_g, generator_optimizer, ckpt_manager = get_generator(PATCH_SIZE, LR_G)

    comparison_image    = 900
    comparison_image_hr = hr_data[comparison_image]
    comparison_image_lr = lr_data[comparison_image]

    generate_images(generator_g, comparison_image_lr, comparison_image_hr, PATCH_SIZE, "a_first_plot")

    epochs_plot = []
    total_generator_g_error_plot = []

    for epoch in range(EPOCHS):
        print("Began epoch "+str(epoch)+" at "+time.ctime())
        
        data_x = lr_data[0:N_TRAINING_DATA]
        data_y = hr_data[0:N_TRAINING_DATA]
        
        for i in range(0, total, BATCH_SIZE):
            r = np.random.randint(0,2,3)
            batch_data = get_batch_data(data_x, i, BATCH_SIZE, r[0], r[1], r[2])
            batch_label = get_batch_data(data_y, i, BATCH_SIZE, r[0], r[1], r[2])
#             batch_data = get_batch_data(data_x, i, BATCH_SIZE)
#             batch_label = get_batch_data(data_y, i, BATCH_SIZE)
            generator_loss = train_step(batch_data, batch_label, generator_g, generator_optimizer).numpy()

        epochs_plot.append(epoch)
        total_generator_g_error_plot.append(generator_loss)
                
        comparison_image_hr = hr_data[comparison_image]
        comparison_image_lr = lr_data[comparison_image]

        generate_images(generator_g, comparison_image_lr, comparison_image_hr, PATCH_SIZE, "epoch_"+str(epoch) ," Epoch: "+str(epoch) )
        print("Finished epoch "+str(epoch)+" at "+time.ctime()+". Loss = "+str(generator_loss)+".")
        hr_data, lr_data = shuffle(hr_data, lr_data)
#         if (epoch + 1) % 5 == 0:
#             save_generator(ckpt_manager, epoch+1)
            
    save_generator(ckpt_manager, "final_epoch")

    plot_losses(epochs_plot, total_generator_g_error_plot)
    generate_images(generator_g, comparison_image_lr, comparison_image_hr, PATCH_SIZE, "z_final_plot")
    print("Finished training at "+time.ctime())