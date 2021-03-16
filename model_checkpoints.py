import tensorflow as tf

def get_generator(PATCH_SIZE, LR_G):
    
    generator_g         = Generator(PATCH_SIZE)
    generator_optimizer = tf.keras.optimizers.Adam(LR_G)
    
    path = "model_checkpoints/"

    ckpt = tf.train.Checkpoint(generator_g=generator_g,
                               generator_optimizer=generator_optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=1)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')
    else:
        print("No checkpoint found! Staring from scratch!")
                
    return generator_g, generator_optimizer, ckpt_manager

def save_generator(ckpt_manager, epoch):
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))