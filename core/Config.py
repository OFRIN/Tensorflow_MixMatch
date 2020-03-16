import tensorflow as tf

def flags_to_dict(flags):
    return {k : flags[k].value for k in flags}

def get_config():
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    
    ###############################################################################
    # Default Config
    ###############################################################################
    flags.DEFINE_string('use_gpu', '0', 'unknown')
    flags.DEFINE_integer('seed', 0, 'unknown')
    flags.DEFINE_integer('labels', 250, 'unknown')
    
    ###############################################################################
    # Training Schedule
    ###############################################################################
    flags.DEFINE_float('learning_rate', 0.002, 'unknown')
    
    flags.DEFINE_integer('batch_size', 64, 'unknown')
    flags.DEFINE_integer('batch_size_per_gpu', 64, 'unknown')
    
    flags.DEFINE_integer('log_iteration', 256, 'unknown')
    flags.DEFINE_integer('valid_iteration', 1024, 'unknown')

    flags.DEFINE_integer('max_epochs', 2048, 'unknown')
    flags.DEFINE_integer('max_iteration', 1024 * 1024, 'unknown') # 1024 epochs -> 2048 epochs
    
    flags.DEFINE_integer('labeled_examples', 250, 'unknown')
    flags.DEFINE_integer('validation_examples', 5000, 'unknown')
    
    ###############################################################################
    # Training Technology
    ###############################################################################
    flags.DEFINE_string('augment', 'weakly_augment', 'None/weakly_augment/randaugment')
    flags.DEFINE_float('weight_decay', 0.02, 'unknown')

    flags.DEFINE_boolean('ema', True, 'unknown')
    flags.DEFINE_float('ema_decay', 0.999, 'unknown')
    
    # for MixMatch
    flags.DEFINE_integer('rampup_length', 16, 'unknown')
    flags.DEFINE_integer('rampup_iteration', 16 * 1024, 'unknown')
    
    flags.DEFINE_integer('num_labeled', 250, 'unknown')
    flags.DEFINE_float('alpha', 0.75, 'unknown')
    flags.DEFINE_float('lambda_u', 75, 'unknown')
    flags.DEFINE_float('T', 0.5, 'unknown')
    flags.DEFINE_integer('K', 2, 'unknown')

    return FLAGS

if __name__ == '__main__':
    import json
    
    flags = get_config()

    print(flags.use_gpu)
    print(flags_to_dict(flags))
    
    # print(flags.mixup)
    # print(flags.efficientnet_option)

    