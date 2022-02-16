hparam = {
    # -------------- dataset relevant ---------------------

    'category': 'chair',
    'max_num_parts': 4,

    # ---------------- training relevant ------------------

    'training_process': '3',  # should be one of 1, 2 and 3

    # only valid when training_process is 2 or 3. If training_process is 2, model_path should
    # be the path of model after training process 1. If training_process is 3, model_path should
    # be the path of model after training process 2
    'model_path': 'results/20220216010504/process_2/checkpoint.h5',
    'epochs': 1000,
    'batch_size': 2,
    'split_ratio': 0.8,  # ratio to split dataset into training set and test set
    'shuffle': True,
    'optimizer': 'adam',  # adam or sgd
    'lr': 1e-4,
    'decay_rate': 0.8,  # decay rate of learning rate
    'decay_step_size': 1000,
    'which_gpu': 7,

    # ---------------- attention settings --------------------

    'use_attention': True,
    'keep_channel': False,
    'use_extra_loss': False,
    'which_layer': '5',
    'num_blocks': 6,
    'num_heads': 8,
    'd_model': 256,  # dimension of the input tensor for attention block
}
