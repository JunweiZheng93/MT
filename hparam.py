hparam = {
    # -------------- dataset relevant ---------------------

    'category': 'chair',
    'max_num_parts': 4,

    # ---------------- training relevant ------------------

    'training_process': '1',  # should be one of 1, 2 and 3

    # only valid when training_process is 2 or 3. If training_process is 2, model_path should
    # be the path of model after training process 1. If training_process is 3, model_path should
    # be the path of model after training process 2
    'model_path': 'results/20220218091912/process_2/checkpoint.h5',
    'epochs': 200,
    'batch_size': 64,
    'split_ratio': 0.8,  # ratio to split dataset into training set and test set
    'shuffle': True,
    'optimizer': 'adam',  # adam or sgd
    'lr': 1e-3,
    'decay_rate': 0.8,  # decay rate of learning rate
    'decay_step_size': 50,
    'bce_weight': 0.8,
    'which_gpu': 1,

    # ---------------- attention settings --------------------

    'use_attention': True,
    'keep_channel': False,
    'use_extra_loss': False,
    'which_layer': '0',
    'num_blocks': 3,
    'num_heads': 8,
    'd_model': 256,  # dimension of the input tensor for attention block
}
