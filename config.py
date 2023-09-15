configs = {
    'cifar10': {
        'train_bs': 128,
        'test_bs': 128,
        'lr': 1e-3,
        'marginal_cost': 3.5e-3,
        'local_steps': 6,
        'random_seed': 101,
        'test_frequency': 500,
        'log_frequency': 10,
        'test_batches': 30,
        'epochs': 70,
        'file_path': 'output',
        'name': 'realfm-best-mc35'
    }
}

# Best ablation was
# train_bs = 128, local_steps = 6
