configs = {
    'cifar10': {
        'train_bs': 128,
        'test_bs': 128,
        'lr': 1e-3,
        'marginal_cost': 4.15e-3,
        'local_steps': 6,
        'random_seed': 101,
        'test_frequency': 500,
        'log_frequency': 30,
        'test_batches': 30,
        'epochs': 100,
        'file_path': 'output',
        'name': 'realfm-best-run1'
    }
}

# Best ablation was
# train_bs = 128, local_steps = 6

# random seeds
# Run 1: 1996
# Run 2: 2015
# Run 3: 2019
