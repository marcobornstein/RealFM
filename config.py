configs = {
    'cifar10': {
        'train_bs': 128,
        'test_bs': 128,
        'lr': 1e-3,
        'marginal_cost': 7e-3,
        'local_steps': 6,
        'random_seed': 1996,
        'test_frequency': 500,
        'log_frequency': 30,
        'test_batches': 30,
        'epochs': 100,
        'file_path': 'output',
        'uniform_payoff': False,
        'uniform_cost': False,
        'name': 'realfm-nonuniformP-run1'
    }
}

# Best ablation was
# train_bs = 128, local_steps = 6, mc = 4.15e-3 (8 devices) UNIFORM
# train_bs = 128, local_steps = 6, mc = 5.62e-3 (16 devices) UNIFORM

# train_bs = 128, local_steps = 6, mc = 9e-3 (16 devices) NON-UNIFORM
# train_bs = 128, local_steps = 6, mc = 7e-3 (8 devices) NON-UNIFORM

# random seeds
# Run 1: 1996
# Run 2: 2015
# Run 3: 2019
