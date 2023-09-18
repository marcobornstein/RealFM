configs = {
    'cifar10': {
        'train_bs': 128,
        'test_bs': 1024,
        'lr': 1e-3,
        'a_opt': 0.95,
        'marginal_cost': 1e-5,
        'local_steps': 6,
        'random_seed': 1996,
        'test_frequency': 500,
        'log_frequency': 30,
        'test_batches': 30,
        'epochs': 100,
        'file_path': 'output',
        'uniform_payoff': True,
        'uniform_cost': False,
        'linear_utility': True,
        'name': 'linear-nonuniformC-run1'
    },

    'mnist': {
            'train_bs': 128,
            'test_bs': 1024,
            'lr': 1e-3,
            'a_opt': 0.995,
            'marginal_cost': 1.2e-2,
            'local_steps': 6,
            'random_seed': 1996,
            'log_frequency': 30,
            'test_batches': 30,
            'epochs': 50,
            'file_path': 'output',
            'uniform_payoff': True,
            'uniform_cost': True,
            'linear_utility': False,
            'name': 'test-mnist'  # 'realfm-nonuniformP-run1'
        }
}


# CIFAR 10 (Realistic)
# train_bs = 128, local_steps = 6, mc = 4.15e-3 (8 devices) UNIFORM
# train_bs = 128, local_steps = 6, mc = 5.62e-3 (16 devices) UNIFORM

# train_bs = 128, local_steps = 6, mc = 9e-3 (16 devices) NON-UNIFORM
# train_bs = 128, local_steps = 6, mc = 7e-3 (8 devices) NON-UNIFORM

# CIFAR 10 (Linear)
# train_bs = 128, local_steps = 6, mc = 1e-5 (8 devices) UNIFORM


# MNIST
# train_bs = 128, local_steps = 6, mc = 1.3e-2 (16 devices) UNIFORM

# random seeds
# Run 1: 1996
# Run 2: 2015
# Run 3: 2019
