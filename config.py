configs = {
    'cifar10': {
        'train_bs': 128,
        'test_bs': 1024,
        'lr': 1e-3,
        'a_opt': 0.9,
        'marginal_cost': 8.353e-4,
        'local_steps': 6,
        'random_seed': 1948,
        'test_frequency': 500,
        'log_frequency': 60,
        'test_batches': 30,
        'epochs': 100,
        'k': 18,
        'file_path': 'output',
        'uniform_payoff': True,
        'uniform_cost': True,
        'linear_utility': False,
        'simple_acc': False,
        'name': 'realfm-uniform-run1'
    },

    'mnist': {
            'train_bs': 128,
            'test_bs': 1024,
            'lr': 1e-3,
            'a_opt': 0.995,
            'marginal_cost': 0.4596,
            'local_steps': 6,
            'random_seed': 1948,
            'log_frequency': 30,
            'test_batches': 30,
            'epochs': 50,
            'simple_acc': True,
            'k': 0.25,
            'file_path': 'output',
            'uniform_payoff': False,
            'uniform_cost': False,
            'linear_utility': False,
            'name': 'realfm-nonuniformPC-run1'
        }
}

# CIFAR 10 (Realistic): Non-Uniform (Expected Payoff is 1 (ranges between 0.9 and 1.1)) & Uniform (Payoff = 1), K=18
# 8 Device: Expected Dataset size is 5,500 --> MC = 8.353e-4
# train_bs = 128, local_steps = 6, mc = 8.353e-4 (8 devices)
# 16 Device: Expected Dataset size is 3,000 --> MC = 1.073e-3
# train_bs = 128, local_steps = 6, mc = 1.073e-3 (16 devices)
# =======================================================================
# CIFAR 10 (Linear):
# 8 Device: Expected Dataset size is 5,500 --> MC = 2.269e-5
# train_bs = 128, local_steps = 6, mc = 2.269e-5 (8 devices)
# 16 Device: Expected Dataset size is 3,000 --> MC = 5.403e-5
# train_bs = 128, local_steps = 6, mc = 5.403e-5 (16 devices)
# =======================================================================

# MNIST (Realistic)
# 8 Device: Expected Dataset size is 7,000 --> MC = 0.350475
# train_bs = 128, local_steps = 6, mc = 0.350475 (8 devices)
# 16 Device: Expected Dataset size is 3,500 --> MC = 0.4596
# train_bs = 128, local_steps = 6, mc = 0.4596 (16 devices)
# =======================================================================
# CIFAR 10 (Linear):
# 8 Device: Expected Dataset size is 7,000 --> MC = 8.537e-7
# train_bs = 128, local_steps = 6, mc = 8.537e-7 (8 devices)
# 16 Device: Expected Dataset size is 3,500 --> MC = 2.414e-6
# train_bs = 128, local_steps = 6, mc = 2.414e-6 (16 devices)
# =======================================================================


# random seeds
# Run 1: 1948
# Run 2: 1996
# Run 3: 2019
