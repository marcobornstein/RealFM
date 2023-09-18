configs = {
    'cifar10': {
        'train_bs': 128,
        'test_bs': 1024,
        'lr': 1e-3,
        'a_opt': 0.95,
        'marginal_cost': 5.694e-3,
        'local_steps': 6,
        'random_seed': 1996,
        'test_frequency': 500,
        'log_frequency': 30,
        'test_batches': 30,
        'epochs': 1,
        'file_path': 'output',
        'uniform_payoff': False,
        'uniform_cost': False,
        'linear_utility': False,
        'name': 'linear-nonuniformPC-run1'
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

# CIFAR 10 (Realistic): Non-Uniform (Expected Payoff is 1 (ranges between 1/2 and 3/2)) & Uniform (Payoff = 1)
# 8 Device: Expected Dataset size is 5,500 --> MC = 4.3755e-3
# train_bs = 128, local_steps = 6, mc = 4.3755e-3 (8 devices)
# 16 Device: Expected Dataset size is 3,000 --> MC = 5.694e-3
# train_bs = 128, local_steps = 6, mc = 5.694e-3 (16 devices)
# =======================================================================
# CIFAR 10 (Linear):
# 8 Device: Expected Dataset size is 5,500 --> MC = 1.00185e-5
# train_bs = 128, local_steps = 6, mc = 1.00185e-5 (8 devices)
# 16 Device: Expected Dataset size is 3,000 --> MC = 2.442e-5
# train_bs = 128, local_steps = 6, mc = 2.442e-5 (16 devices)
# =======================================================================


# MNIST
# train_bs = 128, local_steps = 6, mc = 1.3e-2 (16 devices) UNIFORM

# random seeds
# Run 1: 1996
# Run 2: 2015
# Run 3: 2019
