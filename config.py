configs = {
    'cifar10': {
        'train_bs': 128,
        'test_bs': 1024,
        'lr': 1e-3,
        'a_opt': 0.9,
        'marginal_cost': 9.075e-4,
        'local_steps': 6,
        'random_seed': 1996,
        'test_frequency': 500,
        'log_frequency': 5,
        'test_batches': 30,
        'epochs': 100,
        'k': 25,
        'file_path': 'output',
        'uniform_payoff': True,
        'uniform_cost': True,
        'linear_utility': False,
        'name': 'realfm-uniform-run1'
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
            'k': 1,
            'file_path': 'output',
            'uniform_payoff': True,
            'uniform_cost': True,
            'linear_utility': False,
            'name': 'test-mnist'  # 'realfm-nonuniformP-run1'
        }
}

# CIFAR 10 (Realistic): Non-Uniform (Expected Payoff is 1 (ranges between 4/5 and 6/5)) & Uniform (Payoff = 1), K=25
# 8 Device: Expected Dataset size is 5,500 --> MC = 7.168e-4
# train_bs = 128, local_steps = 6, mc = 7.168e-4 (8 devices)
# 16 Device: Expected Dataset size is 3,000 --> MC = 9.075e-4
# train_bs = 128, local_steps = 6, mc = 9.075e-4 (16 devices)
# =======================================================================
# CIFAR 10 (Linear):
# 8 Device: Expected Dataset size is 5,500 --> MC = 2.528e-5
# train_bs = 128, local_steps = 6, mc = 2.528e-5 (8 devices)
# 16 Device: Expected Dataset size is 3,000 --> MC = 5.995e-5
# train_bs = 128, local_steps = 6, mc = 5.995e-5 (16 devices)
# =======================================================================

# for 16 device, payoff function between 0.9 and 1.1 because data will be zero

# MNIST
# train_bs = 128, local_steps = 6, mc = 1.3e-2 (16 devices) UNIFORM

# random seeds
# Run 1: 2000/2010
# Run 2: 2015
# Run 3: 2019
