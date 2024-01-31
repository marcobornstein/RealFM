configs = {
    'cifar10': {
        'train_bs': 128,
        'test_bs': 512,
        'lr': 0.05,
        'a_opt': 0.95,
        'marginal_cost': 2.5e-4,
        'local_steps': 6,
        'simple_acc': False,
        'random_seed': 1,
        'test_frequency': 500,
        'log_frequency': 60,
        'test_batches': 30,
        'num_train_data': 50000,
        'epochs': 100,
        'k': 10,
        'file_path': 'output',
        'non_iid': False,
        'dirichlet_value': 0.3,
        'uniform_payoff': True,
        'uniform_cost': True,
        'linear_utility': True,
        'name': 'realfm-linear-uniform-run1'
    },

    'mnist': {
            'train_bs': 128,
            'test_bs': 1024,
            'lr': 1e-3,
            'a_opt': 0.9975,
            'simple_acc': True,
            'marginal_cost': 4e-5,
            'local_steps': 6,
            'random_seed': 1,
            'log_frequency': 30,
            'test_batches': 30,
            'num_train_data': 60000,
            'epochs': 100,
            'k': 0.25,
            'file_path': 'output',
            'non_iid': True,
            'dirichlet_value': 0.3,
            'uniform_payoff': True,
            'uniform_cost': True,
            'linear_utility': True,
            'name': 'realfm-linear-uniform-noniid-run1'
        }
}
# Experiments
# CIFAR 10
# 16 Device: 50,000/16 = 3125 data points per device max
# mapping function f(x) = 3125*(1-e^{-kx}), x is the amount of data
# 8 Device: 50,000/8 = 6250 data points per device max
# mapping function f(x) = 6250*(1-e^{-kx}), x is the amount of data
# marginal cost for both cases is mc = 2.5e-4
# =======================================================================
# MNIST
# 16 Device: 60,000/16 = 3750 data points per device max
# mapping function f(x) = 3750*(1-e^{-kx}), x is the amount of data
# 8 Device: 60,000/8 = 7500 data points per device max
# mapping function f(x) = 7500*(1-e^{-kx}), x is the amount of data
# marginal cost for both cases is mc = 4e-5
