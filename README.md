# RealFM
Repository for RealFM.

The code can be run with differing parameters by adjusting the config.py script.

Running the code (in serial) consists of calling: python realfm.py.
Running the code (in parallel) consists of calling: mpirun -n 16 python realfm.py

Packages Used:

- matplotlib               3.7.1
- mpi4py                   3.1.4
- numpy                    1.24.3
- scikit-learn             1.2.2
- scipy                    1.10.1
- torch                    2.2.0.dev20230906
- torchaudio               2.2.0.dev20230906
- torchtnt                 0.1.0
- torchvision              0.16.0.dev20230906