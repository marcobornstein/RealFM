#!/usr/bin/env bash
# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=realfm    # sets the job name if not set from environment
#SBATCH --time=00:40:00    # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger    # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger    # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --ntasks=16
#SBATCH --gres=gpu:4
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE
#SBATCH --nodes=1

module purge
module load cuda/11.7.0
module load mpi
source ../../../../../cmlscratch/marcob/environments/compressed/bin/activate

mpirun -n 16 python realfm.py --dataset mnist --seed 1 --name realfm-linear-uniform-noniid-run1
mpirun -n 16 python realfm.py --dataset mnist --seed 2 --name realfm-linear-uniform-noniid-run2
mpirun -n 16 python realfm.py --dataset mnist --seed 3 --name realfm-linear-uniform-noniid-run3