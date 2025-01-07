#!/bin/bash
#SBATCH -J gensim_mt
#SBATCH --output=gensim_mt_%j.out
#SBATCH --error=gensim_mt_%j.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=20000
#SBATCH --time=48:00:00
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=standard-gpu
#SBATCH --qos=standard-gpu

ml Miniconda3
source activate algen 
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/cm/shared/apps/cuda11.2/toolkit/11.2.2

python /shared/work-external/jvilalta/GenSimAI_2.1/generator_lous.py -i /shared/work-external/jvilalta/multitargetGM/full/chembl_spec_set.smi -e 100 -v 0.1 -t 0.1 -b 100 -s 1.2 -q 2500 -n "gensim_test" -tl -pa /shared/work-external/jvilalta/multitargetGM/full/full_init_spec_set.smi -r /shared/work-external/jvilalta/multitargetGM/full/lous_generative_test_gpu_weights.hdf5 -al 10 -qed 0.6 -sa 5 -ta 0.6

