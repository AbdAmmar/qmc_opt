#!/usr/bin/bash
#SBATCH --job-name=opt
#SBATCH -N 1 -n 36 -c 1 --exclusive #-t 10:00:00
set -e
source $HOME/qp2/quantum_package.rc
source $HOME/qmcchem2/qmcchemrc
python Jopt.py $1
