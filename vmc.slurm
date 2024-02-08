#!/usr/bin/bash
#SBATCH --job-name=vmc
#SBATCH -N 1 -n 36 -c 1 --exclusive -t 0:10:00
set -e
source $HOME/qp2/quantum_package.rc
source $HOME/qmcchem2/qmcchemrc
qmcchem run $1
