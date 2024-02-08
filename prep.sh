#!/bin/bash

source $HOME/qp2/quantum_package.rc
source $HOME/qmcchem2/qmcchemrc

EZFIO=$1

j2e_type=Mu
env_type=Sum_Gauss
j1e_type=None

expo_O=1.0; coef_O=1.0
expo_H=10000.0; coef_H=0.0;

mu=0.87

time_b=30
time_h=00
time_m=10

io=`pwd`

qmcchem edit -c $EZFIO
qmcchem edit -f 1 $EZFIO
qmcchem edit -l $time_b $EZFIO
qmcchem edit -t $((3600*time_h + 60*time_m - 100)) $EZFIO

echo "0.0" > ${EZFIO}/simulation/ci_threshold
echo "${io}/${EZFIO}/work/${EZFIO}.h5" > ${EZFIO}/trexio/trexio_file

qp set_file ${EZFIO}
qp set hamiltonian mu_erf $mu
qp set jastrow j2e_type $j2e_type
qp set jastrow env_type $env_type
qp set jastrow j1e_type $j1e_type
qp set jastrow env_expo "[${expo_O},${expo_H},${expo_H}]"
qp set jastrow env_coef "[${coef_O},${coef_H},${coef_H}]"
qp set jastrow j1e_expo "[[1.0], [1.0], [1.0]]"
qp set jastrow j1e_coef "[[-0.1], [-0.1], [-0.1]]"
qp set qmcchem ci_threshold 0.0
qp run save_for_qmcchem
#qp run save_bitcpsileft_for_qmcchem
  

