#!/bin/bash


if [ $# -ne 5 ];
then 
        echo "Wrong number of parameters. Usage: ./particle_weak_scaling.sh LOW HIGH PARTICLES NODES PPN"
        exit 1
fi

TEMPLATE=jobs/templates/isambard_tx2_weak_mesh

FNAME=jobs/weak_mesh-$3particles-$4nodes-$5ppn.job
cp $TEMPLATE $FNAME

MC_PROCS=$(( $4*$5 ))
MC_ROOFLINE=2socket

sed -i "s/MC_LOW/$1/g" $FNAME
sed -i "s/MC_HIGH/$2/g" $FNAME
sed -i "s/MC_PARTICLES/$3/g" $FNAME
sed -i "s/MC_NODES/$4/g" $FNAME
sed -i "s/MC_PPN/$5/g" $FNAME
sed -i "s/MC_PROCS/$MC_PROCS/g" $FNAME
sed -i "s/MC_ROOFLINE/$MC_ROOFLINE/g" $FNAME

source $FNAME