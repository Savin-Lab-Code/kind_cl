#! /bin/bash

#run script to train with kindergarten cl

#location of repo
workdir='~/projects/kind_cl'

#location of config file
cfgname='~/projecdts/kind_cl/demo/33.cfg'

python $workdir/process/rnn/fulltrain.py $cfgname


