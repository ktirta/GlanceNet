#!/usr/bin/env sh
HOME=`pwd`

# Chamfer Distance
cd $HOME/utils/chamfer_dist
python setup.py install --user

# pointnet2 utils
cd $HOME/utils/pointnet2_ops_lib
python setup.py install --user