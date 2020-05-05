#!/bin/sh

tensorboard --logdir ./data/local --bind_all &
python rlpyt_testing/rplyt_deep_sea_treasure_dqn.py $1 $2
