#!/bin/bash
# MuJoCo Environment Setup for DPPO
# This script sets up the necessary environment variables for mujoco-py to work

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"

echo "MuJoCo environment variables set successfully!"
echo "LD_LIBRARY_PATH includes: $HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
echo "MUJOCO_PY_MUJOCO_PATH: $MUJOCO_PY_MUJOCO_PATH"
