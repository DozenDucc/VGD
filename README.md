## Installation
1. Clone repository
```
git clone --recurse-submodules git@github.com:DozenDucc/VGD.git
cd VGD
```
2. Create conda environment
```
conda create -n dsrl python=3.9 -y
conda activate dsrl
```
3. Install our fork of DPPO 
```
cd dppo
pip install -e .
pip install -e .[robomimic]
pip install -e .[gym]
cd ..
```
4. Install our fork of Stable Baselines3
```
cd stable-baselines3
pip install -e .
cd ..
```
If you need pretrained diffusion policy checkpoints for the Robomimic and Gym experiments, you can download them from our shared drive and place them into `./dppo/log`.

## Running VGD
To run VGD on Robomimic, call
```
python train_vgd.py --config-path=cfg/robomimic --config-name=vgd_transport_guided.yaml
```


## Acknowledgements
This repository builds on [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) and [DPPO](https://github.com/irom-princeton/dppo).
