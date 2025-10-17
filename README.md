## Installation
1. Clone repository
```
git clone --recurse-submodules git@github.com:DozenDucc/VGD.git
cd VGD
```
2. Create conda environment
```
conda create -n vgd python=3.9 -y
conda activate vgd
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

## Running VGD
To run VGD on Robomimic, call
```
python train_vgd.py --config-path=cfg/robomimic --config-name=vgd_can_guided.yaml
```
Replace the config file above with the desired one. 


## Citation
If you use this code, please cite:
<pre>
@inproceedings{ye2025vgd,
  title     = {Steering Diffusion Policies with Value-Guided Denoising},
  author    = {Ye, Hanming},
  booktitle = {NeurIPS 2025 Workshop on Embodied World Models for Decision Making},
  year      = {2025}
}
</pre>

## Acknowledgements
This repository builds on [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) and [DPPO](https://github.com/irom-princeton/dppo). Parts of code adapted from [DSRL](https://github.com/ajwagen/dsrl)
