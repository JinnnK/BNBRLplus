# Belief-Aided-Navigation using Bayesian Neural Networks and Deep Reinforcement Learning for Avoiding Humans in Blind Spots

This repository contains the codes for our paper titled "Belief-Aided-Navigation using Bayesian Neural Networks and Deep Reinforcement Learning for Avoiding Humans in Blind Spots".
The original simulation setting and sourcecode come from [here](https://github.com/JinnnK/TGRF). If you want to see the original version, please refer to the link above.
For more details, here is [arXiv preprint](https://arxiv.org/abs/2403.10105) and [Youtube](https://youtu.be/B7IXV9PRns0?si=-AIeWC6wRPRsBVcS).

## 🔥News
- [06.30] Accepted to IROS 2024!

## Abstract

<p align="center">
<img src="/figures/intro.png" width="700" />
</p>

## Setup
1. In a conda environment or virtual environment with Python 3.x, install the required python package
```
conda env create -f environment.yaml
```
or
```
pip install -r requirements.txt
```

2. Install [OpenAI Baselines](https://github.com/openai/baselines#installation)
```
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```

3. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library

4. Our source code does not work with numpy 1.26.3. Please install version 1.20.3.
```
pip install numpy==1.20.3
```

## Overview
This repository is organized in five parts:
- `crowd_nav/` folder contains configurations and policies used in the simulator.
- `crowd_sim/` folder contains the simulation environment.
- `gst_updated/` folder contains the code for running inference of a human trajectory predictor, named Gumbel Social Transformer (GST) [2].
- `rl/` contains the code for the RL policy networks, wrappers for the prediction network, and ppo algorithm.
- `trained_models/` contains some pretrained models provided by us.

Note that this repository does not include codes for training a trajectory prediction network. Please refer to from [this repo](https://github.com/tedhuang96/gst) instead.

## Run the code
### Training
- Modify the configurations.

  1. PPO and network configurations: modify `arguments.py`
     - `env_name` (must be consistent with `sim.predict_method` in `crowd_nav/configs/config.py`):
        - If you use the GST predictor, set to `CrowdSimPredRealGST-v0`.
        - If you use the ground truth predictor or constant velocity predictor, set to `CrowdSimPred-v0`.
        - If you don't want to use prediction, set to `CrowdSimVarNum-v0`.
     - `use_self_attn`: human-human attention network will be included if set to True, else there will be no human-human attention.
     - `use_hr_attn`: robot-human attention network will be included if set to True, else there will be no robot-human attention.
     - `use_bnn`: Allows you to select whether or not to use BNN. If not using, change it to false.
     - `bnn_policy`: Allows you to choose the type of BNN. Options include `BNBRL+`, `BNBRL`, `BNDNN`, `none`, etc. Please refer to the paper for the characteristics of each neural network.
     - `hidden_dim`, `output_dim`: Allows you to set the number of nodes in the middle and output layers of the BNN.
  2. You can set the characteristics of BNN in `crowd_nav/configs/config.py`.
     - `robot.belief`: Enables or disables belief. If `use_bnn == false`, then it should be set to false.
     - `sim.human_num`: Allows you to set the number of people. Note that the number of people in training and testing must be the same.
     - `sim.belief_radius`: The radius size of belief.
     - `robot.FOV`: The angle of the blind spot. 1 represents 180 degrees.
     - `sim.tracker`: Adds a person who continuously tracks the robot's position.
     - `robot.blink`: Makes the robot unable to obtain information with LiDAR for a certain period of time. The `robot.blink_period` and `robot.blink_time` are fixed values and should not be modified.

- After you change the configurations, run
  ```
  python train.py
  ```
- The checkpoints and configuration files will be saved to the folder specified by `output_dir` in `arguments.py`.

### Testing
Please modify the test arguments in line 20-33 of `test.py` (**Don't set the argument values in terminal!**), and run
```
python test.py
```
Note that the `config.py` and `arguments.py` in the testing folder will be loaded, instead of those in the root directory.
The testing results are logged in `trained_models/your_output_dir/test/` folder, and are also printed on terminal.
If you set `visualize=True` in `test.py`, you will be able to see visualizations.

### Plot the training curves
```
python plot.py
```

## Simulation

### Scenario 1 (No blink, 20 humans)

<p align="center">
<img src="/figures/BNBRL+_case_1.gif" width="350" />
</p>

### Scenario 2 (blink for 0.5 sec, 20 humans)

<p align="center">
<img src="/figures/BNBRL+_case_2.gif" width="350" />
</p>

## Disclaimer
1. I only tested my code in Ubuntu with Python 3.9.16 The code may work on other OS or other versions of Python, but I do not have any guarantee.

2. The performance of my code can vary depending on the choice of hyperparameters and random seeds (see [this reddit post](https://www.reddit.com/r/MachineLearning/comments/rkewa3/d_what_are_your_machine_learning_superstitions/)).
Unfortunately, I do not have time or resources for a thorough hyperparameter search. Thus, if your results are slightly worse than what is claimed in the paper, it is normal.
To achieve the best performance, I recommend some manual hyperparameter tuning.

## Citation
If you find the code or the paper useful for your research, please cite the following papers:
```
@inproceedings{kim2024belief,
  title={Belief Aided Navigation using Bayesian Reinforcement Learning for Avoiding Humans in Blind Spots},
  author={Kim, J. and Kwak, D. and Rim, H. and Kim, D.},
  booktitle={Proceedings of the Conference on AI and Robotics},
  year={2024},
  organization={arXiv preprint arXiv:2403.10105}
}
```

## Credits
[Jinyeob Kim](https://github.com/JinnnK)
Email : wls2074@khu.ac.kr

Part of the code is based on the following repositories:

[1] J. Kim, S. Kang, S. Yang, B. Kim, J. Yura, and D. Kim, “Transformable Gaussian Reward Function for Socially Aware Navigation Using Deep Reinforcement Learning,” Sensors, vol. 24,no. 14, p. 4540, 2024. [Online]. Available: https://doi.org/10.3390/s24144540

[2] P. Chang, N. Chakraborty, and Z. Huang, "Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph," in IEEE International Conference on Robotics and Automation (ICRA), 2023. (Github: https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph)

