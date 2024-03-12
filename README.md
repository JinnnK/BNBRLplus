# Belief-Aided-Navigation using Bayesian Neural Networks and Deep Reinforcement Learning for Avoiding Humans in Blind Spots

This repository contains the codes for our paper titled "Transformable Gaussian Reward Function for Robot Navigation with Deep Reinforcement Learning".
The original simulation setting and sourcecode come from [here](https://sites.google.com/view/intention-aware-crowdnav/home). If you want to see the original version, please refer to the link above.
For more details, here is [arXiv preprint]() and [youtube video]() for experiment in real world.

## Abstract

<p align="center">
<img src="/figures/intro.png" width="550" />
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

Our source code does not work with numpy 1.26.3. Please install version 1.20.3.

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
  1. Environment configurations: Modify `crowd_nav/configs/config.py`. Especially,
     - Choice of type of danger zone:
       - Set `reward.danger_zone_type = 'gaussian'` if my gaussian model is used. It is available for any type of options below. It penalizes the robot depending on the distance between the robot and human following the folumla in the paper. And you can customize your own reward function by adjsuting some hyperparameters in `crowd_nav/configs/config.py`.
         - `reward.discomfort_dist` : The range of discomfort distance. If distance between the robot and human is lower than discomfort distance, it will penalize following the fomula in the paper.
         - `reward.gamma` : Discount Factor.
         - `reward.sigma` : Sigma of gaussian distribution. You can adjust the sigma of gaussian distribution by adjusting this.
         - `reward.penalty_intensity` : The intensity of gaussian distribution. This factor is muliplied with the value of gaussian distribution. If you set this as 1, the discomfort distance penalty will penalize upto the value of `reward.collision_penalty`.
         - `reward.goal_factor` : The parameter of potential reward. It is multiplied with the potential reward linearly and make the robot get to the goal as soon as possible. But also goes up the porbability of collision.
         - `reward.success_reward` : Reward when the robot get to the goal. You can adjust this value, but I do not recommand to adjust this. Because the balance between other value is more important.
         - `reward.collision_penalty` : Reward when the robot collises with human. You can adjust this value, but I do not recommand to adjust this. Because the balance between other value is more important.

       - Set `reward.danger_zone_type = 'future'`. It is a type which the previous paper used. It is only compatible with `robot.policy = 'selfAttn_merge_srnn'`. If you want to see the fomula of this reward model, please refer to [the paper]() or the [previous paper]()
         - `reward.success_reward`, `reward.collision_penalty`, `reward.discomfort_dist`, and `reward.gamma`are same.
         - `env.time_step` : time step of sampling. You can adjust how many sample you want to see. It affect the reward of discomfort distance.
       - Set `reward.danger_zone_type = 'circle'` : It is a type which the more previous paper used. It is only compatible with `robot.policy = 'srnn'`, DS-RNN.
     - Choice of policy of the robot:
       - Set `robot.policy = 'selfAttn_merge_srnn'` if you want to use GST predictor and HH Attn.
       - Set `robot.policy = 'srnn'` if you want to use DS-RNN.

     - Choice of human trajectory predictor:
       - Set `sim.predict_method = 'inferred'` if a learning-based GST predictor is used [2]. Please also change `pred.model_dir` to be the directory of a trained GST model. We provide two pretrained models [here](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph/tree/main/gst_updated/results/).
       - Set `sim.predict_method = 'const_vel'` if constant velocity model is used.
       - Set `sim.predict_method = 'truth'` if ground truth predictor is used.
       - Set `sim.predict_method = 'none'` if you do not want to use future trajectories to change the observation and reward.

  2. PPO and network configurations: modify `arguments.py`
     - `env_name` (must be consistent with `sim.predict_method` in `crowd_nav/configs/config.py`):
        - If you use the GST predictor, set to `CrowdSimPredRealGST-v0`.
        - If you use the ground truth predictor or constant velocity predictor, set to `CrowdSimPred-v0`.
        - If you don't want to use prediction, set to `CrowdSimVarNum-v0`.
     - `use_self_attn`: human-human attention network will be included if set to True, else there will be no human-human attention.
     - `use_hr_attn`: robot-human attention network will be included if set to True, else there will be no robot-human attention.

  3. (belief) Will be added

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

### Evaluation

<p align="center">
<img src="/figures/evaluation.png" width="550" />
</p>

## Disclaimer
1. I only tested my code in Ubuntu with Python 3.9.16 The code may work on other OS or other versions of Python, but I do not have any guarantee.

2. The performance of my code can vary depending on the choice of hyperparameters and random seeds (see [this reddit post](https://www.reddit.com/r/MachineLearning/comments/rkewa3/d_what_are_your_machine_learning_superstitions/)).
Unfortunately, I do not have time or resources for a thorough hyperparameter search. Thus, if your results are slightly worse than what is claimed in the paper, it is normal.
To achieve the best performance, I recommend some manual hyperparameter tuning.

## Citation
If you find the code or the paper useful for your research, please cite the following papers:
```
@inproceedings{,
  title={Belief-Aided-Navigation using Bayesian Neural Networks and Deep Reinforcement Learning for Avoiding Humans in Blind Spots},
  author={Jinyeob Kim},
  booktitle={},
  year={2024}
}

@inproceedings{,
  title={Transformable Gaussian Reward Function for Robot Navigation with Deep Reinforcement Learning},
  author={Jinyeob Kim},
  booktitle={},
  year={2024}
}

@inproceedings{liu2022intention,
  title={Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph},
  author={Liu, Shuijing and Chang, Peixin and Huang, Zhe and Chakraborty, Neeloy and Hong, Kaiwen and Liang, Weihang and Livingston McPherson, D. and Geng, Junyi and Driggs-Campbell, Katherine},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023}
}
```

## Credits
[Jinyeob Kim](https://github.com/JinnnK)
Email : wls2074@khu.ac.kr

Part of the code is based on the following repositories:

[1] P. Chang, N. Chakraborty, and Z. Huang, "Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph," in IEEE International Conference on Robotics and Automation (ICRA), 2023. (Github: https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph)

[2] Z. Huang, R. Li, K. Shin, and K. Driggs-Campbell. "Learning Sparse Interaction Graphs of Partially Detected Pedestrians for Trajectory Prediction," in IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 1198–1205, 2022. (Github: https://github.com/tedhuang96/gst)

