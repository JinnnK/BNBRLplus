from . import VecEnvWrapper
import numpy as np
from .running_mean_std import RunningMeanStd
import torch
import os
from collections import deque
import scipy.stats as sc

import copy
import pickle

from gst_updated.src.gumbel_social_transformer.temperature_scheduler import Temp_Scheduler
from gst_updated.scripts.wrapper.crowd_nav_interface_parallel import CrowdNavPredInterfaceMultiEnv

class VecPretextNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that processes the observations and rewards, used for GST predictors
    and returns from an environment.
    config: a Config object
    test: whether we are training or testing
    """

    def __init__(self, venv, ob=False, ret=False, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, config=None, test=False):
        VecEnvWrapper.__init__(self, venv)

        self.config = config
        self.device=torch.device(self.config.training.device)
        if test:
            self.num_envs = 1
        else:
            self.num_envs = self.config.env.num_processes

        self.max_human_num = config.sim.human_num + config.sim.human_num_range

        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = torch.zeros(self.num_envs).to(self.device)
        self.gamma = gamma
        self.epsilon = epsilon
        self.belief = config.robot.belief
        self.predict_steps = config.sim.predict_steps
        self.sensor_range = config.robot.sensor_range
        self.belief_method = config.sim.belief_method
        self.robot_FOV = config.robot.FOV
        self.robot_radius = config.robot.radius
        self.belief_radius = config.sim.belief_radius
        self.human_radius = config.humans.radius
        self.disc_dist = config.reward.discomfort_dist
        self.penalty_intensity = config.reward.penalty_intensity
        self.goal_factor = config.reward.goal_factor
        self.sigma = config.reward.sigma
        self.normalized_factor = sc.norm(0, self.sigma).pdf(0)
        self.collision_penalty = config.reward.collision_penalty
        self.bel_disc_factor = config.reward.belief_discount_factor
        self.blink = config.robot.blink
        self.blink_period = config.robot.blink_period
        self.blink_time = config.robot.blink_time
        self.time_step = config.env.time_step
        blink_cycle_length = int((self.blink_period + self.blink_time) / self.time_step)
        self.blink_cycle = torch.zeros(blink_cycle_length, device=self.device, dtype=torch.bool)
        self.blink_idx = torch.zeros((int(self.num_envs), 1), device=self.device, dtype=torch.long)
        self.prev_mask = torch.zeros((int(self.num_envs), int(self.max_human_num), 1), device=self.device, dtype=torch.long)

        # load and configure the prediction model
        load_path = os.path.join(os.getcwd(), self.config.pred.model_dir)
        if not os.path.isdir(load_path):
            raise RuntimeError('The result directory was not found.')
        checkpoint_dir = os.path.join(load_path, 'checkpoint')
        with open(os.path.join(checkpoint_dir, 'args.pickle'), 'rb') as f:
            self.args = pickle.load(f)

        self.predictor = CrowdNavPredInterfaceMultiEnv(load_path=load_path, device=self.device, config = self.args, num_env = self.num_envs)

        temperature_scheduler = Temp_Scheduler(self.args.num_epochs, self.args.init_temp, self.args.init_temp, temp_min=0.03)
        self.tau = temperature_scheduler.decay_whole_process(epoch=100)

        # handle different prediction and control frequency
        self.pred_interval = int(self.config.data.pred_timestep//self.time_step)
        self.buffer_len = (self.args.obs_seq_len - 1) * self.pred_interval + 1

        self.belief_cnt = torch.Tensor(self.num_envs, self.max_human_num).fill_(0.0).to(self.device)

        self.robot_FOV = self.robot_FOV * np.pi / 2
        self.robot_FOV = self.clamp(self.robot_FOV, -np.pi * 2, np.pi * 2)



    def talk2Env_async(self, data):
        self.venv.talk2Env_async(data)


    def talk2Env_wait(self):
        outs=self.venv.talk2Env_wait()
        return outs

    def step_wait(self):
        obs, rews, done, infos = self.venv.step_wait()

        # process the observations and reward
        obs, rews = self.process_obs_rew(obs, done, rews=rews)

        return obs, rews, done, infos

    def _obfilt(self, obs):
        if self.ob_rms and self.config.RLTrain:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def clamp(self, value, min_val, max_val):
        return max(min_val, min(value, max_val))

    def reset(self):
        # queue for inputs to the pred model
        # fill the queue with dummy values
        self.traj_buffer = deque(list(-torch.ones((self.buffer_len, self.num_envs, self.max_human_num, 2), device=self.device)*15.0),
                                 maxlen=self.buffer_len) # (n_env, num_peds, obs_seq_len, 2)
        self.mask_buffer = deque(list(torch.zeros((self.buffer_len, self.num_envs, self.max_human_num, 1), dtype=torch.bool, device=self.device)),
                                 maxlen=self.buffer_len) # (n_env, num_peds, obs_seq_len, 1)
        self.belief_traj_buffer = deque(list(-torch.ones((self.buffer_len, self.num_envs, self.max_human_num, 2), device=self.device)*15.0),
                                 maxlen=self.buffer_len) # (n_env, num_peds, obs_seq_len, 2)
        self.belief_mask_buffer = deque(list(torch.zeros((self.buffer_len, self.num_envs, self.max_human_num, 1), dtype=torch.bool, device=self.device)),
                                 maxlen=self.buffer_len) # (n_env, num_peds, obs_seq_len, 1)

        self.step_counter = 0

        # for calculating the displacement of human positions
        self.last_pos = torch.zeros(self.num_envs, self.max_human_num, 2).to(self.device)

        self.temp_belief_edges = torch.full((self.num_envs, self.max_human_num, 2 * (self.predict_steps + 1)), 99.0).to(self.device)
        self.prev_belief_edges = torch.full((self.num_envs, self.max_human_num, 2), 99.0).to(self.device)
        self.abs_unsorted_belief_edges = torch.full((self.num_envs, self.max_human_num, 2 * (self.predict_steps + 1)), 99.0).to(self.device)
        self.temp_robot_pose = torch.zeros(self.num_envs, 1, 2).to(self.device)
        self.prev_belief_in_mask_input = torch.full((self.num_envs, self.max_human_num, 1), True, dtype=bool).to(self.device)

        obs = self.venv.reset()
        obs, _ = self.process_obs_rew(obs, np.zeros(self.num_envs))

        blink_cycle_length = int((self.blink_period) / self.time_step)
        self.blink_cycle[:blink_cycle_length] = True
        self.blink_cycle = ~self.blink_cycle
        self.blink_idx[:, :] = -1

        return obs

    def reset_belief_data(self, env_id):
        # reset the data of the env_id
        self.temp_belief_edges[env_id] = 99.0
        self.abs_unsorted_belief_edges[env_id] = 99.0
        self.prev_belief_edges[env_id] = 99.0
        self.prev_belief_in_mask_input[env_id] = True
        self.blink_idx[env_id] = -1.

    def get_angle(self, prev_robot_pose, curr_robot_pose, robot_to_human):
        # robot to human vector
        robot_to_robot = curr_robot_pose - prev_robot_pose

        # angle between robot to human vector and robot's heading vector
        angle = torch.atan2(robot_to_human[:, :, 1], robot_to_human[:, :, 0]) - torch.atan2(robot_to_robot[:, :, 1], robot_to_robot[:, :, 0])

        # convert to [-pi, pi]
        angle[angle > np.pi] -= 2 * np.pi
        angle[angle < -np.pi] += 2 * np.pi

        angle = abs(angle)
        return angle

    '''
    1. Process observations:
    Run inference on pred model with past obs as inputs, fill in the predicted trajectory in O['spatial_edges']

    2. Process rewards:
    Calculate reward for colliding with predicted future traj and add to the original reward,
    same as calc_reward() function in crowd_sim_pred.py except the data are torch tensors
    '''
    def process_obs_rew(self, O, done, rews=0.):
        # O: robot_node: [nenv, 1, 7], spatial_edges: [nenv, observed_human_num, 2*(1+predict_steps)],temporal_edges: [nenv, 1, 2],
        # pos_mask: [nenv, max_human_num], pos_disp_mask: [nenv, max_human_num]
        # prepare inputs for pred_model
        # find humans' absolute positions
        human_pos = O['robot_node'][:, :, :2] + O['spatial_edges'][:, :, :2]

        # insert the new ob to deque
        self.traj_buffer.append(human_pos)
        self.mask_buffer.append(O['visible_masks'].unsqueeze(-1))
        # [obs_seq_len, nenv, max_human_num, 2] -> [nenv, max_human_num, obs_seq_len, 2]
        in_traj = torch.stack(list(self.traj_buffer)).permute(1, 2, 0, 3)
        in_mask = torch.stack(list(self.mask_buffer)).permute(1, 2, 0, 3).float()
        # in_traj : the position of human for 5 steps
        # in_mask : visible == 1, invisible == 0

        # index select the input traj and input mask for GST
        in_traj = in_traj[:, :, ::self.pred_interval]
        in_mask = in_mask[:, :, ::self.pred_interval]

        # forward predictor model
        out_traj, out_mask = self.predictor.forward(input_traj=in_traj, input_binary_mask=in_mask)
        out_mask = out_mask.bool()
        # out_traj is the position of the human to scientific number
        # out_mask is bool of 5 steps

        # add penalties if the robot collides with predicted future pos of humans
        # deterministic reward, only uses mu_x, mu_y and a predefined radius
        # constant radius of each personal zone circle
        # [nenv, human_num, predict_steps]
        hr_dist_future = out_traj[:, :, :, :2] - O['robot_node'][:, :, :2].unsqueeze(1)
        # hr_dist_future is the relative (x,y) = (the position of human - the position of the robot)
        # [nenv, human_num, predict_steps]
        collision_idx = torch.norm(hr_dist_future, dim=-1) < self.robot_radius + self.human_radius
        # collision_idx is the bool of the circle of pred is hit or not.
        # [1,1, predict_steps]
        # mask out invalid predictions
        # [nenv, human_num, predict_steps] AND [nenv, human_num, 1]
        collision_idx = torch.logical_and(collision_idx, out_mask)
        coefficients = 2. ** torch.arange(2, self.config.sim.predict_steps + 2, device=self.device).reshape(
            (1, 1, self.config.sim.predict_steps))  # 4, 8, 16, 32, 64
        # [1, 1, predict_steps]
        collision_penalties = self.config.reward.collision_penalty / coefficients
        # collision_penalties is the table of the penalties
        # [nenv, human_num, predict_steps]
        reward_future = collision_idx.to(torch.float)*collision_penalties
        # reward_future is the table of reward that the robot get
        # [nenv, human_num, predict_steps] -> [nenv, human_num*predict_steps] -> [nenv,]
        # keep the values & discard indices
        reward_future, _ = torch.min(reward_future.reshape(self.num_envs, -1), dim=1)
        # reward_future is the minimum reward that the robot get
        # seems that rews is on cpu
        # rews on the right side is the reward with no prediction.
        # reward_future on the right side is the reward with only the prediction.
        # rews on the left side is merging both of the rewards.
        rews = rews + reward_future.reshape(self.num_envs, 1).cpu().numpy()

        # get observation back to env
        robot_pos = O['robot_node'][:, :, :2].unsqueeze(1)

        # convert from positions in world frame to robot frame
        out_traj[:, :, :, :2] = out_traj[:, :, :, :2] - robot_pos

        # only take mu_x and mu_y
        out_mask = out_mask.repeat(1, 1, self.config.sim.predict_steps * 2)
        new_spatial_edges = out_traj[:, :, :, :2].reshape(self.num_envs, self.max_human_num, -1)
        O['spatial_edges'][:, :, 2:][out_mask] = new_spatial_edges[out_mask]

        if self.belief != False:

            # switch to relative position
            self.temp_belief_edges = self.temp_belief_edges - O['robot_node'][:, :, :2].repeat(1, self.max_human_num, self.predict_steps+1)
            # set the last 2 steps of the temp belief edges to 999.0
            self.temp_belief_edges[:,:,-2:] = 99.0

            # if new episode, reset the some compoenets of that episode
            x_diff = abs(self.temp_robot_pose[: ,: ,0] - O['robot_node'][:, :, 0])
            y_diff = abs(self.temp_robot_pose[: ,: ,1] - O['robot_node'][:, :, 1])

            for i in range(self.num_envs):
                if x_diff[i] > 0.8 or y_diff[i] > 0.8:
                    self.reset_belief_data(i)

            # switch to relative position
            unsorted_belief_edges = self.abs_unsorted_belief_edges - O['robot_node'][:, :, :2].repeat(1, self.max_human_num, self.predict_steps+1)
            #### print(unsorted_belief_edges)
            # shift the (previous) belief edges 2 steps to the left
            unsorted_belief_edges[:,:,:-2] = copy.deepcopy(unsorted_belief_edges[:, :, 2:])

            # from (previous) temp belief edges add NEW DATA to (previous) belief edges
            non_15_indices = (self.temp_belief_edges[:, :, 0] < 50.0)
            non_15_indices = non_15_indices.unsqueeze(-1).repeat(1, 1, 2 * (self.predict_steps + 1))
            unsorted_belief_edges[non_15_indices] = self.temp_belief_edges[non_15_indices]

            # self.blink_cycle
            if self.blink:
                # check blink is True or False
                result = []
                #TODO hard coded now
                self.blink_cycle[11] = True
                temp_idx = copy.deepcopy(self.blink_idx[:, :])
                for env_idx, _ in enumerate(temp_idx):
                    if temp_idx[env_idx, :] == -1:
                        temp_idx[env_idx, :] += 1

                for i in range(self.blink_idx.shape[0]):
                    idx = temp_idx[i]
                    cycle_values = self.blink_cycle[idx]
                    result.append(cycle_values)
                result = torch.stack(result)
                # print(result)

                for env_idx, val in enumerate(result):
                    if val == True:
                        # check the human is visible or not
                        belief_visibility_in_mask = self.get_angle(self.temp_robot_pose, O['robot_node'][:, :, :2],
                                                                    unsorted_belief_edges[:, :, :2]).reshape(1, self.num_envs, self.max_human_num, 1) + 360.0
                        belief_visibility_in_mask = (belief_visibility_in_mask > self.robot_FOV).float()
                        belief_visibility_in_mask = belief_visibility_in_mask.reshape(self.num_envs, self.max_human_num, 1)
                    else:
                        # check the human is visible or not
                        belief_visibility_in_mask = self.get_angle(self.temp_robot_pose, O['robot_node'][:, :, :2],
                                                                    unsorted_belief_edges[:, :, :2]).reshape(1, self.num_envs, self.max_human_num, 1)
                        belief_visibility_in_mask = (belief_visibility_in_mask > self.robot_FOV).float()
                        belief_visibility_in_mask = belief_visibility_in_mask.reshape(self.num_envs, self.max_human_num, 1)

                # update blink_idx
                self.blink_idx[:, :] = self.blink_idx[:, :] + 1
                self.blink_idx[self.blink_idx >= self.blink_cycle.shape[0]] = 0

            else:
                # check the human is visible or not
                belief_visibility_in_mask = self.get_angle(self.temp_robot_pose, O['robot_node'][:, :, :2],
                                                            unsorted_belief_edges[:, :, :2]).reshape(1, self.num_envs, self.max_human_num, 1)
                # print(belief_visibility_in_mask)
                belief_visibility_in_mask = (belief_visibility_in_mask > self.robot_FOV).float()
                belief_visibility_in_mask = belief_visibility_in_mask.reshape(self.num_envs, self.max_human_num, 1)
            # print(belief_visibility_in_mask)
            self.prev_mask = copy.deepcopy(belief_visibility_in_mask)
            unsorted_belief_edges = unsorted_belief_edges * belief_visibility_in_mask + (1 - belief_visibility_in_mask) * 99.0
            # print(unsorted_belief_edges)
            # check out of range of the belief edges
            norms = torch.norm(unsorted_belief_edges[:, :, :2], dim=2)
            unsorted_belief_edges[norms > self.sensor_range + self.robot_radius * 2] = 99.0
            belief_visibility_in_mask[norms > self.sensor_range + self.robot_radius * 2] = 0

            #change the visibility mask to 5 steps
            belief_visibility_in_mask = belief_visibility_in_mask.reshape(1, self.num_envs, self.max_human_num, 1)

            if self.belief_method == 'inferred':
                # Trajectory prediction
                belief_human_pos = O['robot_node'][:, :, :2] + unsorted_belief_edges[:, :, :2]
                belief_in_mask_input = belief_visibility_in_mask.to(torch.bool).reshape(self.num_envs, self.max_human_num, 1)

                self.belief_traj_buffer.append(belief_human_pos)
                self.belief_mask_buffer.append(belief_in_mask_input)

                # update belief_masks
                O['belief_masks'] = belief_in_mask_input.view(self.num_envs, self.max_human_num)

                belief_in_traj = torch.stack(list(self.belief_traj_buffer)).permute(1, 2, 0, 3)
                belief_in_mask = torch.stack(list(self.belief_mask_buffer)).permute(1, 2, 0, 3).float()

                belief_in_traj = belief_in_traj[:, :, ::self.pred_interval]
                belief_in_mask = belief_in_mask[:, :, ::self.pred_interval]
                belief_out_traj, belief_out_mask = self.predictor.forward(input_traj=belief_in_traj, input_binary_mask=belief_in_mask)

                # save in_mask_input for next step
                self.prev_belief_in_mask_input = belief_in_mask_input

            elif self.belief_method == 'const_vel':
                # Trajectory prediction
                # TODO make trajectory prediction with constant velocity
                pass

            else:
                raise NotImplementedError

            # reward for belief
            # belief_count
            self.belief_cnt = self.belief_cnt + O['belief_masks'].float()
            self.belief_cnt = self.belief_cnt * O['belief_masks'].float()

            # reward for belief trajectory
            belief_cnt_traj = self.belief_cnt.view(self.num_envs, self.max_human_num, 1).repeat(1, 1, self.predict_steps)
            belief_cnt_traj = self.bel_disc_factor ** belief_cnt_traj

            # penalty for belief trajectroy
            belief_out_mask = belief_out_mask.bool()
            belief_hr_dist_future = belief_out_traj[:, :, :, :2] - O['robot_node'][:, :, :2].unsqueeze(1)
            belief_collision_idx = torch.norm(belief_hr_dist_future, dim=-1) < self.robot_radius + self.human_radius
            belief_collision_idx = torch.logical_and(belief_collision_idx, belief_out_mask)
            belief_coefficients = 2. ** torch.arange(2, self.config.sim.predict_steps + 2, device=self.device).reshape(
                (1, 1, self.config.sim.predict_steps))
            belief_collision_penalties = self.collision_penalty / belief_coefficients
            belief_reward_future = belief_collision_idx.to(torch.float) * belief_collision_penalties
            belief_reward_future = torch.mul(belief_reward_future, belief_cnt_traj)
            belief_reward_future, _ = torch.min(belief_reward_future.reshape(self.num_envs, -1), dim=1)
            rews = rews + belief_reward_future.reshape(self.num_envs, 1).cpu().numpy()

            # convert from positions in world frame to robot frame
            belief_out_traj[:, :, :, :2] = belief_out_traj[:, :, :, :2] - robot_pos
            # only take mu_x and mu_y
            belief_out_mask = belief_out_mask.repeat(1, 1, self.config.sim.predict_steps * 2)
            belief_new_spatial_edges = belief_out_traj[:, :, :, :2].reshape(self.num_envs, self.max_human_num, -1)
            unsorted_belief_edges[:, :, 2:][belief_out_mask] = belief_new_spatial_edges[belief_out_mask]

            # sort all invisible humans by distance to robot
            belief_hr_dist_cur = torch.linalg.norm(unsorted_belief_edges[:, :, :2], dim=-1)
            belief_sorted_idx = torch.argsort(belief_hr_dist_cur, dim=1)
            for i in range(self.num_envs):
                O['belief_edges'][i] = unsorted_belief_edges[i][belief_sorted_idx[i]]

            # if larger than 50.0, set to 15.0
            O['belief_edges'] = torch.where(O['belief_edges'] > 50.0, torch.full_like(O['belief_edges'], 15.0), O['belief_edges'])

            # save temp robot pose
            self.temp_robot_pose = copy.deepcopy(O['robot_node'][:, :, :2])

            # save data
            self.temp_belief_edges[:,:,:-2] = copy.deepcopy(O['spatial_edges'][:, :, 2:])
            self.temp_belief_edges = torch.where(self.temp_belief_edges > 14.0, torch.full_like(self.temp_belief_edges, 99.0), self.temp_belief_edges)
            self.prev_belief_edges = copy.deepcopy(O['spatial_edges'][:, :, :2])
            self.prev_belief_edges = torch.where(self.prev_belief_edges > 14.0, torch.full_like(self.prev_belief_edges, 99.0), self.prev_belief_edges)

            # switch relative position to absolute position
            self.abs_unsorted_belief_edges = unsorted_belief_edges + O['robot_node'][:, :, :2].repeat(1, self.max_human_num, self.predict_steps+1)
            self.temp_belief_edges = self.temp_belief_edges + self.temp_robot_pose.repeat(1, self.max_human_num, self.predict_steps+1)

        # sort all humans by distance to robot
        # [nenv, human_num]
        hr_dist_cur = torch.linalg.norm(O['spatial_edges'][:, :, :2], dim=-1)
        sorted_idx = torch.argsort(hr_dist_cur, dim=1)
        # sorted_idx = sorted_idx.unsqueeze(-1).repeat(1, 1, 2*(self.config.sim.predict_steps+1))
        for i in range(self.num_envs):
            O['spatial_edges'][i] = O['spatial_edges'][i][sorted_idx[i]]


        if self.belief != False:
            obs={'robot_node':O['robot_node'],
                'spatial_edges':O['spatial_edges'],
                'temporal_edges':O['temporal_edges'],
                'visible_masks':O['visible_masks'],
                'detected_human_num': O['detected_human_num'],
                'belief_edges': O['belief_edges'],
                'belief_masks': O['belief_masks']
            }
        else:
            obs={'robot_node':O['robot_node'],
                'spatial_edges':O['spatial_edges'],
                'temporal_edges':O['temporal_edges'],
                'visible_masks':O['visible_masks'],
                'detected_human_num': O['detected_human_num'],
            }

        self.last_pos = copy.deepcopy(human_pos)
        self.step_counter = self.step_counter + 1

        return obs, rews
