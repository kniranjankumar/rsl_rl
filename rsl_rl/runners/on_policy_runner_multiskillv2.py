# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO, ResidualPPO
from rsl_rl.modules import *
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner


class MultiSkillOnPolicyRunnerv2(OnPolicyRunner):

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        print(actor_critic_class)
        actor_critic = actor_critic_class(len(self.policy_cfg["skill_compositions"]),                               #num_skills
                                          self.cfg["obs_sizes"],               #obs_sizes
                                          self.cfg["actor_obs"],               #actor_obs
                                          self.cfg["critic_obs"],              #critic_obs
                                          self.cfg["meta_network_obs"],
                                          self.env.num_actions,             #num_actions
                                          **self.policy_cfg).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.load_skills(self.cfg["skill_paths"])
        _, _ = self.env.reset()


    def load_skills(self, paths, load_optimizer=False):
        weight_layer_count = 0
        
        for skill_name, path in paths.items():
            print("loading from ", path)
            loaded_dict = torch.load(path)
            model_state_dict = {}
            weight_net_state_dict = {}
            if skill_name != "door_openv2":
                branches = [int(key[6]) for key in loaded_dict["model_state_dict"].keys() if "actor" in key and key[6].isdigit()]
                if len(branches) > 0:
                    num_branches = max(branches)
                for name, params in loaded_dict["model_state_dict"].items():
                    if "actor" in name:
                        if skill_name != "straight_walk" and skill_name != "standing": # should be 1 if using standing policy
                            # Load just the residual skills if i > 1
                            if "actor."+str(num_branches) in name:
                                name_ = name[8:]
                                model_state_dict[name_] = params
                        else:
                            name_ = name[6:]
                            model_state_dict[name_] = params
                    elif "weights" in name:
                            # print("loading weights", params)
                            weight_net_state_dict[name[8:]] = params
            else:
                for name, params in loaded_dict["model_state_dict"].items():
                    if "actor.residual" in name:
                        model_state_dict[name[15:]] = params
                    elif "meta_backbone" in name:
                        weight_net_state_dict["layers"+name[28:]] = params
                        
            if skill_name in self.alg.actor_critic.actor.keys():
                self.alg.actor_critic.actor[skill_name].load_state_dict(model_state_dict, True)
                for parms in self.alg.actor_critic.actor[skill_name].parameters():
                    parms.requires_grad = False
            if len(weight_net_state_dict.keys()) > 0:
                self.alg.actor_critic.weights[skill_name].load_state_dict(weight_net_state_dict, True)
                weight_layer_count += 1
                for parms in self.alg.actor_critic.weights[skill_name].parameters():
                    parms.requires_grad = False

        # weight_init= len(paths)*[1/len(paths)-0.2]+ (len(self.alg.actor_critic.actor)-len(paths))*[0.01] # this sums up to 1, but may not be necessary
        # self.alg.actor_critic.weights.data = torch.tensor(weight_init, device=self.device)
        return loaded_dict['infos']
        

