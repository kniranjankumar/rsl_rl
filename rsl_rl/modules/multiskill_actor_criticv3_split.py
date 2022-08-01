
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

class Policy(nn.Module):
    def __init__(self,hidden_dims, input_dim, num_actions, activation):
        super(Policy, self).__init__()
        actor_layers = []
        actor_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                pass
                # actor_layers[-1].bias.data.fill_(1)
            else:
                actor_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.base_extractor = nn.ModuleList(actor_layers)
        self.action_means = nn.Linear(hidden_dims[-1], num_actions)
        self.action_stds = nn.Linear(hidden_dims[-1], num_actions)
        self.action_stds.bias.data.fill_(1.0)
    
    def forward(self, obs):
        x = obs
        for layer in self.base_extractor:
            x = layer(x) 
        
        return self.action_means(x), torch.abs(self.action_stds(x))

class Weights(nn.Module):
    def __init__(self, hidden_dims, input_dim, num_skills, activation):
        super(Weights, self).__init__()
        actor_layers = []
        actor_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                actor_layers.append(nn.Linear(hidden_dims[-1], num_skills))
            else:
                actor_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.layers = nn.Sequential(*actor_layers)
        # self.layers.apply(self.weights_init)

    def forward(self, obs): # or goals or whatever
        return nn.functional.softmax(self.layers(obs))    
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            # torch.nn.init.uniform_(m.bias)
            torch.nn.init.constant_(m.bias, 0.5)
                
class MetaBackbone(nn.Module):
    def __init__(self,hidden_dims, input_dim, output_dim, activation):
        super(MetaBackbone, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[-1], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(activation)
        self.base_extractor = nn.Sequential(*layers)

    def forward(self, obs): # or goals or whatever
        return self.base_extractor(obs)
     
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            # torch.nn.init.uniform_(m.bias)
            torch.nn.init.constant_(m.bias, 0.5)

class SynthObsNet(nn.Module):
    
    def __init__(self,hidden_dims, input_dim, output_dim, activation):
        super(SynthObsNet, self).__init__()
        self.self_attn = nn.MultiheadAttention(128,2)
        
    def forward(self, query, key, value):
        
        return self.self_attn(query, key, value)
    
class MultiSkillActorCriticSplit(nn.Module):
    def __init__(self,
                num_skills: List,
                obs_sizes: Dict[str, int],
                actor_obs: Dict,
                critic_obs: List[List[str]], # do we need critic branches?
                meta_network_obs: List[str],
                num_actions: List[int],
                actor_hidden_dims: Dict,
                critic_hidden_dims: Union[List[int], List[List[int]]], # do we need critic branches?
                weight_hidden_dims: Dict,
                meta_backbone_dims: List[int],
                synthetic_obs_scales: Dict[str, int],
                synthetic_obs_ingredients: List[str],
                skill_compositions: Dict[str, List[str]],
                activation: str = "elu",
                init_noise_std: float = 1.0,
                ):
        super(MultiSkillActorCriticSplit, self).__init__()
        self.num_actions = num_actions
        activation = get_activation(activation)
        actor_obs_size = {name:self.get_obs_size(obs_sizes, actor_obs_) for name, actor_obs_ in actor_obs.items()}
        critic_obs_size = [self.get_obs_size(obs_sizes, critic_obs_) for critic_obs_ in critic_obs]
        synthetic_obs_size = self.get_obs_size(obs_sizes, synthetic_obs_scales.keys())
        synth_net_input_size = None# door position, target_position, robot_position, table_position, relative wall corner position?
        self.meta_weight_net_input_size = 128
        self.skill_names = set(actor_hidden_dims.keys()).union(set(weight_hidden_dims.keys()))
        self.actor_branches = {name:Policy(actor_hidden_dims[name], actor_obs_size[name], num_actions, activation) for name in actor_hidden_dims.keys()}
        self.actor = nn.ModuleDict(self.actor_branches) 
        # actor_stds =  [nn.Parameter(init_noise_std * torch.ones(num_actions)) for i, num_action_ in enumerate(num_actions)]
        self.critic = self.create_network(critic_obs_size[-1], 1, critic_hidden_dims, activation)
        # self.weights = Weights([128], actor_obs_size[-1], num_skills, activation)
        self.weight_branches = {name:Weights(weight_hidden_dims[name][:-1], actor_obs_size[name], weight_hidden_dims[name][-1], activation) for name in weight_hidden_dims.keys()}
        self.weights = nn.ModuleDict(self.weight_branches)
        self.synthetic_obs_scales = synthetic_obs_scales
        self.synthetic_obs_size = self.get_obs_size(obs_sizes, list(synthetic_obs_scales.keys()))
        meta_network_obs_size = self.get_obs_size(obs_sizes, meta_network_obs)
        self.meta_backbone = MetaBackbone(meta_backbone_dims, meta_network_obs_size, num_skills, activation)
        self.synthetic_obs_net = SynthObsNet([256,128,64], synth_net_input_size, synthetic_obs_size, activation)
        # Maybe we should feed the computed synthetic observations back in to compute weights?
        self.synthetic_obs_ingredients = synthetic_obs_ingredients
        self.visualize_weights = None
        self.distribution = None
        self.is_recurrent = False
        self.std = None
        self.residual_action_magnitude = 0.
        self.num_skills = num_skills
        self.actor_obs = actor_obs
        self.critic_obs = critic_obs
        self.actor_obs_indices = {key:value for key, value in zip(actor_obs.keys(),self.get_obs_indices(obs_sizes, actor_obs.values()))}
        self.critic_obs_indices = self.get_obs_indices(obs_sizes, critic_obs)
        self.skill_compositions = skill_compositions
        self.synthetic_observation_buffer = {}
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
    def get_obs_size(self,  obs_sizes, obs_names):
        return sum([obs_sizes[obs_name] for obs_name in obs_names])
        
    def get_obs_segment_start_end(self, obs_sizes):
        start = np.cumsum([0]+[length for obs_name, length in obs_sizes.items()][:-1])
        end = np.cumsum([length for obs_name, length in obs_sizes.items()])
        segments = {k: (start_,end_) for k, start_, end_ in zip(obs_sizes.keys(), start, end)}
        return segments
        
    def get_obs_indices(self, obs_sizes, actor_obs):
        segments = self.get_obs_segment_start_end(obs_sizes)
        obs_indices = []
        for actor_obs_ in actor_obs:
            start_end_indices = [segments[obs_name] for obs_name in actor_obs_]
            indices = [range(start,end) for start,end in start_end_indices]
            indices = [item for sublist in indices for item in sublist]
            obs_indices.append(torch.tensor(indices, device=next(self.parameters()).device))
        return obs_indices
        
    def create_network(self, obs_size: int,  output_size:int ,actor_arch: List, activation: Callable):
        network_layers = []
        network_layers.append(nn.Linear(obs_size, actor_arch[0]))
        network_layers.append(activation)
        for l in range(len(actor_arch)):
            if l == len(actor_arch) - 1:
                # pass
                network_layers.append(nn.Linear(actor_arch[l], output_size))
            else:
                network_layers.append(nn.Linear(actor_arch[l], actor_arch[l + 1]))
                network_layers.append(activation)
        network = nn.Sequential(*network_layers)
        return network

    def select_obs(self, observations, name):
        # if name in self.synthetic_obs_scales.keys():
        #     return self.synth_obs[name]*self.synthetic_obs_scales[name]
        # else:
            return torch.index_select(observations, -1, self.actor_obs_indices[name].to(observations.device))
    
    def compute_synthetic_obs_and_metaweights(self, observations):
        weights = self.meta_backbone(observations)
        syth_obs_ingredients = [self.select_obs(observations, name) for name in self.synthetic_obs_ingredients]
        value = torch.stack(synthetic_obs_ingredients, dim=0)
        object_ids = torch.arange(0,len(syth_obs_ingredients)).repeat(observations.size(0),1)
        keys = torch.cat([object_ids,values], dim=-1)
        query = self.select_obs(observations, "target_position")
        synth_obs = self.synthetic_obs_net(query, key, value)
        # weights, synth_obs = torch.split(, [self.num_skills, self.synthetic_obs_size], dim=1)
        # print(synth_obs[0])
        # if len(self.synthetic_obs_scales)>1:
        #     self.synth_obs = {k:v for k,v in zip(self.synthetic_obs_scales.keys(), torch.split(synth_obs, list(self.synthetic_obs_scales.values()), dim=1))}
        # else:
        #     self.synth_obs = synth_obs
        # print(synth_obs.size())
        return nn.functional.softmax(weights, dim=1), nn.functional.normalize(synth_obs)
    
    def combine_skills(self, means, stds, weights):
        """This function combines skills following AMP

        Args:
            means (List): List of means for each skill
            stds (List): List of stds for each skill
        """
        # print(weights.size(), stds.size())
        # stds = stds # If stds become too low, scaled weights blow up
        # scaled_weights = torch.clamp(weights,0.1,0.8)/stds
        # scaled_weights = nn.functional.softmax(weights)/stds
        stds = stds+1e-2
        scaled_weights = weights/stds #+ 1e-2
        mean_skill_weights = scaled_weights.mean(-1)[0]
        
        self.visualize_weights = (mean_skill_weights/mean_skill_weights.sum()).tolist()
        
        # print(scaled_weights.max(), scaled_weights.min(), stds.max(), stds.min(), nn.functional.softmax(self.weights))
        combined_std = 1/scaled_weights.sum(dim=1) 
        combined_mean = combined_std *(means*scaled_weights).sum(dim=1)
        return combined_mean, combined_std

    def update_distribution(self, observations):
        self.instance_weights, synth_obs = self.compute_synthetic_obs_and_metaweights(observations)
        aug_observations = torch.cat([observations, synth_obs], dim=1)
        skill_outputs = {name:branch(self.select_obs(aug_observations, name)) for name, branch in self.actor.items()}
        # skill_means_std = [torch.split(output,self.num_actions,dim=1) for output in skill_outputs]
        weights = {name:weight_net(self.select_obs(aug_observations, name)) for name, weight_net in self.weights.items()}
        skill_means, skill_std = [], []
        for skill_name, compositions_list in self.skill_compositions.items():
            if skill_name in self.weight_branches.keys(): # this skill has multiple branches which should be combined
                chosen_skill_outputs = [skill_outputs[skill_] for skill_ in compositions_list]
                mean_chosen_skill_outputs, std_chosen_skill_outputs = zip(*chosen_skill_outputs)    
                mean, std = self.combine_skills(torch.stack(mean_chosen_skill_outputs,1), torch.stack(std_chosen_skill_outputs,1), torch.stack(self.num_actions*[weights[skill_name]], dim=-1))

            else:
                mean = skill_outputs[skill_name][0]
                std = skill_outputs[skill_name][1]
            skill_means.append(mean)
            skill_std.append(std)   
        # skill_means, skill_std = zip(*skill_outputs)
        self.std = torch.stack(skill_std, 1)
        self.residual_action_magnitude = torch.norm(skill_means[-1],p=2,dim=1).mean()
        # print(self.residual_action)
        # print(self.residual_action_magnitude)
        # self.std = self.get_std_from_logstd(log_std)

        # skill_std = torch.unbind(std,1)
        # skill_std = torch.ReLU(skill_std)
        
        
        combined_mean, combined_std = self.combine_skills(torch.stack(skill_means,1), self.std, torch.stack(self.num_actions*[self.instance_weights], dim=-1))
        # print()
        self.distribution = Normal(combined_mean, combined_std)
    
    def get_std_from_logstd(self, log_std):
        below_threshold = torch.exp(log_std) * (log_std <= 0)
        # Avoid NaN: zeros values that are below zero
        safe_log_std = log_std * (log_std > 0) + 1e-6
        above_threshold = (torch.log1p(safe_log_std) + 1.0) * (log_std > 0)
        std = below_threshold + above_threshold
        return std
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
       
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        self.instance_weights, synth_obs = self.compute_synthetic_obs_and_metaweights(observations)
        aug_observations = torch.cat([observations, synth_obs*3], dim=1)
        # self.instance_weights = self.compute_synthetic_obs_and_metaweights(aug_observations)
        
        skill_outputs = {name:branch(self.select_obs(aug_observations, name)) for name, branch in self.actor.items()}
        # skill_means_std = [torch.split(output,self.num_actions,dim=1) for output in skill_outputs]
        weights = {name:weight_net(self.select_obs(aug_observations, name)) for name, weight_net in self.weights.items()}
        skill_means, skill_std = [], []
        for skill_name, compositions_list in self.skill_compositions.items():
            if skill_name in self.weight_branches.keys(): # this skill has multiple branches which should be combined
                chosen_skill_outputs = [skill_outputs[skill_] for skill_ in compositions_list]
                mean_chosen_skill_outputs, std_chosen_skill_outputs = zip(*chosen_skill_outputs)    
                mean, std = self.combine_skills(torch.stack(mean_chosen_skill_outputs,1), torch.stack(std_chosen_skill_outputs,1), torch.stack(self.num_actions*[weights[skill_name]], dim=-1))
            else:
                mean = skill_outputs[skill_name][0]
                std = skill_outputs[skill_name][1]
            skill_means.append(mean)
            skill_std.append(std)   
        # skill_means, skill_std = zip(*skill_outputs)
        self.std = torch.stack(skill_std, 1)
        self.residual_action_magnitude = torch.norm(skill_means[-1],p=2,dim=1).mean()
        # print(self.residual_action)
        # print(self.residual_action_magnitude)
        # self.std = self.get_std_from_logstd(log_std)

        # skill_std = torch.unbind(std,1)
        # skill_std = torch.ReLU(skill_std)
        
        combined_mean, combined_std = self.combine_skills(torch.stack(skill_means,1), self.std, torch.stack(self.num_actions*[self.instance_weights], dim=-1))
        return combined_mean
        # return skill_outputs[-1][0]

    def evaluate(self, critic_observations, **kwargs):
        self.instance_weights, synth_obs = self.compute_synthetic_obs_and_metaweights(critic_observations)
        aug_observations = torch.cat([critic_observations, synth_obs], dim=1)
        value = self.critic(aug_observations)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None