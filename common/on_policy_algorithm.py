import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    # Class variables
    rollout_buffers_list = list()
    policies_list = list()

    # rollout_buffer: RolloutBuffer
    # policy: ActorCriticPolicy

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
        # zhouxch: Multi agent settings
        n_agents: int = 10, 
        n_policies: int = 10, 
        agent_policy_map: Dict = None, 
        n_obs: int = 1, 
        n_act: int = 80, 
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
            # zhouxch: Multi agent settings
            n_agents=n_agents, 
            n_policies=n_policies, 
            agent_policy_map=agent_policy_map, 
            n_obs=n_obs, 
            n_act=n_act, 
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}

        # Multi agent settings
        self.n_agents=n_agents 
        self.n_policies=n_policies 
        self.agent_policy_map=agent_policy_map 
        self.n_obs=n_obs 
        self.n_act=n_act 

        # self.rollout_buffers_list: List[RolloutBuffer]
        # self.policies_list: List[ActorCriticPolicy]
        # self.rollout_buffers_list = list()
        # self.policies_list =  list()

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer
            else:
                self.rollout_buffer_class = RolloutBuffer

        # Init buffers
        for agent_idx in range(self.n_agents):
            # If each policy has its rollout_buffer
            # n_agents_use_this_policy = 0
            # for agent in self.agent_policy_map:
            #     if self.agent_policy_map[agent] == i:
            #         n_agents_use_this_policy += 1
            # print(n_agents_use_this_policy, " agents use policy ", i)

            # If each agent has its rollout_buffer
            rollout_buffer = self.rollout_buffer_class(
                self.n_steps,
                self.observation_space,  # type: ignore[arg-type]
                self.action_space,
                device=self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.n_envs,
                **self.rollout_buffer_kwargs,
            )
            self.rollout_buffers_list.append(rollout_buffer)

        # Init policies
        for policy_idx in range(self.n_policies):
            policy = self.policy_class(  # type: ignore[assignment]
                self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
            )
            policy = policy.to(self.device)
            self.policies_list.append(policy)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffers_list, 
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        for i in range(self.n_agents):
            assert self._last_obs_list[i] is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        for i in range(self.n_policies):
            self.policies_list[i].set_training_mode(False)

        # Reset rollout buffer
        n_steps = 0
        for i in range(self.n_agents):
            rollout_buffers_list[i].reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            for i in range(self.n_policies):
                self.policies_list[i].reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            actions_list = list()
            clipped_actions_list = list()
            values_list = list()
            log_probs_list = list()

            new_obs_list = list()
            rewards_list = list()
            dones_list = list()

            # Sample a new noise matrix if needed
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                for i in range(self.n_policies):
                    self.policies_list[i].reset_noise(env.num_envs)

            # Get action from policy
            for i in range(self.n_agents):
                with th.no_grad():
                    policy_idx = self.agent_policy_map[i]
                    # Convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(self._last_obs_list[i], self.device)
                    actions, values, log_probs = self.policies_list[policy_idx](obs_tensor)
                    
                actions = actions.cpu().numpy()
                actions_list.append(actions)
                values_list.append(values)
                log_probs_list.append(log_probs)

            # Rescale and perform action
            if isinstance(self.action_space, spaces.Box):
                for i in range(self.n_agents):
                    policy_idx = self.agent_policy_map[i]
                    if self.policies_list[policy_idx].squash_output:
                        # Unscale the actions to match env bounds
                        # if they were previously squashed (scaled in [-1, 1])
                        clipped_actions = self.policies_list[policy_idx].unscale_action(actions_list[i])
                    else:
                        # Otherwise, clip the actions to avoid out of bound error
                        # as we are sampling from an unbounded Gaussian distribution
                        clipped_actions = np.clip(actions_list[i], self.action_space.low, self.action_space.high)
                        clipped_actions_list.append(clipped_actions)

            # Agents step forward
            for i in range(self.n_agents):
                new_obs, rewards, dones, infos = env.step(clipped_actions_list[i])
                new_obs_list.append(new_obs)
                rewards_list.append(rewards)
                dones_list.append(dones)
                self._update_info_buffer(infos)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            # Callback
            if not callback.on_step():
                return False

            n_steps += 1

            # Todo: these codes are not used now. They need some modification if used
            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # These codes are important, but it should not have impact on 
            # current performance. 
            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            for i in range(self.n_agents):
                rollout_buffers_list[i].add(
                    self._last_obs_list[i],  # type: ignore[arg-type]
                    actions_list[i],
                    rewards_list[i],
                    self._last_episode_starts,  # type: ignore[arg-type]
                    values_list[i],
                    log_probs_list[i],
                )
                self._last_obs_list[i] = new_obs_list[i]  # type: ignore[assignment]
                self._last_episode_starts = dones_list[i]

        # I don't know if there is some problems
        for i in range(self.n_agents):
            with th.no_grad():
                # Compute value for the last timestep
                policy_idx = self.agent_policy_map[i]
                values = self.policies_list[policy_idx].predict_values(obs_as_tensor(new_obs_list[i], self.device))  # type: ignore[arg-type]

            rollout_buffers_list[i].compute_returns_and_advantage(last_values=values, dones=dones_list[i])

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffers_list, n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    for i in range(self.n_agents):
                        curr_agent_rew_list: List[float] = []
                        curr_agent_len_list: List[float] = []
                        for ep_info in self.ep_info_buffer:
                            if ep_info.get("r" + str(i)) != None:
                                curr_agent_rew_list.append(ep_info["r" + str(i)])
                            if ep_info.get("l" + str(i)) != None:
                                curr_agent_len_list.append(ep_info["l" + str(i)])
                        self.logger.record("rollout/ep_rew_mean" + str(i), safe_mean(curr_agent_rew_list))
                        self.logger.record("rollout/ep_len_mean" + str(i), safe_mean(curr_agent_len_list))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]
        # state_dicts = ["policies_list[" + str(i) + "]" for i in range(15)]
        # state_dicts = ["policies_list"]
        # for i in range(15):
        #     state_dicts.append("policies_list[" + str(i) + "]" + ".optimizer")

        return state_dicts, []
