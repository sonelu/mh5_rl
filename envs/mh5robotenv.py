from collections import deque
import logging
from typing import Callable
import numpy as np
from numpy.typing import NDArray

import mujoco
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
import pathlib

MAX_EPISODE_STEPS = 2000

DEFAULT_K_HEALTHY = 5.0
DEFAULT_K_FORWARD = 1.25
DEFAULT_K_CONTROL = 0.02
DEFAULT_K_CONTACT = 5e-5
DEFAULT_K_FALLING = -1000.0

class MH5RobotEnv(MujocoEnv):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
            self,
            model_path: str = "assets/scene.xml",
            timestep: float = 0.005, # dt = timestep * frame_skip = 0.005 * 1 = 0.005 => 200Hz
            frame_skip: int = 1,
            k_healthy: float = DEFAULT_K_HEALTHY,
            k_forward: float = DEFAULT_K_FORWARD,
            k_control: float = DEFAULT_K_CONTROL,
            k_contact: float = DEFAULT_K_CONTACT,
            k_falling: float = DEFAULT_K_FALLING,
            healthy_reward_constant: bool = False,
            terminate_when_unhealthy: bool = True,
            healthy_z_range = (0.15, 0.25),
            contact_cost_range = (-np.inf, 10.0),
            reset_noise_scale: float = 3e-3,
            orientation: Callable[[], NDArray] | None = None,
            **kwargs
        ):

        # self._rew_config = rew_config
        self._k_healthy = k_healthy
        self._k_forward = k_forward
        self._k_control = k_control
        self._k_contact = k_contact
        self._k_falling = k_falling
        self._healthy_reward_constant = healthy_reward_constant
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_cost_range = contact_cost_range
        self._reset_noise_scale = reset_noise_scale
        self._orientation = orientation
        self._steps_taken: int = 0

        MujocoEnv.__init__(
            self,
            model_path=f"{pathlib.Path(__file__).parent.resolve()}/{model_path}",
            frame_skip=frame_skip,
            observation_space=None,
            **kwargs
        )

        # we exclude the position_x and position_y from the observation as these cannot
        # be produced without external positioning sensors and anyway are not relevant
        # for the walking task
        obs_size = self.data.qpos.size + self.data.qvel.size - 2
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)
        self.model.opt.timestep = timestep

    def update_reward_coeff(
            self,
            k_healthy: float = DEFAULT_K_HEALTHY,
            k_forward: float = DEFAULT_K_FORWARD,
            k_control: float = DEFAULT_K_CONTROL,
            k_contact: float = DEFAULT_K_CONTACT
    ) -> None:
        self._k_healthy = k_healthy
        self._k_forward = k_forward
        self._k_control = k_control
        self._k_contact = k_contact

    @property
    def z_pos(self):
        return self.data.qpos[2]

    @property
    def is_healthy(self) -> bool:
        min_z, max_z = self._healthy_z_range
        is_healthy: bool = min_z < self.z_pos < max_z
        return is_healthy

    def _get_obs(self) -> NDArray[np.float64]:
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        obs = np.concatenate((position[2:], velocity), dtype=np.float64)
        return obs

    def _get_rew(self) -> tuple[np.float64, dict[str, NDArray[np.float64]]]:
        # forward reward only if upright
        if self.z_pos >= self._healthy_z_range[0]:
            forward_reward = self._k_forward * self.data.qvel[0]
            if self._healthy_reward_constant:
                healthy_reward = self._k_healthy
            else:
                healthy_reward = self._k_healthy * self._steps_taken / MAX_EPISODE_STEPS
        else:
            forward_reward = 0.0
            healthy_reward = self._k_falling

        rewards = forward_reward + healthy_reward

        control_cost = self.control_cost
        contact_cost = self.contact_cost
        costs = control_cost + contact_cost

        reward = rewards - costs

        reward_info = {
            "reward_healthy": np.array(healthy_reward),
            "reward_forward": np.array(forward_reward),
            "cost_ctrl": np.array(control_cost),
            "cost_contact": np.array(contact_cost),
        }

        return reward, reward_info

    @property
    def control_cost(self):
        control_cost = self._k_control * np.sum(np.square(self.data.ctrl))
        return control_cost

    @property
    def contact_cost(self):
        contact_forces = self.data.cfrc_ext
        contact_cost = self._k_contact * np.sum(np.square(contact_forces))
        min_cost, max_cost = self._contact_cost_range
        contact_cost = np.clip(contact_cost, min_cost, max_cost)
        return contact_cost

    def step(self, action):
        if np.isnan(np.sum(action)):
            logging.getLogger("mh5_env").error(f"action contains nan: {action}")
            raise ValueError(f"action contains nan: {action}")
        self.do_simulation(action, self.frame_skip)
        self._steps_taken += 1
        x_velocity, y_velocity = self.data.qvel[:2]
        observation = self._get_obs()

        if np.isnan(np.sum(observation)):
            logging.getLogger("mh5_env").error(f"observation contains nan: {observation}")
            raise ValueError(f"observation contains nan: {observation}")

        reward, reward_info = self._get_rew()

        if np.isnan(reward):
            logging.getLogger("mh5_env").error(f"reward is nan: {reward}")
            raise ValueError(f"reward is nan: {reward}")

        terminated: bool = (not self.is_healthy) and self._terminate_when_unhealthy

        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            # "tendon_length": self.data.ten_length,
            # "tendon_velocity": self.data.ten_velocity,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos
        if self._orientation is not None:
            orient = self._orientation()
            qpos[3:7] = orient
        qpos = qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        self._steps_taken = 0
        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            # "tendon_length": self.data.ten_length,
            # "tendon_velocity": self.data.ten_velocity,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }