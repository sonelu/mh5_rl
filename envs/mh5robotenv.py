import logging
from typing import Callable
import numpy as np
from numpy.typing import NDArray

import mujoco
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import pathlib

# DEFAULT_CAMERA_CONFIG = {
#     "trackbodyid": 1,
#     "distance": 2.0,
#     # "lookat": np.array((0.0, 0.0, 0.25)),
#     "elevation": -20.0,
#     "azimuth": 150,

# }


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
            # this will allow more stability in calculations, but will still
            # be in line with the actual robot 200Hz control cycle
            # dt = timestep * frame_skip = 0.001 * 5 = 0.005 => 200Hz
            timestep: float = 0.001,
            frame_skip: int = 5,
            forward_reward_weight: float = 1.25,
            ctrl_cost_weight: float = 0.1,
            contact_cost_weight: float = 5e-7,
            contact_cost_range: tuple[float, float] = (-np.inf, 10.0),
            healthy_reward: float = 10.0,
            terminate_when_unhealthy: bool = True,
            healthy_z_range: tuple[float, float] = (0.16, 0.25),
            reset_noise_scale: float = 3e-3,
            exclude_current_positions_from_observation: bool = True,
            include_cinert_in_observation: bool = False,
            include_cvel_in_observation: bool = False,
            include_qfrc_actuator_in_observation: bool = False,
            include_cfrc_ext_in_observation: bool = False,
            orientation: Callable[[], NDArray] | None = None,
            # default_camera_config = DEFAULT_CAMERA_CONFIG,
            **kwargs
        ):

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self._include_cinert_in_observation = include_cinert_in_observation
        self._include_cvel_in_observation = include_cvel_in_observation
        self._include_qfrc_actuator_in_observation = include_qfrc_actuator_in_observation
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation

        self._orientation = orientation

        MujocoEnv.__init__(
            self,
            model_path=f"{pathlib.Path(__file__).parent.resolve()}/{model_path}",
            frame_skip=frame_skip,
            observation_space=None,
            # default_camera_config = default_camera_config,
            **kwargs
        )

        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += self.data.cinert[1:].size * include_cinert_in_observation
        obs_size += self.data.cvel[1:].size * include_cvel_in_observation
        obs_size += (self.data.qvel.size - 6) * include_qfrc_actuator_in_observation
        obs_size += self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.model.opt.timestep = timestep

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z
        return is_healthy

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._include_cinert_in_observation is True:
            com_inertia = self.data.cinert[1:].flatten()
        else:
            com_inertia = np.array([])
        if self._include_cvel_in_observation is True:
            com_velocity = self.data.cvel[1:].flatten()
        else:
            com_velocity = np.array([])

        if self._include_qfrc_actuator_in_observation is True:
            actuator_forces = self.data.qfrc_actuator[6:].flatten()
        else:
            actuator_forces = np.array([])
        if self._include_cfrc_ext_in_observation is True:
            external_contact_forces = self.data.cfrc_ext[1:].flatten()
        else:
            external_contact_forces = np.array([])

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            )
        )

    def _get_rew(self, x_velocity: float, z_position: float, action):
        # forward reward only if upright
        if z_position >= self._healthy_z_range[0]:
            forward_reward = self._forward_reward_weight * x_velocity
        else:
            forward_reward = 0.0

        position_reward = self._healthy_reward * z_position
        rewards = forward_reward + position_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        reward_info = {
            "reward_position": position_reward,
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
        }

        return reward, reward_info

    def mass_center(self):
        """Calculate center of mass as weighted average: (model.body_mass.T * data.xipos) / sum(model.body_mass)."""
        num = np.einsum("b,bj->j", self.model.body_mass, self.data.xipos)
        denom = self.model.body_mass.sum()
        return (num / denom)[0:2].copy()

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        return control_cost

    @property
    def contact_cost(self):
        contact_forces = self.data.cfrc_ext
        contact_cost = self._contact_cost_weight * np.sum(np.square(contact_forces))
        min_cost, max_cost = self._contact_cost_range
        contact_cost = np.clip(contact_cost, min_cost, max_cost)
        return contact_cost

    def step(self, action):
        if np.isnan(np.sum(action)):
            logging.getLogger("mh5_env").error(f"action contains nan: {action}")
            raise ValueError(f"action contains nan: {action}")
        xy_position_before = self.mass_center()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.mass_center()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        observation = self._get_obs()
        if np.isnan(np.sum(observation)):
            logging.getLogger("mh5_env").error(f"observation contains nan: {observation}")
            raise ValueError(f"observation contains nan: {observation}")
        reward, reward_info = self._get_rew(x_velocity, observation[2], action)
        if np.isnan(reward):
            logging.getLogger("mh5_env").error(f"reward is nan: {reward}")
            raise ValueError(f"reward is nan: {reward}")
        # terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        # we do not stop the rollout when the robot is down; we want it to learn to stand up
        terminated = False
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
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