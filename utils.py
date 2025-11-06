import datetime
import logging
from typing import Callable

import numpy as np
from numpy.typing import NDArray

import mujoco

import torch
import torch.nn as nn

from tensordict.nn import TensorDictModule, TensorDictModuleBase
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.envs import TransformedEnv, GymEnv, Compose, DoubleToFloat, StepCounter
from torchrl.record import TensorboardLogger, CSVLogger

from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives.value import GAE

from envs import MH5RobotEnv, DEFAULT_REW_CONFIG, DEFAULT_OBS_CONFIG


def get_device(device: str = "cpu") -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            logging.getLogger("trainer").warning("cuda is not available, will default to cpu")
            return torch.device("cpu")
    if device == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            logging.getLogger("trainer").warning("mps is not available, defaulting to cpu")
    # default to avoid warnings
    return torch.device("cpu")


def make_loggers(log_dir: str, level: str = "INFO") -> tuple[logging.Logger, TensorboardLogger, CSVLogger]:
    # directory name for the output
    # logger_str = f"{datetime.datetime.now():%Y%m%d%H%M%S}"

    # tensorboard and CSV (for video) logger
    # this will also create the directory

    # message logger
    logger = logging.getLogger("trainer")
    logger.setLevel(level)
    # file_handler = logging.FileHandler(f"logs/{logger_str}/logger.log")
    # file_handler.setLevel(level)
    # file_handler.setFormatter(_CustomFormatter())
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(level)
    # console_handler.setFormatter(_CustomFormatter())
    # logger.addHandler(file_handler)
    # logger.addHandler(console_handler)
    tb_logger = TensorboardLogger(exp_name="", log_dir=log_dir)
    csv_logger = CSVLogger(exp_name="", log_dir=log_dir, video_format="mp4")

    return (logger, tb_logger, csv_logger)


def make_environment(name: str, device: torch.device, orientation: str = "standing", **kwargs) -> TransformedEnv:
    orient_fun = get_orientation_func(orientation)
    kwargs['orientation'] = orient_fun
    rew_config = DEFAULT_REW_CONFIG
    rew_config.update(kwargs.get('rew_config', {}))
    kwargs['rew_config'] = rew_config
    obs_config = DEFAULT_OBS_CONFIG
    obs_config.update(kwargs.get('obs_config', {}))
    kwargs['obs_config'] = obs_config
    return TransformedEnv(
        GymEnv(name, device=device, **kwargs),
        Compose(
            # ObservationNorm(in_keys=['observation']),
            DoubleToFloat(),
            StepCounter(),
        )
    )


def make_model(
        policy_net_spec: list[int],
        value_net_spec: list[int],
        gae_gamma: float,
        gae_lambda: float,
        env: TransformedEnv,
        device: torch.device
) -> dict[str, ProbabilisticActor | ValueOperator | GAE]:
    # Returns policy
    actor_net = nn.Sequential()
    for i, num_cells in enumerate(policy_net_spec):
        actor_net.add_module(f"Layer-{i}", nn.LazyLinear(num_cells, device=device))
        actor_net.add_module(f"Tanh-{i}", nn.Tanh())
    actor_net.add_module("Output", nn.LazyLinear(2 * env.action_spec.shape[-1], device=device))
    actor_net.add_module("Normal-Param-Extractor", NormalParamExtractor())

    policy_module = TensorDictModule(actor_net, in_keys=['observation'], out_keys=['loc', 'scale'])

    policy_module = ProbabilisticActor(
        module=policy_module,
        in_keys=['loc', 'scale'],
        spec=env.action_spec,
        distribution_class=TanhNormal,
        distribution_kwargs={
            'low': env.action_spec_unbatched.space.low,
            'high': env.action_spec_unbatched.space.high,
        },
        return_log_prob=True,
    )

    value_net = nn.Sequential()
    for i, num_cells in enumerate(value_net_spec):
        value_net.add_module(f"Layer-{i}", nn.LazyLinear(num_cells, device=device))
        value_net.add_module(f"Tanh-{i}", nn.Tanh())
    value_net.add_module("Output", nn.LazyLinear(1, device=device))

    value_module = ValueOperator(
        module=value_net,
        in_keys=['observation'],
    )

    advantage_module = GAE(
        gamma=gae_gamma,
        lmbda=gae_lambda,
        value_network=value_module,
        average_gae=True,
    )

    return {
        'policy': policy_module,
        'value': value_module,
        'advantage': advantage_module,
    }

def orientation_standing() -> NDArray[np.float32]:
    return np.array([1.0, 0.0, 0.0 , 0.0])

def orientation_back() -> NDArray[np.float32]:
    orient = np.array([1.0, 0.0, 0.0 , 0.0])
    mujoco.mju_euler2Quat(orient, [0.0, -np.pi/2, 0.0], "xyz")
    return orient

def orientation_front() -> NDArray[np.float32]:
    orient = np.array([1.0, 0.0, 0.0 , 0.0])
    mujoco.mju_euler2Quat(orient, [0.0, np.pi/2, 0.0], "xyz")
    return orient

def get_orientation_func(orientation: str) -> Callable[[], NDArray[np.float32]]:
    if orientation =="standing":
        return orientation_standing
    if orientation =="back":
        return orientation_back
    if orientation == "front":
        return orientation_front
    # default
    logging.getLogger("utils").warning(f"orientation {orientation} not supported, will default to standing")
    return orientation_standing