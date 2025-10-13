from gymnasium.envs.registration import register
from .mh5robotenv import MH5RobotEnv

register(
    id="MH5Robot-v8",
    entry_point=MH5RobotEnv,
    max_episode_steps=1000,
)