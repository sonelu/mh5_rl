# import os
import hydra
import torchrl.trainers.algorithms.configs
from omegaconf import OmegaConf


from envs import MH5RobotEnv

from tqdm import tqdm

import torch
import torch.nn as nn


from torchrl.envs import TransformedEnv
from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.trainers.trainers import LogScalar, Trainer
# from torchrl.trainers.algorithms.ppo import PPOTrainer

# from torchrl.collectors import SyncDataCollector
# from torchrl.data.replay_buffers import ReplayBuffer
# from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
# from torchrl.data.replay_buffers.storages import LazyTensorStorage
# from torchrl.objectives import ClipPPOLoss
from torchrl.envs import default_info_dict_reader
from torchrl.record import VideoRecorder

# from torchinfo import summary

class Validation:
    def __init__(
            self,
            trainer: Trainer,
            frequency: int,
            validation_env: TransformedEnv,
            policy_model: nn.Module
    ):
        self.trainer = trainer
        self.logger = self.trainer.logger
        self.frequency = frequency
        self.validation_env = validation_env
        self.policy_model = policy_model
        self.best_eval_sum = 0

    def __call__(self) -> None:
        step = self.trainer.collected_frames
        if step > 0 and step % self.frequency == 0:
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                val_rollout = self.validation_env.rollout(2000, self.policy_model)
                self.logger.log_scalar(name="validation(mean)/reward", value=val_rollout['next', 'reward'].mean().item(), step=step)
                self.logger.log_scalar(name="validation(sum)/reward", value=val_rollout['next', 'reward'].sum().item(), step=step)
                self.logger.log_scalar(name="validation/step_count", value=val_rollout['step_count'].max().item(), step=step)
                for measure in ['reward_position', 'reward_forward', 'reward_ctrl', 'reward_contact']:
                    self.logger.log_scalar(name=f"validation(mean)/{measure}", value=val_rollout['next', measure].mean().item(), step=step)
                    self.logger.log_scalar(name=f"validation(sum)/{measure}", value=val_rollout['next', measure].sum().item(), step=step)
                # we need to update `iter` in VideoRecorder otherwise it will be used as step in the log and wandb will complain that
                # is out of sync with the current step
                self.validation_env.transform[-1].iter = step
                self.validation_env.transform[-1].dump()
                if val_rollout['step_count'].sum().item() > self.best_eval_sum:
                    self.best_eval_sum = val_rollout['step_count'].sum().item()
                    self.trainer._save_trainer()

                del val_rollout


@hydra.main(version_base="1.2", config_path="config", config_name="config.yaml")
def main(cfg):
    # use the Hydra output directory for all outputs
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # output directory for wandb logging
    cfg.logger['save_dir'] = output_dir
    # save checkpoint location updated with the Hydra directory
    cfg.trainer['save_trainer_file'] = output_dir + '/' + cfg.trainer['save_trainer_file']

    trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.logger.log_str(name="trainer config", value=str(trainer))

    # include in the TensorDict the detail rewards for logging
    trainer.collector.env.set_info_dict_reader(default_info_dict_reader(['reward_position', 'reward_forward', 'reward_ctrl', 'reward_contact']))

    # setup the evaluation environment with VideoRecorder and extra reward info
    evaluation_env = hydra.utils.instantiate(cfg.environments.evaluation_env)
    evaluation_env.append_transform(VideoRecorder(trainer.logger, "video", fps=100))
    evaluation_env.set_info_dict_reader(default_info_dict_reader(['reward_position', 'reward_forward', 'reward_ctrl', 'reward_contact']))

    # disable the standard logging
    trainer._pre_steps_log_ops = []
    # register new logging hooks
    log_train_rew_mean = LogScalar(key=("next", "reward"), logname="training(mean)/reward", log_pbar=False, include_std=False, reduction="mean")
    trainer.register_op("pre_steps_log", log_train_rew_mean)
    log_train_rew_sum = LogScalar(key=("next", "reward"), logname="training(sum)/reward", log_pbar=False, include_std=False, reduction="sum")
    trainer.register_op("pre_steps_log", log_train_rew_sum)

    for measure in ['reward_position', 'reward_forward', 'reward_ctrl', 'reward_contact']:
        trainer.register_op("pre_steps_log", LogScalar(
            key=("next", measure), logname=f"training(mean)/{measure}", log_pbar=False, include_std=False, reduction="mean"
        ))
        trainer.register_op("pre_steps_log", LogScalar(
            key=("next", measure), logname=f"training(sum)/{measure}", log_pbar=False, include_std=False, reduction="sum"
        ))

    log_train_steps_max = LogScalar(key=("step_count"), logname="training/step_count", log_pbar=False, include_std=False, reduction="max")
    trainer.register_op("pre_steps_log", log_train_steps_max)

    # register validation hook
    trainer.register_op("post_steps", Validation(trainer, 100_000, evaluation_env, trainer.collector.policy))

    trainer.train()


if __name__ == "__main__":
    main()
