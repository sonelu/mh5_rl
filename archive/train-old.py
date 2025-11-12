from collections import defaultdict
import datetime
import time
import json

import torch
import torchrl
import tensordict

from envs.mh5robotenv import MH5RobotEnv

from tensordict.nn import  TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import GymEnv, TransformedEnv, Compose, ObservationNorm, DoubleToFloat, StepCounter
from torchrl.envs.utils import check_env_specs, set_exploration_type, ExplorationType

from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from torchrl.record import TensorboardLogger, VideoRecorder, CSVLogger

from tqdm import tqdm

if __name__ == "__main__":
    print(f"torch version: {torch.__version__}")
    print(f"torchrl version: {torchrl.__version__}")
    print(f"tensordict version: {tensordict.__version__}")

    config = {
        'env_kwargs': {
            'max_episode_steps': 2000,
            'timestep': 0.002,
            # 'frame_skip': 4,
            'healthy_z_range': (0.16, 0.24)
        },
        'num_cells': 512,
        'continue_from_checkpoint': 'logs/20251025195846/best_valuation_checkpoint.pt',
        'continue_training_rate': True,
        'frames_per_batch': 2000,
        'total_frames': 1_000_000,
        'validate_every_number_of_batches': 20,
        'gamma': 0.99,
        'lmbda': 0.95,
        'clip_epsilon': 0.2,
        'entropy_eps': 1e-4,
        'lr': 1e-4,
        'num_epochs': 10,
        'sub_batch_size': 128,
        'max_grad_norm': 1.0,
        # 'timestep': 1.0 / 100.0,
    }



    device = torch.device("cpu")
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    print(f"device: {device}")

    logger_str = f"{datetime.datetime.now():%Y%m%d%H%M%S}"
    logger = TensorboardLogger(exp_name=logger_str, log_dir="logs")
    video_logger = CSVLogger(exp_name=logger_str, log_dir="logs", video_format="mp4")

    print("Using configuration: \n")
    print(json.dumps(config, indent=4), "\n")
    with open(f"logs/{logger_str}/config.json", "w",encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    train_env = TransformedEnv(
        GymEnv("MH5Robot-v8", device=device, **config['env_kwargs']),
        Compose(
            # ObservationNorm(in_keys=['observation']),
            DoubleToFloat(),
            StepCounter(),
        ),
    )

    # train_env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

    check_env_specs(train_env)

    # pre-heat the environment
    rollout = train_env.rollout(3)

    eval_env = TransformedEnv(
        GymEnv("MH5Robot-v8", device=device, from_pixels=True, pixels_only=False, **config['env_kwargs']),
        train_env.transform.clone())
    eval_env.append_transform(VideoRecorder(video_logger, "video", fps=50))

    if config['continue_from_checkpoint'] is not None:
        checkpoint = torch.load(config['continue_from_checkpoint'], weights_only=True)
    else:
        checkpoint = None

    actor_net = nn.Sequential(
        nn.LazyLinear(config['num_cells'], device=device),
        nn.Tanh(),
        nn.LazyLinear(config['num_cells'], device=device),
        nn.Tanh(),
        nn.LazyLinear(config['num_cells'], device=device),
        nn.Tanh(),
        nn.LazyLinear(2 * train_env.action_spec.shape[-1], device=device),
        NormalParamExtractor(),
    )

    policy_module = TensorDictModule(actor_net, in_keys=['observation'], out_keys=['loc', 'scale'])

    policy_module = ProbabilisticActor(
        module=policy_module,
        in_keys=['loc', 'scale'],
        spec=train_env.action_spec,
        distribution_class=TanhNormal,
        distribution_kwargs={
            'low': train_env.action_spec_unbatched.space.low,
            'high': train_env.action_spec_unbatched.space.high,
        },
        return_log_prob=True,
    )

    if checkpoint is not None:
        policy_module.load_state_dict(checkpoint['policy_module'])
        print("restored policy module from checkpoint")

    value_net = nn.Sequential(
        nn.LazyLinear(config['num_cells'], device=device),
        nn.Tanh(),
        nn.LazyLinear(config['num_cells'], device=device),
        nn.Tanh(),
        nn.LazyLinear(config['num_cells'], device=device),
        nn.Tanh(),
        nn.LazyLinear(1, device=device)
    )

    value_module = ValueOperator(
        module=value_net,
        in_keys=['observation'],
    )

    if checkpoint is not None:
        value_module.load_state_dict(checkpoint['value_module'])
        print("restored value module from checkpoint")

    # we need to do this to "initialize" the Lazy modules
    # print("Running policy:", policy_module(train_env.reset()))
    # print("Running value:", value_module(train_env.reset()))
    policy_module(train_env.reset())
    value_module(train_env.reset())

    collector = SyncDataCollector(
        create_env_fn=train_env,
        policy=policy_module,
        frames_per_batch=config['frames_per_batch'],
        total_frames=config['total_frames'],
        split_trajs=False,
        device=device,
        use_buffers=False,  # https://github.com/pytorch/rl/issues/3066#issuecomment-3077398138
    )


    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=config['frames_per_batch']),
        sampler=SamplerWithoutReplacement(),
    )

    advantage_module = GAE(
        gamma=config['gamma'],
        lmbda=config['lmbda'],
        value_network=value_module,
        average_gae=True,
    )

    if checkpoint is not None:
        advantage_module.load_state_dict(checkpoint['advantage_module'])
        print("restored advantage module from checkpoint")

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=config['clip_epsilon'],
        entropy_bonus=bool(config['entropy_eps']),
        entropy_coeff=config['entropy_eps'],
        critic_coeff=1.0,
        loss_critic_type='smooth_l1',
    )

    if checkpoint is not None:
        loss_module.load_state_dict(checkpoint['loss_module'])
        print("restored loss module from checkpoint")

    optim = torch.optim.Adam(loss_module.parameters(), config['lr'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optim,
        T_max=(config['total_frames'] // config['frames_per_batch']),
        eta_min=0.0
    )

    if checkpoint is not None and config['continue_training_rate']:
        optim.load_state_dict(checkpoint['optimizer'])
        print("restored optimizer from checkpoint")
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("restored scheduler from checkpoint")

    pbar = tqdm(total=config['total_frames'])
    best_eval_sum = 0

    for i, tensordict_data in enumerate(collector):
        pbar.set_description("training  ")
        for _ in range (config['num_epochs']):
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(config['frames_per_batch'] // config['sub_batch_size']):
                subdata = replay_buffer.sample(config['sub_batch_size'])
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                    loss_vals['loss_objective']
                    + loss_vals['loss_critic']
                    + loss_vals['loss_entropy']
                )
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), config['max_grad_norm'])
                optim.step()
                optim.zero_grad()

        pbar.update(tensordict_data.numel())

        reward_mean = tensordict_data['next', 'reward'].mean().item()
        step_count = tensordict_data['step_count'].max().item()
        lr = optim.param_groups[0]['lr']
        logger.log_scalar(name="train/reward_mean", value=reward_mean, step=pbar.n)
        logger.log_scalar(name="train/step_count", value=step_count, step=pbar.n)
        logger.log_scalar(name="train/lr", value=lr, step=pbar.n)

        scheduler.step()

        if (i + 1) % config['validate_every_number_of_batches'] == 0:
            pbar.set_description("validating")
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                eval_env.reset()
                eval_rollout = eval_env.rollout(config['env_kwargs']['max_episode_steps'], policy_module)

                reward_mean = eval_rollout['next', 'reward'].mean().item()
                reward_sum = eval_rollout['next', 'reward'].sum().item()
                step_count = eval_rollout['step_count'].max().item()
                logger.log_scalar(name="eval/reward_mean", value=reward_mean, step=pbar.n)
                logger.log_scalar(name="eval/reward_sum", value=reward_sum, step=pbar.n)
                logger.log_scalar(name="eval/step_count", value=step_count, step=pbar.n)

                if reward_sum > best_eval_sum:
                    best_eval_sum = reward_sum
                    pbar.set_description("saving    ")
                    # log z chest position trajectory during the evaluation
                    for s, p in enumerate(eval_rollout['observation'][:,0]):
                        logger.log_scalar(name=f"eval_z_pos/{pbar.n}", value=p.item(), step=s)
                    # save the video
                    eval_env.transform[-1].iter = pbar.n
                    eval_env.transform[-1].dump()
                    # save model checkpoint
                    torch.save({
                        "policy_module": policy_module.state_dict(),
                        "value_module": value_module.state_dict(),
                        "advantage_module": advantage_module.state_dict(),
                        "loss_module": loss_module.state_dict(),
                        "optimizer": optim.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }, f=f"logs/{logger_str}/best_valuation_checkpoint.pt")

                del eval_rollout

    pbar.set_description("finished  ")

    eval_env.close()
    train_env.close()
