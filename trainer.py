import os
import hydra
from omegaconf import OmegaConf

from tqdm import tqdm
import torch

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.record import VideoRecorder
from torchrl.envs.utils import set_exploration_type, ExplorationType

from torchinfo import summary

from utils import get_device, make_loggers, make_environment, make_model


@hydra.main(version_base="1.2", config_path="config", config_name="baseline.yaml")
def main(cfg):
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger, tb_logger, csv_logger = make_loggers(
        log_dir=output_dir,
        level=cfg.logger.log_level)
    logger.info(f"Using config:\n{OmegaConf.to_yaml(cfg)}\n")
    with open(f"{output_dir}/summary.md", "w") as summary_file:
        summary_file.write(cfg.summary)

    device = get_device(cfg.environment.device)
    logger.info(f"using device: {device}")

    train_env = make_environment(
        name=cfg.environment.robot,
        device=device,
        orientation=cfg.environment.orientation,
        **cfg.environment.kwargs)

    logger.info(f"Training environment:\n{train_env}\n{train_env.spec}")

    eval_env = make_environment(
        name=cfg.environment.robot,
        device=device,
        orientation=cfg.environment.orientation,
        from_pixels=True,
        pixels_only=False,
        **cfg.environment.kwargs)
    eval_env.append_transform(VideoRecorder(csv_logger, "video", fps=cfg.environment.fps))

    logger.info(f"Evaluation environment:\n{eval_env}\n{eval_env.spec}")

    model = make_model(
        policy_net_spec=cfg.model.policy_net_spec,
        value_net_spec=cfg.model.value_net_spec,
        gae_gamma=cfg.model.gae_gamma,
        gae_lambda=cfg.model.gae_lambda,
        env=train_env,
        device=device
    )

    # warm up the lazy layers
    rollout = train_env.rollout(10)
    model['policy'](rollout)
    model['value'](rollout)
    logger.info(f"Policy model:\n{summary(model['policy'], depth=8, verbose=0)}")
    logger.info(f"Value model:\n{summary(model['value'], depth=8, verbose=0)}")

    collector = SyncDataCollector(
        create_env_fn=train_env,
        policy=model['policy'],
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        split_trajs=False,
        device=device,
        use_buffers=False,  # https://github.com/pytorch/rl/issues/3066#issuecomment-3077398138
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=cfg.collector.frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    loss_module = ClipPPOLoss(
        actor_network=model['policy'],
        critic_network=model['value'],
        clip_epsilon=cfg.ClipPPOLoss.clip_epsilon,
        entropy_bonus=bool(cfg.ClipPPOLoss.entropy_eps),
        entropy_coeff=cfg.ClipPPOLoss.entropy_eps,
        critic_coeff=cfg.ClipPPOLoss.critic_coeff,
        loss_critic_type=cfg.ClipPPOLoss.loss_critic_type,
    )

    optim = torch.optim.Adam(loss_module.parameters(), cfg.optim.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optim,
        T_max=(cfg.collector.total_frames // cfg.collector.frames_per_batch),
        eta_min=cfg.scheduler.eta_min
    )


    if cfg.continue_session.from_checkpoint is None:
        logger.info("New training session will be started")
    else:
        logger.info(f"Will continue using checkpoint {cfg.continue_session.from_checkpoint}")
        checkpoint = torch.load(cfg.continue_session.from_checkpoint, weights_only=True)
        model['policy'].load_state_dict(checkpoint['policy'])
        logger.info("policy net loaded from checkpoint")
        model['value'].load_state_dict(checkpoint['value'])
        logger.info("value net loaded from checkpoint")
        model['advantage'].load_state_dict(checkpoint['advantage'])
        logger.info("advantage net loaded from checkpoint")
        loss_module.load_state_dict(checkpoint['loss'])
        logger.info("loss net loaded from checkpoint")
        if not cfg.continue_session.restart_lr:
            optim.load_state_dict(checkpoint['optim'])
            logger.info("optimizer loaded from checkpoint")
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("scheduler loaded from checkpoint")

    pbar = tqdm(total=cfg.collector.total_frames)
    best_eval_sum = 0
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)

    for i, tensordict_data in enumerate(collector):
        pbar.set_description("training  ")
        for _ in range (cfg.training_loop.num_epochs):
            model['advantage'](tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(cfg.collector.frames_per_batch // cfg.training_loop.sub_batch_size):
                subdata = replay_buffer.sample(cfg.training_loop.sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                    loss_vals['loss_objective']
                    + loss_vals['loss_critic']
                    + loss_vals['loss_entropy']
                )
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.ClipPPOLoss.max_grad_norm)
                optim.step()
                optim.zero_grad()

        pbar.update(tensordict_data.numel())

        reward_mean = tensordict_data['next', 'reward'].mean().item()
        step_count = tensordict_data['step_count'].max().item()
        lr = optim.param_groups[0]['lr']
        tb_logger.log_scalar(name="train/reward_mean", value=reward_mean, step=pbar.n)
        csv_logger.log_scalar(name="train/reward_mean", value=reward_mean, step=pbar.n)
        tb_logger.log_scalar(name="train/step_count", value=step_count, step=pbar.n)
        csv_logger.log_scalar(name="train/step_count", value=step_count, step=pbar.n)
        tb_logger.log_scalar(name="train/lr", value=lr, step=pbar.n)
        csv_logger.log_scalar(name="train/lr", value=lr, step=pbar.n)

        scheduler.step()

        if (i + 1) % cfg.training_loop.validate_every_number_of_batches == 0:
            pbar.set_description("validating")
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                eval_env.reset()
                eval_rollout = eval_env.rollout(cfg.environment.kwargs.max_episode_steps, model['policy'])

                reward_mean = eval_rollout['next', 'reward'].mean().item()
                reward_sum = eval_rollout['next', 'reward'].sum().item()
                step_count = eval_rollout['step_count'].max().item()
                tb_logger.log_scalar(name="eval/reward_mean", value=reward_mean, step=pbar.n)
                csv_logger.log_scalar(name="eval/reward_mean", value=reward_mean, step=pbar.n)
                tb_logger.log_scalar(name="eval/reward_sum", value=reward_sum, step=pbar.n)
                csv_logger.log_scalar(name="eval/reward_sum", value=reward_sum, step=pbar.n)
                tb_logger.log_scalar(name="eval/step_count", value=step_count, step=pbar.n)
                csv_logger.log_scalar(name="eval/step_count", value=step_count, step=pbar.n)

                pbar.set_description("saving    ")
                # log z chest position trajectory during the evaluation
                for s, p in enumerate(eval_rollout['observation'][:,0]):
                    tb_logger.log_scalar(name=f"eval_z_pos/{pbar.n}", value=p.item(), step=s)
                    csv_logger.log_scalar(name=f"eval_z_pos/{pbar.n}", value=p.item(), step=s)
                # save the video in all cases
                eval_env.transform[-1].iter = pbar.n
                eval_env.transform[-1].dump()
                # save model checkpoint


                if reward_sum > best_eval_sum:
                    best_eval_sum = reward_sum
                    # save with the step info
                    torch.save({
                        "policy": model['policy'].state_dict(),
                        "value": model['value'].state_dict(),
                        "advantage": model['advantage'].state_dict(),
                        "loss": loss_module.state_dict(),
                        "optimizer": optim.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }, f=f"{output_dir}/checkpoints/valuation_checkpoint_{pbar.n}.pt")
                    # save as best valuation
                    torch.save({
                        "policy": model['policy'].state_dict(),
                        "value": model['value'].state_dict(),
                        "advantage": model['advantage'].state_dict(),
                        "loss": loss_module.state_dict(),
                        "optimizer": optim.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        }, f=f"{output_dir}/checkpoints/valuation_checkpoint_best.pt")

                del eval_rollout

    pbar.set_description("finished  ")

    eval_env.close()
    train_env.close()


if __name__ == "__main__":
    main()
