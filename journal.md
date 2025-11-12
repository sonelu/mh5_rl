# [2025/11/06-13:30:56](20251106133056/trainer.log) Task: Walking - uncontrolled

Learning for uncontrolled walking (no joystick).

Important differences from other runs:
* learning rate: 1e-4; (lr of 1e-3, 5e-4, 2.5e-4 are unstable and results in `nan`s)

Keep an eye:
* the policy might learn to keep the robot frozen upright to accumulate live reward and ignore moving forward; might need to increase the reward for forward moving to compensate for this

Issues:
* still issues `nan` after 2.5M timestep (25% of training). Will attempt to reduce the LR
* there is a significant portion of training where the robot learns to just freeze in position and run a full 2000 steps episode with 5.0 reward for surviving. Most likely the weight for forward movement needs to be increased to make it worth exploring
* To understand why the `nan`s appear we might need to log the loss values

# [2025/11/06-15:37:18](20251106153718/trainer.log) Task: Walking - uncontrolled

Learning for uncontrolled walking (no joystick).

Important differences from other runs:
* learning rate: 1e-4; (trying again with this one although last run stopped because of `nan`s)
* changed `forward_reward_weight` to 5.0 to encourage more exploration in the move forward

Keep an eye:
* the policy might learn to keep the robot frozen upright to accumulate live reward and ignore moving forward; might need to increase the reward for forward moving to compensate for this

Issues:
* still issues `nan`s after 660k steps. Will have to reduce the LR and compensate with smaller sub-batch and possibly number of epoch increase to stimulate training

# [~2025/11/06-16:31:50] Task: Walking - uncontrolled

Learning for uncontrolled walking (no joystick).

Important differences from other runs:
* learning rate: 1e-5
* `forward_reward_weight`: 5.0 to encourage more exploration in the move forward
* `sub-sub_batch_size`: 64 (double the number of training backprops to compensate for smaller LR)
* `num_epochs`: 20 (double the number of training backprops to compensate for the smaller LR)

Keep an eye:
* the policy might learn to keep the robot frozen upright to accumulate live reward and ignore moving forward; might need to increase the reward for forward moving to compensate for this

Issues:
* training too slow; stopped at 2M step as it's not progressing well
* it is possible that the `nan` issues were due to the incorrect environment setup that was fixed [here](https://github.com/sonelu/mh5_rl/commit/401d2b08feeffeaa9e5c985523e5890cfc699b2a)


# [~2025/11/06-18:35:22] Task: Walking - uncontrolled

Learning for uncontrolled walking (no joystick).

Important differences from other runs:
* learning rate: 2e-5 (attempting to see if the `nan` was fixed by changes to the env handling of z_position)
* `forward_reward_weight`: 5.0 to encourage more exploration in the move forward
* `sub-sub_batch_size`: 64 (double the number of training backprops to compensate for smaller LR)
* `num_epochs`: 10 (to reduce the excessive training time)

Keep an eye:
* the policy might learn to keep the robot frozen upright to accumulate live reward and ignore moving forward; might need to increase the reward for forward moving to compensate for this

Issues:
* failed after aprox 4M steps
* doesn't seem to learn after 2.5M steps

# [2025/11/06-22:07:16](20251106220716/trainer.log) Task: Walking - uncontrolled

Learning for uncontrolled walking (no joystick).

Important differences from other runs:
* learning rate: 2e-5 (attempting to see if the `nan` was fixed by changes to the env handling of z_position)
* `forward_reward_weight`: 5.0 to encourage more exploration in the move forward
* `sub-sub_batch_size`: 64 (double the number of training backprops to compensate for smaller LR)
* `num_epochs`: 10 (to reduce the excessive training time)

Keep an eye:
* the policy might learn to keep the robot frozen upright to accumulate live reward and ignore moving forward; might need to increase the reward for forward moving to compensate for this

Issues:
* At 3M frames the robot learns to freeze in position and stay there for the duration of the episode. There is no movement forward.