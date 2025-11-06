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