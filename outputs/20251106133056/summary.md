# Task: Walking - uncontrolled

Learning for uncontrolled walking (no joystick).

Important differences from other runs:
* learning rate: 1e-4; (lr of 1e-3, 5e-4, 2.5e-4 are unstable and results in nans)

Keep an eye:
* the policy might learn to keep the robot frozen upright to accumulate live reward and ignore moving forward; might need to increase the reward for forward moving to compensate for this

Issues:
* still issues `nan` after 2.5M timesteps (25% of training). Will attempt to reduce the LR
* there is a significant portion of training where the robot learns to just freeze in position and run a full 2000 steps episode with 5.0 reward for surving. Most likelly the weight for forward movement needs to be increased to make it worth exploring
* To understand why the `nan`s appear we might need to log the loss values
