# Task: Walking - uncontrolled

Learning for uncontrolled walking (no joystick).

Important differences from other runs:
* learning rate: 1e-5 (attempting to see if the `nan` was fixed by changes to the env handling of z_position)
* `forward_reward_weight`: 2.5 to encourage more exploration in the move forward
* `sub-sub_batch_size`: 64 (double the number of training backprops to compensate for smaller LR)
* `num_epochs`: 10 (to reduce the excessive training time)

Keep an eye:
* the policy might learn to keep the robot frozen upright to accumulate live reward and ignore moving forward; might need to increase the reward for forward moving to compensate for this

Issues:
* At 3M frames the robot learns to freeze in position and stay there for the duration of the episode. There is no movement forward.