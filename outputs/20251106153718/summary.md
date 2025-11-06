# Task: Walking - uncontrolled

Learning for uncontrolled walking (no joystick).

Important differences from other runs:
* learning rate: 1e-4; (trying again with this one although last run stopped because of `nan`s)
* changed `forward_reward_weight` to 5.0 to encourage more exploration in the move forward

Keep an eye:
* the policy might learn to keep the robot frozen upright to accumulate live reward and ignore moving forward; might need to increase the reward for forward moving to compensate for this
