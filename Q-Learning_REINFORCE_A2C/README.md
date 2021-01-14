DQN Implementation

To change any of the hyperparameters, edit them in the __init__ function for QNetwork.
To turn action repeat off, set action_repeat to 1.
To turn off target updates, set target_update to 1.
To automatically sample the last transitions from the most recent episode, set live_update > 0.
If sample_scalar is not None, then transitions with rewards greater than the minimum will be sampled with a probability sample_scalar times higher than normal transitions.

The --env flag should be the name of the environment, ie CartPole-v0 or MountainCar-v0.
The --render flag will render a training episode every 50 episodes.
The --train flag will train the model, saving the weights every 1000 episodes. If the --train flag is 0, the test reward will be printed from the agent instead.
The --model flag should point to a model weights file, with the extension .h5, which will be loaded into the model in the agent.

A random seed and keras random seed are set for the environment, so all experiments should be repeatable.

