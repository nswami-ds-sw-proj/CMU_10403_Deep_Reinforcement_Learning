



In this assignment, we implemented Model Based Reinforcement Learning (MBRL) , with Probabilistic Ensembles w/ Trajectory Sampling. This is a newer approach to more complex
reinforcement learning tasks, where the agent attempts to compensate for its lack of prior knowledge, by training it to be able to model and functionally 
represent the dynamics of the environment, so that the agent can predict the result of taking a specific action in the environment and act according to such predictions. 
This is accomplished in this implementation via using ensembles of probabilistic networks that output a distribution over given state-action pairs, training them via
propagating hallucinated trajectories through various networks in the ensemble, and planning actions via the use of this ensembled accompanied with a cost function. 

In 




