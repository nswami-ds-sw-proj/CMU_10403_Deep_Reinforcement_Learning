# 10-403 Assignment 1: Bayesian Optimization w/ Gaussian Processes + CMA-ES

## Gaussian Processes (GP) / Bayesian Optimization
In this assignment we implemented and analyzed Bayesian Optimization via Gaussian Processes, using various acquisition functions. The three acquisition functions used were GP-greedy, GP-UCB (Upper Confidence Bound), and GP-Thompson (based on Thompson Sampling). Attached code implements this process on self-generated datasets, and the average returns over time for the three acquisition functions is plotted as a function of steps taken in the GP. We used a squared exponential kernel (RBF Kernel) to generate the appropriate covariance matrices in this implementation.
## CMA-ES
In the second part of this assignment, we implemented the **Covariance Matrix Adaptation Evolutionary Strategy (CMA-ES)**, which is a black-box, gradient free 
method of optimizing a model. We used this algorithm to train and develop a policy in order to solve the virtual Cartpole-v0 environment (http://gym.openai.com/docs/#environments). We analyzed how this model performed in this environment with variations in starting population size, and compared the average reward over training as a result of these hyperparameter changes. Attached code will train a policy on the environment with initial starting population sizes of 20, 50, and 100.  

# Background / Reference Links
See attached .pdf for writeup

-https://cmudeeprl.github.io/Spring202010403website/assets/lectures/s20_lecture2_bandits_updated.pdf
-https://distill.pub/2019/visual-exploration-gaussian-processes/
-
