# Q-Learning

## Intro

Q-learning is a reinforcement learning technique used in machine learning. The purpose of Q-Learning is to learn a policy, which tells an agent what action to take under certain circumstances. It does not require an environment model and can handle problems with transitions and stochastic rewards without requiring adaptations.

For any finite Markov decision process (FMDP), Q-learning finds a policy that is optimal in that it maximizes the expected value of the total reward over all successive steps from the current state. Q-learning can identify an optimal policy of action selection for any FMDP, given infinite time of exploration and a partly random policy. "Q" names the function that returns the reward used to provide reinforcement and can be considered the "quality" of an action taken in a given state. The function Q will be aproximated by an Residual Neural Networkd in this work.

Used to update the Neural Network, the optimality principle of Bellman is a definition of recursion for an optimal Q function. Q(S(t), A(t)) equals the sum of the immediate reward after performing an action at some time and an expected future reward after a transition to a next state.


Q(S(t),A(t) )←Q(S(t),A(t) )+ α[R(t+1)+γmax(Q(S(t+1),A(t+1)))-Q(S(t),A(t) )]

Algorithm flow

for each iteration:
1. Initialize Neural Networkd to aproximate Q
2. Choose action to take: The action is chosen according to epsilon-greedy strategy
3. Take the action
4. Observe the environment and measure reward
5. Update Neural Network

end

Code in DQN.py

to start training, use

```cmd
python DQN.py train
```