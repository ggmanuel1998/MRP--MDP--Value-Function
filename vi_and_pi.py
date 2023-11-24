### MDP Value Iteration and Policy Iteration
import argparse
import numpy as np
import gymnasium as gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

parser = argparse.ArgumentParser(
    description="A program to run assignment 1 implementations.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--env",
    type=str,
    help="The name of the environment to run your algorithm on.",
    choices=["Deterministic-4x4-FrozenLake-v0", "Stochastic-4x4-FrozenLake-v0"],
    default="Deterministic-4x4-FrozenLake-v0",
)

parser.add_argument(
    "--render-mode",
    "-r",
    type=str,
    help="The render mode for the environment. 'human' opens a window to render. 'ansi' does not render anything.",
    choices=["human", "ansi"],
    default="human",
)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary of a nested lists
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    value_function = np.zeros(nS)
    while True:
        delta = 0
        for s in range(nS):
            v = 0
            a = policy[s]
            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * value_function[next_state])
            delta = max(delta, np.abs(value_function[s] - v))
            value_function[s] = v
        if delta < tol:
            break
    return value_function



def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    new_policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        q_values = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, done in P[s][a]:
                q_values[a] += prob * (reward + gamma * value_from_policy[next_state])
        new_policy[s] = np.argmax(q_values)
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    policy = np.zeros(nS, dtype=int)
    while True:
        value_function = policy_evaluation(P, nS, nA, policy, gamma, tol)
        new_policy = policy_improvement(P, nS, nA, value_function, policy, gamma)
        if np.all(policy == new_policy):
            break
        policy = new_policy
    return value_function, policy

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    value_function = np.zeros(nS)
    while True:
        delta = 0
        for s in range(nS):
            v = value_function[s]
            value_function[s] = max([sum([prob * (reward + gamma * value_function[next_state])
                                           for prob, next_state, reward, done in P[s][a]])
                                     for a in range(nA)])
            delta = max(delta, np.abs(v - value_function[s]))
        if delta < tol:
            break
    policy = [np.argmax([sum([prob * (reward + gamma * value_function[next_state])
                               for prob, next_state, reward, done in P[s][a]])
                         for a in range(nA)])
              for s in range(nS)]
    return value_function, np.array(policy)

def render_single(env, policy, max_steps=100):
    """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

    episode_reward = 0
    ob, _ = env.reset()
    for t in range(max_steps):
        env.render()
        time.sleep(0.25)
        a = policy[ob]
        ob, rew, done, _, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    env.render()
    if not done:
        print(
            "The agent didn't reach a terminal state in {} steps.".format(
                max_steps
            )
        )
    else:
        print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":
    # read in script argument
    args = parser.parse_args()

    # Make gym environment
    env = gym.make(args.env, render_mode=args.render_mode)

    env.nS = env.nrow * env.ncol
    env.nA = 4

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_pi, 100)

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_vi, 100)
   
    # Imprimir los valores de la función de valor para los primeros tres estados
    print("Valores de la función de valor después de Policy Iteration para los primeros tres estados:")
    print(V_pi[:3])

    print("Valores de la función de valor después de Value Iteration para los primeros tres estados:")
    print(V_vi[:3])

