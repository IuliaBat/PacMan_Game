# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from pacman import Directions
from game import Agent
import random
import game
import util
import numpy as np
from collections import defaultdict

#######################
# QLearnAgent class which implements reinforcement learning (Q-learning) in order to play Pacman #
#######################

class QLearnAgent(Agent):

    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining = 10):
        """
        Constructor, called when we start running the game
        :param alpha: learning rate (default: 0.2)
        :param epsilon: exploration rate (default: 0.05)
        :param gamma: discount factor (default: 0.8)
        :param numTraining: number of training episodes (default: 10)

        These values are either passed from the command line or are
        set to the default values above. We need to create the set variables from them.
        """

        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        # table Q(s,a) as a nested dictionary: Q-values of all (state,action) pairs
        self.possible_actions = ['North', 'East', 'South', 'West']
        self.QValues = defaultdict(lambda: np.zeros(len(self.possible_actions)))
        
        # previous s,a,r (state, action, reward), initially null
        self.pacman_state = None
        self.pacman_action = None
        self.pacman_reward = None

    #######################
    # Accessor functions for the variable episodesSoFars controlling learning #
    #######################

    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
            return self.numTraining

    #######################
    # Accessor functions for parameters #
    #######################

    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value
        
    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    #######################
    # Functions for previous s,a,r #
    #######################

    def set_State(self, state):
        self.pacman_state = state

    def get_State(self):
        return self.pacman_state

    def set_Action(self, action):
        self.pacman_action = action

    def get_Action(self):
        return self.pacman_action

    def set_Reward(self, score):
        self.pacman_reward = score

    def get_Reward(self):
        return self.pacman_reward

    #######################
    # Action Related Functions #
    #######################

    def legalActions(self, legal_actions):
        """
        Turn list of legal actions into corresponding array of actions
        e.g., ['West', 'East'] -> [0. 1. 0. 1.]
        :param legal_actions: list of legal actions as strings obtained from state.getLegalPacmanActions()
        :return: numpy array where '1' values are legal actions and '0' values are illegal actions
        """
        legal_actions_array = np.zeros(len(self.possible_actions), dtype = float)
        for action in legal_actions:
            if action == 'North':
                legal_actions_array[0] = 1
            elif action == 'East':
                legal_actions_array[1] = 1
            elif action == 'South':
                legal_actions_array[2] = 1
            elif action == 'West':
                legal_actions_array[3] = 1
        return legal_actions_array

    def stringToAction(self, string_action):
        """
        Convert string to action
        e.g., 'North' -> Directions.NORTH
        :param string_action: string value of action
        :return: corresponding action which can be further interpreted by the game
        """
        if string_action == 'North':
            return Directions.NORTH
        elif string_action == 'East':
            return Directions.EAST
        elif string_action == 'South':
            return Directions.SOUTH
        elif string_action == 'West':
            return Directions.WEST

    def indexOfAction(self, legal_action):
        """
        Convert string to action index
        e.g., 'North' -> 0
        :param legal_action: string value of action
        :return: corresponding index of action
        """
        if legal_action == 'North':
            return 0
        elif legal_action == 'East':
            return 1
        elif legal_action == 'South':
            return 2
        elif legal_action == 'West':
            return 3

    def actionFromIndex(self, legal_index):
        """
        Convert index to action
        e.g., 0 -> Directions.NORTH
        :param legal_index: index value of action
        :return: corresponding action which can be further interpreted by the game
        """
        if legal_index == 0:
            return Directions.NORTH
        elif legal_index == 1:
            return Directions.EAST
        elif legal_index == 2:
            return Directions.SOUTH
        elif legal_index == 3:
            return Directions.WEST

    #######################
    # Functions for Reinforcement Learning #
    #######################

    def updateQValues(self, prev_state, action, now_state, reward, best_next_action):
        """
        - Update Q-Values table -
        Calculate new value for a state-action pair: (s,a) where a takes pacman from s to s'
        :param prev_state: s - previous state of pacman
        :param action: a - action that takes pacman from previous state to new state
        :param now_state: s' new state of pacman
        :param reward: reward of previous state s
        :param best_next_action: index of maximum legal action value of the (s', a) pair
        """
        updateTarget = reward + self.getGamma() * self.QValues[now_state][best_next_action]
        updateDelta = updateTarget - self.QValues[prev_state][action]
        self.QValues[prev_state][action] += self.getAlpha() * updateDelta

    def getQValue(self, state, action):
        """
        Access value of a given state-action pair
        :param state: s
        :param action: a
        :return: (s,a) value
        """
        return self.QValues[state][action]

    def greedyPick(self, state):
        """
        Given a state, return the index of the largest action value
        :param state: s
        :return: index of maxQValues(s,a)
        """
        best_next_action = np.argmax(self.QValues[state])
        return best_next_action

    def eGreedyProbability(self, no_valid_actions, valid_actions_array, state):
        """
        Based on Q-Table for given state + currently legal actions, calculate probability distribution of actions
        :param no_valid_actions: number legal actions
        :param valid_actions_array: legal actions
        :param state: state
        :return: probability distribution of actions
        """
        # sets initially equal probabilities (p = epsilon/no_valid_actions) for all legal actions from: N,E,S,W
        action_probabilities = valid_actions_array * self.epsilon / no_valid_actions

        print 'HERE: eGreedyProbability function'
        print 'initial legal action probabilities: ', action_probabilities
        print 'corresp. valid actions: ', valid_actions_array
        print 'Qvalues for state: ', self.QValues[state]

        # index of max value in QValues table from 'state' and which is a valid action for next state
        max_index = None
        # max value in QValues table from 'state' and which is a valid action for next state
        max_val = None

        # iterate through actions: index=0,1,2,3
        for index in range(len(valid_actions_array)):
            # if action is legal
            if valid_actions_array[index] == float(1):
                # update max if necessary
                if max_index != None:
                    if self.QValues[state][index] > max_val:
                        max_index = index
                        max_val = self.QValues[state][index]
                # initialize max
                else:
                    max_index = index
                    max_val = self.QValues[state][index]
        print 'max index: ', max_index
        print 'max value: ', max_val

        # attributes a higher probability (p += 1-epsilon) to the legal action with largest value
        action_probabilities[max_index] += (1.0 - self.epsilon)
        return action_probabilities

    #######################
    # getAction method #
    #######################

    def getAction(self, state):
        """
        The main method required by the game. Called every time that
        Pacman is expected to move
        :param state: current game state
        :return: action for pacman to take
        """
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        print ""
        print "1. Legal moves: ", legal
        print "2. Pacman position: ", state.getPacmanPosition()
        print "3. Ghost positions: ", state.getGhostPositions()
        print "4. Food locations: "
        print state.getFood()
        print "5. Score: ", state.getScore()

        # *********************************************************************

        # get legal actions as array of 1s and 0s
        legal_actions_array = self.legalActions(legal)
        print "Legal moves array: ", legal_actions_array

        # get the whole current state of the game (agent states, food, capsules)
        # in a format which can be stored as key of dictionary QValues
        current_state = state.__hash__()
        print "Current state of game key: ", current_state
        # get current reward
        current_reward = state.getScore()
        
        if self.get_State() == None:

            # set state s1
            self.set_State(current_state)

            # look for action maxQ(s1,a) = a1 which is legal (will be 0 first)
            # get probability distribution of actions
            action_probabilities = self.eGreedyProbability(len(legal), legal_actions_array, self.get_State())
            # get next best action index
            best_next_action = np.argmax(action_probabilities)
            # choose next action index
            if self.getEpisodesSoFar() == self.getNumTraining():
                next_action = best_next_action
            else:
                # or (less exploration):
                # next_action = np.random.choice(np.arange(len(action_probabilities)), p=action_probabilities)
                p = random.randrange(0, 1)
                if p > self.epsilon:
                    next_action = indexOfAction(random.choice(legal))  # exploration
                else:
                    next_action = best_next_action  # exploitation

            # set action a1
            self.set_Action(next_action)
            # set initial reward r1
            self.set_Reward(current_reward)

            # take action a1 (or the action chosen from the prob. distrib.)
            return self.actionFromIndex(next_action)

        elif self.get_State() != None:

            # get action maxQ(s_current,a) which is legal
            # get probability distribution of actions
            action_probabilities = self.eGreedyProbability(len(legal), legal_actions_array, current_state)
            print "Resulting ACTION PROBS (%): ", action_probabilities
            # get next best action index
            best_next_action = np.argmax(action_probabilities)
            print "Best next action: ", best_next_action
            # choose next action index
            if self.getEpisodesSoFar() == self.getNumTraining():
                next_action = best_next_action
            else:
                # or (less exploration):
                # next_action = np.random.choice(np.arange(len(action_probabilities)), p=action_probabilities)
                p = random.randrange(0, 1)
                if p > self.epsilon:
                    next_action = indexOfAction(random.choice(legal)) # exploration
                else:
                    next_action = best_next_action  # exploitation

                print "Next action: ", next_action

            # update QValues table
            # Q(s,a) = Q(previous state, action from previous state to current state)
            self.updateQValues(self.get_State(), self.get_Action(), current_state, self.get_Reward(), best_next_action)

            # set state
            self.set_State(current_state)
            # set action
            self.set_Action(next_action)
            # calculate reward signal r for moving from previous state to this current state
            change_in_reward = current_reward - self.get_Reward()
            # set reward
            self.set_Reward(change_in_reward)

            # Now pick what action to take and return it
            pick = self.actionFromIndex(next_action)
            return pick

        # *********************************************************************

    def final(self, state):
        """
        Handle the end of episodes
        This is called by the game after a win or a loss.
        """

        # get the whole current state of the game
        current_state = state.__hash__()
        print "Current state of game key: ", current_state

        # get current reward
        current_reward = state.getScore()
        # calculate reward signal r for moving from previous state to this current state
        change_in_reward = current_reward - self.get_Reward()

        # update Q-Values table
        self.QValues[self.get_State()][self.get_Action()] = change_in_reward

        print "A game just ended!"
        
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)

"""
# Coursework References:
1. 6CCS3ML1 Machine Learning/6CCS3PRE Pattern Recognition: Lecture 9/page 48/Q-learning Pseudo-code
2. GeeksforGeeks. (2019). Q-Learning in Python - GeeksforGeeks. [online] Available at: https://www.geeksforgeeks.org/q-learning-in-python/.

Notes:
- running time (with all print statements included) on my computer: 1min 05sec
"""

