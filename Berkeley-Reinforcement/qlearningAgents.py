# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # Initialise QValues table, where each possible (state,action) is initialized to 0
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # return QValue from QValues table for specified (state,action)
        return self.qValues[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # Following is formula to calculate Value from QValue:
        #     Value(state) = max-over-all-legal-actions(QValue(state, action))

        # * Fetch legal actions agent can take from current state
        #   This could be going North, South, East, West or Stop.
        # * If no legal actions, then return 0.0
        # * If there are legal actions i.e in terminal state, then get
        #   max QValue from the list of QValues computed using `getQValue`
        #   for all legal actions from current state
        legalActions = self.getLegalActions(state)
        if legalActions:
            maxVal = float("-inf")
            for action in legalActions:
                q = self.getQValue(state, action)
                if q >= maxVal:
                    maxVal = q
            return maxVal
        return 0.0

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        # * Fetch legal actions agent can take from current state
        # * If there are no legal actions, then return None as best action
        # * If there are legal actions i.e. in terminal state, then select
        #   the action leading to highest QValue from current state as the
        #   best action
        legalActions = self.getLegalActions(state)
        if legalActions:
            maxVal = float("-inf")
            bestAction = None
            for action in legalActions:
                q = self.getQValue(state, action)
                if q >= maxVal:
                    maxVal = q
                    bestAction = action
            return bestAction
        return None

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # * Fetch legal actions agent can take from current state
        # * If there are no legal actions i.e in terminal state, then
        #   return None as action
        # * If there are legal actions, then use `flipCoin` function to
        #   get probability distribution over epsilon. Depending on the
        #   `flipCoin` output, we do the following:
        #   - If true, Select action randomly from legal action
        #   - If false, Select best policy action calculated using
        #     `getPolicy` function for current state
        if legalActions:
            if util.flipCoin(self.epsilon):
                action = random.choice(legalActions)
            else:
                action = self.getPolicy(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # Fetch legal actions agent can take from next state
        legalActions = self.getLegalActions(nextState)

        # Fetch existing QValue i.e QValue at time Tminus1 using `getQValue` function with current state and action as arguments
        qValue_Tminus1 = self.getQValue(state, action)

        # Get list of QVals for nextState and nextAction using `getQValue` function
        Qvals = []
        if legalActions:
            for nextAction in legalActions:
                Qvals.append(self.getQValue(nextState, nextAction))

        # Using max value from list of Qvals, reward, discount and QValue_Tminus1, compute TD
        # Calculate Temporal Difference at time T
        # TD_T = Reward + discounting-factor * max-actions(
        #                                       QValue(nextstate, nextaction)
        #                                      ) - QValue_Tminus1(state, action)

        if Qvals:
            TD_T = reward + self.discount * max(Qvals) - qValue_Tminus1
        else:
            TD_T = reward - qValue_Tminus1

        # Compute new QValue using the formula:
        #    QValue(state, action) = QValue_Tminus1(state, action) + learning_rate * TD_T(action, state)
        self.qValues[(state, action)] = qValue_Tminus1 + self.alpha * TD_T

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
