# Berkeley AI Reinforcement Learning Project

- This is an attempt to solve reinforcement project from [Berkeley AI Projects](http://ai.berkeley.edu/reinforcement.html)
- Project is to edit the following files to get GridWorld working:
  - valueIterationAgents.py - A value iteration agent for solving known MDPs.
  - qlearningAgents.py - Q-learning agents for Gridworld, Crawler and Pacman.
  - analysis.py- A file to put your answers to questions given in the project.

- Pseudo code for functions in qlearningAgents.py:
* __init__:
```
function __init__:
  Input: init arguments
  Output: None
  * Initialise QValues table, where each possible (state,action) is initialized to 0
```

* getQValue:
```
function getQValue:
  Input: state, action
  Output: QValue
  * Return QValue from QValues table for specified (state,action)
```

* computeValueFromQValue:
```
function computeValueFromQValue:
  Input: state
  Output: Value
  * Following is formula to calculate Value from QValue:
      Value(state) = max-over-all-legal-actions(QValue(state, action))
  * Fetch legal actions agent can take from current state using
    `getLegalActions` function. This could be going North, South,
    East, West or Stop.
  * If no legal actions, then return 0.0
  * If there are legal actions i.e in terminal state, then get
    max QValue from the list of QValues computed using `getQValue`
    for all legal actions from current state. Return this max value as output
```

* computeActionFromQValues:
```
function computeActionFromQValues:
  Input: state
  Output: Action
  * Fetch legal actions agent can take from current state using
    `getLegalActions` function
  * If there are no legal actions, then return None as best action
  * If there are legal actions i.e. in terminal state, then select
    the action leading to highest QValue from current state as the
    best action and return that as Action output
```

* getAction:
```
function getAction:
  Input: state
  Output: Action
  * Fetch legal actions agent can take from current state using
    `getLegalActions` function
  * If there are no legal actions i.e in terminal state, then
    return None as action
  * If there are legal actions, then use `flipCoin` function to
    get probability distribution over epsilon. Depending on the
    `flipCoin` output, we do the following:
    - If true, Select action randomly from legal action and
      return that as output
    - If false, Select best policy action calculated using
      `getPolicy` function for current state and return that
      as output
```

* update:
```
function update:
  Input: state, action, nextState, reward
  Output: None
  * Fetch legal actions agent can take from next state using
    `getLegalActions` function
  * Fetch existing QValue i.e QValue at time T-1 using `getQValue` function with current state and action as arguments
  * Get list of QValues for nextState and nextAction using `getQValue` function
  * Using max value from list of Qvalues, and using reward, discount and QValue_Tminus1, compute TD (Temporal Difference)
  * Calculate Temporal Difference at time T
  * TD_T = Reward + discounting-factor * max-actions(
  *                                       QValue(nextstate, nextaction)
  *                                      ) - QValue_Tminus1(state, action)
  * Using all the variables we computed, we set QValue for (state, action) in main QValues table using below formula:
  *    QValue(state, action) = QValue_Tminus1(state, action) + learning_rate * TD_T(action, state)
```
