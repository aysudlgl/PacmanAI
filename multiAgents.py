# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isTerminal():
        Returns whether or not the game state is a terminal state
        """

        def minimax(agentIndex, gameState, depth):
            if depth == self.depth or gameState.isTerminal():
                return self.evaluationFunction(gameState)

            # Pacman (Maximizing agent)
            if agentIndex == 0:
                return maxValue(agentIndex, gameState, depth)
            # Ghosts (Minimizing agents)
            else:
                return minValue(agentIndex, gameState, depth)

        def maxValue(agentIndex, gameState, depth):
            value = float('-inf')
            bestAction = None

            for action in gameState.getLegalActions(agentIndex):
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                nextValue, value = minimax(1, successorGameState, depth)

                if nextValue > value:
                    value = nextValue
                    bestAction = action

            return value, bestAction

        def minValue(agentIndex, gameState, depth):
            value = float('inf')
            nextAgent = agentIndex + 1

            if nextAgent >= gameState.getNumAgents():
                nextAgent = 0
                depth += 1

            bestAction = None
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                nextValue, _ = minimax(nextAgent, successor, depth)

                if nextValue < value:
                    value = nextValue
                    bestAction = action

            return value, bestAction

        value, action = minimax(0, gameState, 0)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):
        def alphaBetaPruning(agentIndex, depth, gameState, alpha, beta):
            if depth == self.depth or gameState.isTerminal():
                return self.evaluationFunction(gameState)

            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState, alpha, beta)
            else:
                return minValue(agentIndex, depth, gameState, alpha, beta)

        def maxValue(agentIndex, depth, gameState, alpha, beta):
            value = float('-inf')
            bestAction = None
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                nextValue, value = alphaBetaPruning(1, depth, successor, alpha, beta)
                if nextValue > value:
                    value = nextValue
                    bestAction = action
                if value > beta:
                    return value, bestAction
                alpha = max(alpha, value)
            return value, bestAction

        def minValue(agentIndex, depth, gameState, alpha, beta):
            value = float('inf')
            nextAgent = agentIndex + 1
            if nextAgent >= gameState.getNumAgents():
                nextAgent = 0
                depth += 1
            bestAction = None
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                nextValue, _ = alphaBetaPruning(nextAgent, depth, successor, alpha, beta)
                if nextValue < value:
                    value = nextValue
                    bestAction = action
                if value < alpha:
                    return value, bestAction
                beta = min(beta, value)
            return value, bestAction

        # Pacman is agentIndex 0, start with depth 0, alpha=-inf, beta=+inf
        value, action = alphaBetaPruning(0, 0, gameState, float('-inf'), float('inf'))
        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        def expectimax(agentIndex, depth, gameState):
            if depth == self.depth or gameState.isTerminal():
                return self.evaluationFunction(gameState)

            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState)
            else:
                return expValue(agentIndex, depth, gameState)

        def maxValue(agentIndex, depth, gameState):
            value = float('-inf')
            bestAction = None
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                nextValue, value = expectimax(1, depth, successor)
                if nextValue > value:
                    value = nextValue
                    bestAction = action
            return value, bestAction

        def expValue(agentIndex, depth, gameState):
            value = 0
            nextAgent = agentIndex + 1
            if nextAgent >= gameState.getNumAgents():
                nextAgent = 0
                depth += 1
            actions = gameState.getLegalActions(agentIndex)
            probability = 1.0 / len(actions)
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                nextValue, _ = expectimax(nextAgent, depth, successor)
                value += probability * nextValue
            return value, None

        value, action = expectimax(0, 0, gameState)
        return action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    """
     An evaluation function that uses Manhattan distance to compute:
     - Distance to the nearest food
     - Distance to ghosts
     - Bonus for eating food and staying safe from ghosts
     """
    # Get the Pacman position and food grid
    pacmanPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghost.scaredTimer for ghost in ghostStates]


    score = currentGameState.getScore()

    # Here we calculate the Manhattan distance to the closest food
    foodDistances = [manhattanDistance(pacmanPos, food) for food in foodGrid.asList()]
    if foodDistances:
        score += 1.0 / min(foodDistances)

    #here we calculate distances to ghosts and avoid them if they're not scared
    for i, ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        distToGhost = manhattanDistance(pacmanPos, ghostPos)

        if scaredTimes[i] > 0:  # Ghost is scared, encourage eating it
            score += 200 / distToGhost  # The closer the better for scared ghosts
        else:  # Ghost is not scared, avoid it
            if distToGhost > 0:
                score -= 10 / distToGhost  # Penalize being close to non-scared ghosts

    # bonus for being in a state with fewer food pellets left
    score -= 3 * len(foodGrid.asList())  # Encourage eating all food

    return score


# Abbreviation
better = betterEvaluationFunction
