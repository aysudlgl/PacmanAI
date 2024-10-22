# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    stack = util.Stack()
    visited = set()

    # Push the start state to the stack: (state, actions, cost)
    start_node = problem.getStartState()
    stack.push((start_node, [], 0))

    while not stack.isEmpty():
        # Pop the current node from the stack
        location, actions, cost = stack.pop()

        # Check if the current state is the goal
        if problem.isGoalState(location):
            return actions  # Return the sequence of actions

        # If the state hasn't been visited, we proceed
        if location not in visited:
            visited.add(location)

            # Expand the state by getting its successors
            for successor, action, stepCost in problem.getSuccessors(location):
                if successor not in visited:
                    # Create a new action sequence by appending the new action
                    new_actions = actions + [action]
                    stack.push((successor, new_actions, cost + stepCost))

    return []


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    visited = set()

    # Push the start state to the stack: (state, actions, cost)
    start_node = problem.getStartState()
    queue.push((start_node, [], 0))

    while not queue.isEmpty():
        # Pop the current node from the stack
        location, actions, cost = queue.pop()

        # Check if the current state is the goal
        if problem.isGoalState(location):
            return actions  # Return the sequence of actions

        # If the state hasn't been visited, we proceed
        if location not in visited:
            visited.add(location)

            # Expand the state by getting its successors
            for successor, action, stepCost in problem.getSuccessors(location):
                if successor not in visited:
                    # Create a new action sequence by appending the new action
                    new_actions = actions + [action]
                    queue.push((successor, new_actions, cost + stepCost))

    return []


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Initialize priority queue and visited dictionary
    priority_queue = util.PriorityQueue()
    visited = {}

    # Create the start node
    start_state = problem.getStartState()
    start_node = {
        "pred": None,          # Predecessor node (previous node)
        "act": None,           # Action that led to this state
        "state": start_state,  # Current state
        "cost": 0,             # Cost to reach this state (g(n))
        "eq": heuristic(start_state, problem)  # Heuristic value (h(n))
    }

    priority_queue.push(start_node, start_node["cost"] + start_node["eq"])

    while not priority_queue.isEmpty():
        # Pop the node with the lowest cost + heuristic (f(n) = g(n) + h(n))
        node = priority_queue.pop()
        location = node["state"]
        cost = node["cost"]
        heuristic_value = node["eq"]


        if location in visited:
            continue

        visited[location] = True

        if problem.isGoalState(location):
            break

        for successor, action, stepCost in problem.getSuccessors(location):
            if successor not in visited:
                # Create a new node for each successor
                new_node = {
                    "pred": node,          # Current node is the predecessor
                    "state": successor,    # Successor state
                    "act": action,         # Action that led to the successor
                    "cost": cost + stepCost,  # Updated cost (g(n))
                    "eq": heuristic(successor, problem)  # Heuristic value (h(n))
                }
                priority_queue.push(new_node, new_node["cost"] + new_node["eq"])

    actions = []
    while node["act"] is not None:
        actions.insert(0, node["act"])
        node = node["pred"]

    return actions


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
