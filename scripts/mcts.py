from __future__ import division

import time
import math
import random
from tqdm import tqdm

def randomPolicy(state):
    reward = 0
    while not state.isTerminal():
        try:
            action = random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        reward += state.getReward()
        state = state.takeAction(action)
    return reward + state.getReward()


class treeNode():
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = state.heuristic()
        self.children = {}

    def __str__(self):
        s=[]
        s.append("totalReward: %s"%(self.totalReward))
        s.append("numVisits: %d"%(self.numVisits))
        s.append("isTerminal: %s"%(self.isTerminal))
        s.append("possibleActions: %s"%(self.children.keys()))
        return "%s: {%s}"%(self.__class__.__name__, ', '.join(s))

class mcts():
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2),
                 rolloutPolicy=randomPolicy):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy
        self.gs = []
        self.offset = 0

    def search(self, initialState, needDetails=False):
        self.root = treeNode(initialState, None)

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in tqdm(range(self.searchLimit)):
                self.executeRound()

        bestChild = self.getBestChild(self.root, 0)
        # print(*[self.root.state.inits[i] for i in range(len(self.root.state.inits))], self.root.isTerminal)
        action=(action for action, node in self.root.children.items() if node is bestChild).__next__()
        childrens = [bestChild]
        actions = [action]
#         print(bestChild)
#         print(children[-1])
        #### semi offline
        while not childrens[-1].state.isTerminal():
            try:
                ch = self.getBestChild(childrens[-1], 0)
                print(ch.numVisits, ch.totalReward)
#                 print("YAY")
#                 print(children[-1].children.items())
#                 print(ch)
                action=(action for action, node in childrens[-1].children.items() if node is ch).__next__()
                childrens.append(ch)
                actions.append(action)
#                 print(actions)
            except:
                break
           #### semi offline
#         bestGrandChild = self.getBestChild(bestChild, 0)
#         print(bestChild, bestGrandChild)
#         action2 = (action for action, node in bestChild.children.items() if node is bestGrandChild).__next__()
        if needDetails:
            return {"action": actions, "expectedReward": bestChild.totalReward / bestChild.numVisits}
        else:
            return actions

    def executeRound(self):
        node = self.selectNode(self.root)
        reward = self.rollout(node.state)
        self.backpropogate(node, reward)

    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
#                 test = self.getBestChild(node, self.explorationConstant)
                node = self.getBestChild(node, self.explorationConstant)
#                 if test is None:
#                     return node
#                 else:
#                     node = test
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        actions = node.state.getPossibleActions()
        for action in actions:
            if action not in node.children:
                newNode = treeNode(node.state.takeAction(action), node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode

        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        mainreward = reward
        while node is not None:
            node.numVisits += 1
            ###TODO hardcoded stuff here
            reward *= 0.95 
            if reward>0:
                reward = max(reward,0.05*mainreward)
            else:
                reward = min(reward,0.05*mainreward)
            node.totalReward += reward #+ node.state.heuristic()
            node = node.parent

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        if len(bestNodes) == 0:
            return None
        return random.choice(bestNodes)
    
    def createGraph(self):
        # self.explorationConstant is to be used
        # maybe reinitialise 'self.g' here, but need to see how to store multiple graphs to plot. 
        # maybe keep multiple self.g's in a list and plot them one by one (and define self.gs = [] in __init__ and create a new self.g in createGraph and append it to self.gs)
        # plot all of them at the end, one by one by maybe accessing the self.g in self.gs and running the plot function on it.
        return
        g = ig.Graph()
        self.gs.append(g)
        vertices = [self.root]
        self.gs[-1].add_vertex()
        vqueue = [self.root]
        edges = []
        print("p",self.root)
        print("p1",self.root.children)
        while vqueue != []:
            currentv = vqueue.pop(0)
            for action, child in currentv.children.items():
                if child not in vertices:
                    self.gs[-1].add_vertex()
                    # add some description for the vertex too
                    vertices.append(child)
                    vqueue.append(child)
                    # print(child)
                if (currentv,child) not in edges:
                    self.gs[-1].add_edge(vertices.index(currentv),vertices.index(child)) #{"label": action} #add weights and all here?
                    # add child.totalReward //// child.numVisits //// node.numVisits //// math.sqrt(2 * math.log(node.numVisits) / child.numVisits) //// child.totalReward / child.numVisits + self.explorationValue * math.sqrt(2 * math.log(node.numVisits) / child.numVisits)
                    edges.append((currentv,child))
    
    def plotGraph(self, index, layouttype = 'rt'):
        return
        if index >= len(self.gs):
            raise ValueError("Index out of range")
        layout = self.gs[index].layout(layouttype, root=[0])
        ig.plot(self.gs[index], layout = layout)