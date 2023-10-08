
from mcts import mcts
from copy import deepcopy
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import random
import time
import joblib
import numpy as np
random.seed(0)

disableTQDM = False
if disableTQDM:
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

explicit = False
sleeptimer = 0.1

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 32, dropout = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
rules = torch.load('Rule-Learning/Rule-Learning/learnt/1Object/4rules_4_2.pth')
pmodel = joblib.load('models/pmodelv2.joblib') #or pmodelv1.joblib, idk what is better (v1 prunes too much but still reaches goal / v2 prunes less and takes longer to run)


class Stepper():
    def __init__(self, inits, goals, rules, pmodel, obstacles = [], explicit = False, sleeptimer = 0.1):
        self.inits = inits
        self.goals = goals
        self.rules = rules
        self.pmodel = pmodel
        self.obstacles = obstacles
        self.explicit = explicit
        self.sleeptimer = sleeptimer
        
        self.coll = False
        self.prohibitedActions = []
    
    def heuristic(self, num_iter = 10, num_action = 8):
        farness = 0
        for i in range(len(self.inits)):
            farness += abs(self.inits[i][0]-self.goals[i][0])+abs(self.inits[i][1]-self.goals[i][1])
        farness *= -1
        return 1*num_iter/num_action*(farness)
        return 1*num_iter/num_action*(self.partialGoalReached()[1])
        
        
    def getPossibleActions(self):
        actions= []
        for j in range(len(self.inits)):
            actions.extend([(i, j) for i in range(len(rules))])
        return actions
    
    def takeAction(self, action):
        new = deepcopy(self)
        new.prohibitedActions = []
        new.coll = False
        ruleIndex, obj = action
        rule = self.rules[ruleIndex]
        
        OHEaction = [0]*len(self.rules)
        OHEaction[ruleIndex] = 1
        

        
        output = rule(torch.tensor([*self.inits[obj]]*2))
        new.inits[obj] = (output[0].item(), output[1].item())
        
        if self.explicit:
            return new

        if action in self.prohibitedActions:
            new.coll = True
            return new
        
        answer = False
        for i in range(1, len(self.inits)):
            answer = bool(self.pmodel.predict([[*self.inits[obj],*self.inits[(obj+i)%len(self.inits)],*OHEaction]])[0])
            if answer == True:
                break
        for obs in self.obstacles:
            if answer == True:
                break
            answer = (bool(self.pmodel.predict([[*self.inits[obj],*obs,*OHEaction]])[0]))
        
        if answer == True:
            new.coll = True
            self.prohibitedActions.append(action)
            return new
        else:
            new.coll = False
            return new
        
    def isTerminal(self):
        if self.outOfBounds():
#             print("OUTOFBOUNDS")
            return True
        if self.collision():
            #print("COLL")
            return True
        if self.goalReached():
#             print("GOAL")
            return True
        return False
    
    def getReward(self):
        if self.outOfBounds():
            return -1
        if self.collision():
            return -1
        if self.goalReached():
            return 10
        if self.partialGoalReached()[0]:
            return 2*self.partialGoalReached()[1]
        return 0
    
    def collision(self):
        if self.explicit:
            return self.explicitCollision()
        return self.coll
#         time.sleep(0.1)
#         return abs(self.x1-self.x2)+abs(self.y1-self.y2) < 0.05
    def explicitCollision(self):
#         time.sleep(0.1)
#         return abs(self.x1-self.x2)+abs(self.y1-self.y2) < 0.05
        time.sleep(self.sleeptimer)
        
        objEdges = [(self.inits[i][0]+(0.1)/2, self.inits[i][0]-(0.1)/2, self.inits[i][1]+(0.1)/2, self.inits[i][1]-(0.1)/2) for i in range(len(self.inits))]
        for i in range(len(self.inits)):
            for j in range(i+1, len(self.inits)):
                if not (objEdges[i][0]<objEdges[j][1] or objEdges[i][1]>objEdges[j][0] or objEdges[i][2]<objEdges[j][3] or objEdges[i][3]>objEdges[j][2]):
                    return True
        
        for obs in self.obstacles:
            currEdgesv3 = (obs[0]+(0.1)/2, obs[0]-(0.1)/2, obs[1]+(0.1)/2, obs[1]-(0.1)/2)
            for i in range(len(self.inits)):
                if not (objEdges[i][0]<currEdgesv3[1] or objEdges[i][1]>currEdgesv3[0] or objEdges[i][2]<currEdgesv3[3] or objEdges[i][3]>currEdgesv3[2]):
                    return True
#         if abs(self.x1)>0.55 or abs(self.y1)>0.55 or abs(self.x2)>0.55 or abs(self.y2)>0.55:
#             return True
        return False
    
    def outOfBounds(self):
        for i in range(len(self.inits)):
            if abs(self.inits[i][0])>0.55 or abs(self.inits[i][1])>0.55:
                return True
        return False

    def goalReached(self):
        if self.partialGoalReached()[1] == len(self.inits):
            return True
        return False
        for goal in self.goals:
            flag = False
            for i in range(len(self.inits)):
                if self.closeCheck(self.inits[i][0], self.inits[i][1], goal[0], goal[1]):
                    flag = True
                    break
            if not flag:
                return False
        return True
    
    def partialGoalReached(self):
        cnt = 0
        for i in range(len(self.inits)):
            if self.closeCheck(self.inits[i][0], self.inits[i][1], self.goals[i][0], self.goals[i][1]):
                cnt += 1
        return cnt>0, cnt
        cnt = len(self.goals)
        for goal in self.goals:
            flag = False
            for i in range(len(self.inits)):
                if self.closeCheck(self.inits[i][0], self.inits[i][1], goal[0], goal[1]):
                    flag = True
                    break
            if not flag:
                cnt-=1
        if cnt>0:
            return True, cnt
        return False, cnt
#         return abs(self.x1-self.goalx)+abs(self.y1-self.goaly) < 0.05
#         return (abs(self.x1-self.goalx)**2)+(abs(self.y1-self.goaly)**2) < (0.05)**2
    
    def closeCheck(self, x1, y1, x2, y2):
        return abs(x1-x2)+abs(y1-y2) < 0.05

# obstacles = [(0.0,0.05*i) for i in range(-10,11)]
# obstacles.extend([(0.05*i, 0.07) for i in range(1,11)])
# obstacles.extend([(0.05*i,0.37) for i in range(1,7)])
# obstacles.extend([(0.5,0.2),(0.45,0.2)])

# 1st domain (v0)
# obstacles = [(0.05*i,-0.15) for i in range(-10,-4)]
# obstacles.extend([(0.05*i,-0.15) for i in range(5,11)])
# obstacles.extend([(0.05*i,0.15) for i in range(-3,4)])
# obstacles.extend([(0.15,0.05*i) for i in range(3,11)])
# obstacles.extend([(-0.15,0.05*i) for i in range(3,11)])
# obstacles.extend([(0.25,0.05*i) for i in range(-10,-2)])
# obstacles.extend([(-0.25,0.05*i) for i in range(-10,-2)])
# initstep = [(-0.3, 0.3), (0.0, -0.3), (0.3, 0.3)]
# goalstep = [(-0.3,0.0), (0.0,0.0), (0.3,0.0)]


# 1st domain (v1) {its working}
# obj_size = 0.1
# obj_divide = 2
# obstacles = []
# obstacles.extend([(-0.25, x) for x in np.arange(-0.5, -0.15 + obj_size/obj_divide, obj_size/obj_divide)])
# obstacles.extend([(0.25, x) for x in np.arange(-0.5, -0.15 + obj_size/obj_divide, obj_size/obj_divide)])
# obstacles.extend([(x, -0.15) for x in np.arange(-0.25, 0.25 + obj_size/obj_divide, obj_size/obj_divide)])
# obstacles.extend([(-0.15, x) for x in np.arange(0.15, 0.5 + obj_size/obj_divide, obj_size/obj_divide)])
# obstacles.extend([(0.15, x) for x in np.arange(0.15, 0.5 + obj_size/obj_divide, obj_size/obj_divide)])
# obstacles.extend([(x, 0.15) for x in np.arange(-0.5, -0.15 + obj_size/obj_divide, obj_size/obj_divide)])
# obstacles.extend([(x, 0.15) for x in np.arange(0.15, 0.5 + obj_size/obj_divide, obj_size/obj_divide)])
# initstep = [(-0.4, 0.4), (0.0, -0.4), (0.4, 0.4)]
# goalstep = [(-0.4,0.0), (0.0,0.0), (0.4,0.0)]


# 2nd domain
# obj_size = 0.1
# obj_divide = 2
# obstacles = []
# obstacles.extend([(x, 0.15) for x in np.arange(-0.15, 0.15 + obj_size/obj_divide, obj_size/obj_divide)])
# obstacles.extend([(x, -0.15) for x in np.arange(-0.15, 0.15 + obj_size/obj_divide, obj_size/obj_divide)])
# obstacles.extend([(0.15, x) for x in np.arange(-0.1, 0.1 + obj_size/obj_divide, obj_size/obj_divide)])
# initstep = [(-0.3,0.0)]
# goalstep = [(0.3,0.0)]

# 3rd domain
# obj_size = 0.1
# obj_divide = 2
# obstacles = []
# obstacles.extend([(-0.1, x) for x in np.arange(-0.5, -0.15 + obj_size/obj_divide, obj_size/obj_divide)])
# obstacles.extend([(0.1, x) for x in np.arange(-0.5, -0.15 + obj_size/obj_divide, obj_size/obj_divide)])
# obstacles.extend([(x, -0.15) for x in np.arange(-0.1, 0.1 + obj_size/obj_divide, obj_size/obj_divide)])
# obstacles.extend([(-0.1, x) for x in np.arange(0.15, 0.5 + obj_size/obj_divide, obj_size/obj_divide)])
# obstacles.extend([(0.1, x) for x in np.arange(0.15, 0.5 + obj_size/obj_divide, obj_size/obj_divide)])
# obstacles.extend([(x, 0.15) for x in np.arange(-0.1, 0.1 + obj_size/obj_divide, obj_size/obj_divide)])
# initstep = [(-0.25, -0.2), (-0.25, 0.2)]
# goalstep = [(0.25, -0.2), (0.25, 0.2)]


# 4th domain
obj_size = 0.1
obj_divide = 2
obstacles = []
obstacles.extend([(x, -0.15) for x in np.arange(-0.5, 0.15 + obj_size/obj_divide, obj_size/obj_divide)])
obstacles.extend([(x, 0.15) for x in np.arange(-0.15, 0.5 + obj_size/obj_divide, obj_size/obj_divide)])
obstacles.extend([(-0.15, x) for x in np.arange(-0.15, 0.5 + obj_size/obj_divide, obj_size/obj_divide)])
obstacles.extend([(0.15, x) for x in np.arange(-0.5, 0.15 + obj_size/obj_divide, obj_size/obj_divide)])
initstep = [(-0.45, -0.45), (-0.45, 0.45), (0.45, -0.45), (0.45, 0.45)]
goalstep = [(-0.15, -0.45), (-0.45, 0.15), (0.45, -0.15), (0.15, 0.45)]


print(obstacles)

starttime = time.perf_counter()
random.seed(0)
tree = mcts(iterationLimit=100000)
initialState = Stepper(initstep, goalstep, rules, pmodel, obstacles = obstacles, explicit=explicit, sleeptimer=sleeptimer)
states = [initialState]
chosenActions = [(-1,-1)]
actualStates = []
i = 0
parentState = initialState
while True:
    try:
        actions = tree.search(initialState=initialState)
    except:
        initialState.prohibitedActions = []
        initialState.coll = False
        actions = tree.search(initialState=initialState)
    for bestAction in actions:
        initialState = initialState.takeAction(bestAction)
        states.append(initialState)
        chosenActions.append(bestAction)
        print(bestAction, *[initialState.inits[i] for i in range(len(initialState.inits))])    
    i+=len(actions)
    print(i)
    goodPlan = True
    cnt = -1
    if not explicit:
        for state in tqdm(states):
            cnt+=1
            if state.explicitCollision():
                print("BAD")
                state.coll = True
                goodPlan = False
                initialState = parentState
                initialState.prohibitedActions.append(chosenActions[cnt])
                print(chosenActions[cnt], *[initialState.inits[i] for i in range(len(initialState.inits))])
                states = []
                chosenActions = []
                break
            else:
                print("GOOD")
                actualStates.append(state)
                parentState = state
    else:
        actualStates.extend(states)
    if goodPlan and initialState.goalReached():
        break
    if i>200:
            print("DID NOT REACH!")
            break
    states = []
    chosenActions = []
print("ACTUAL STATES")
for aS in actualStates:
    print(*[aS.inits[i] for i in range(len(aS.inits))])
print("OVER STATES")
print(i)        
endtime = time.perf_counter()
print(endtime - starttime)
# 0 -> +1,0
# 1 -> -1,0
# 2 -> 0,+1
# 3 -> 0, -1
# time is 14minutes



