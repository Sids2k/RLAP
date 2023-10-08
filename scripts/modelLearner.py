
import math
epoch_num = 3
batch_size = 1 # for scheduler, ignore.
batches = 32 #16 or 1
SGD = True #when true, ignores batches
slot_dim = 2 + 2 + 4 #TODO automate this # init + final + ruleOHE
slot_len = 16
slot_hidden = 16
rule_len = 16
slot_num = 2
rule_num = 4 # 5 for 'no move' also
mlp_input = 4
mlp_hidden = 32
mlp_output = 2
dropout = 0.0 #lower is giving much better results, 0.0 is best i think
tau = 0.8
lr = (5e-4)/2 #5e-4
max_lr = (5e-3)/2 #5e-3
maxMSE = math.sqrt(2)
KLtype = 'output' #rulevec or output or slotvec or model
betaKL = -1e-1 #make this negative only
betaKLMulti = 1
maxBetaKL = 100
minBetaKL = 0.02
betaRuleCE = 0 #1e-1
epsKL = 1e-7
betaCE = 1 #1e-1
betaMSE = 5 #1
endstate = (100.0,100.0)
seed = 0
test_train_split = 0.8
version = 3

import gc
gc.set_threshold(0)



globalBetaCE = betaCE
globalBetaMSE = betaMSE
globalBetaKL = betaKL



import numpy as np
import matplotlib.pyplot as plt
data = np.load(f'demo_data/datasetv{version}.npy', allow_pickle = True)
import torch.nn as nn
import torch
import torch.nn.functional as F
import sklearn.ensemble as se
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import pickle
import random
os.environ['PYTHONHASHSEED'] = str(seed)
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)

data, datatest = train_test_split(data, test_size = 1-test_train_split, random_state = seed)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 32, dropout = 0.0):
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
oldrules = torch.load('Rule-Learning/Rule-Learning/learnt/1Object/4rules_4_2.pth')



class entity():
    def __init__(self, data):
        super().__init__()
        self.init = data[0]
        self.fin = data[1] 
        self.size = data[2]
        self.action = data[3]
        ruleOHE = torch.zeros(rule_num)
        ruleOHE[int(self.action)] = 1.0
        self.data = (*self.init, *self.fin, *ruleOHE)





class slot(nn.Module):
    def __init__(self, entity, slot_len = slot_len):
        super().__init__()
        self.data = torch.tensor(entity.data)
    def query_gen(self, W):
        if type(W) == torch.nn.Parameter:
            return torch.matmul(self.data, W)
        else:
            return W(self.data)
    def key_gen(self, W):
        if type(W) == torch.nn.Parameter:
            return torch.matmul(self.data, W)
        else:
            return W(self.data)



class rule(nn.Module):
    def __init__(self, rule_len, rule_mlp):
        super().__init__()
        self.rule_vec = nn.Parameter(torch.randn(rule_len))
        self.rule_mlp = rule_mlp
    def key_gen(self, W):
        if type(W) == torch.nn.Parameter:
            return torch.matmul(self.rule_vec, W)
        else:
            return W(self.rule_vec)
    def rule_output(self, x):
        return self.rule_mlp(x)




class model(nn.Module):
    def __init__(self):
        super().__init__()
        if slot_hidden == -1:
            self.Wq = nn.Parameter(torch.randn(slot_dim, slot_len))
            self.Wk = nn.Parameter(torch.randn(rule_len, slot_len))
            self.Wqn = nn.Parameter(torch.randn(slot_dim, slot_len))
            self.Wkn = nn.Parameter(torch.randn(slot_dim, slot_len))
        else:
            self.MLPq = MLP(slot_dim, slot_len, slot_hidden)
            self.MLPk = MLP(rule_len, slot_len, slot_hidden)
            self.MLPqn = MLP(slot_dim, slot_len, slot_hidden)
            self.MLPkn = MLP(slot_dim, slot_len, slot_hidden)
        self.rule_pred = MLP(mlp_input+rule_num, 1, mlp_hidden)
        

    def forward(self, primarySlots, secondarySlots, rules):
        '''
        Part 1: Primary slot and Rule selection
        Part 1a: Key and Query generation
        '''
        if slot_hidden == -1:
            self.slot_query = torch.stack([slot.query_gen(self.Wq) for slot in primarySlots])
            self.rule_key = torch.stack([rule.key_gen(self.Wk) for rule in rules])
        else:
            self.slot_query = torch.stack([slot.query_gen(self.MLPq) for slot in primarySlots])
            self.rule_key = torch.stack([rule.key_gen(self.MLPk) for rule in rules])
        self.rule_mlp = [rule.rule_mlp for rule in rules]
        '''
        Part 1b: Choosing Primary slot and Rule
        '''
        self.slot_rule_matrix = torch.stack([torch.stack([torch.dot(self.slot_query[i], self.rule_key[j]) for j in range(len(self.rule_key))]) for i in range(len(self.slot_query))])
        self.best_slot_rule_mask = nn.functional.gumbel_softmax(self.slot_rule_matrix.view(-1), tau = tau, hard = True).view(len(self.slot_query), len(self.rule_key))
        self.chosen_primary_slot = torch.mm(self.slot_query.T,self.best_slot_rule_mask).sum(dim=1)
        self.primary_slot_mask = self.best_slot_rule_mask.sum(dim=1)
        self.chosen_rule = torch.mm(self.rule_key.T,self.best_slot_rule_mask.T).sum(dim=1)
        self.rule_mask = self.best_slot_rule_mask.sum(dim=0)
        '''
        Part 2: Contextual slot selection
        Part 2a: Key and Query generation
        '''
        if slot_hidden == -1:
            self.slot_query_2 = torch.stack([slot.query_gen(self.Wqn) for slot in primarySlots])
            self.slot_key = torch.stack([slot.key_gen(self.Wkn) for slot in secondarySlots])
        else:
            self.slot_query_2 = torch.stack([slot.query_gen(self.MLPqn) for slot in primarySlots])
            self.slot_key = torch.stack([slot.key_gen(self.MLPkn) for slot in secondarySlots])
        '''
        Part 2b: Choosing Contextual slot
        '''
        self.primary_slot_query = torch.mm(self.slot_query_2.T, self.primary_slot_mask.view(-1,1)).sum(dim=1)
        self.slot_slot_matrix = torch.stack([torch.dot(self.primary_slot_query, self.slot_key[j]) for j in range(len(self.slot_key))])
        self.best_slot_mask = nn.functional.gumbel_softmax(self.slot_slot_matrix.view(-1), tau = tau, hard = True).view(-1,1)
        self.chosen_secondary_slot = torch.mm(self.slot_key.T,self.best_slot_mask).sum(dim=1)
        self.secondary_slot_mask = self.best_slot_mask.sum(dim=1)
        '''
        Part 3: Applying MLP
        '''
        self.primary_slot = torch.matmul(torch.stack([slot.data for slot in primarySlots]).T, self.primary_slot_mask).T
#         print(primary_slot[:2])
        self.secondary_slot = torch.matmul(torch.stack([slot.data for slot in secondarySlots]).T, self.secondary_slot_mask).T
#         print(secondary_slot[:2])
        self.all_predicted_outputs = torch.stack([rule_mlp(torch.cat((self.primary_slot[:2], self.primary_slot[:2]))) for rule_mlp in self.rule_mlp]) # check if the :2 works or not
        self.predicted_output = torch.matmul(self.all_predicted_outputs.T, self.rule_mask)
#         print(self.rule_mask)
        #TODO !!!! maybe add 'with torch.no_grad:' here
#         print(torch.cat((self.primary_slot[:2], self.secondary_slot[:2], self.rule_mask)))
        self.no_grad_rule_mask = self.rule_mask.detach().clone()
#         print(self.no_grad_rule_mask.requires_grad)
#         print('----')
#         print(self.rule_mask.requires_grad)
        self.predicted_collision = self.rule_pred(torch.cat((self.primary_slot[:2], self.secondary_slot[:2], self.no_grad_rule_mask))) # check if the :2 works or not
#         print(self.predicted_collision)
        return self.primary_slot, self.secondary_slot, self.rule_mask, self.predicted_output, self.all_predicted_outputs, self.predicted_collision[0]
    




class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))




rules = [rule(rule_len, oldrules[_]) for _ in range(rule_num)]
m = model()
mse = nn.MSELoss()
cross = nn.CrossEntropyLoss()
jsdloss = JSD()
params = list(m.parameters())
for i in range(len(rules)):
    params += list(rules[i].parameters()) 
optimiser = torch.optim.Adam(params, lr = lr) 
######## #TODO we have a lot of 'constants' here, make them options at the beginning of the notebook
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr = max_lr, total_steps = (2*epoch_num+10)*batch_size)
mselosses = []
belosses = []
kllosses = []
rcelosses = []
XColl = []
yColl = []
XColltest = []
yColltest = []

for epoch in tqdm(range(epoch_num)):
    m.train() 
    operation_to_rule = {k:{j:0 for j in range(rule_num)} for k in range(rule_num)}
    pAccuracy = 0
    pTotal = 0
    pFalsePos = 0
    for k in range(rule_num):
        rules[k].train()
        for params in rules[k].rule_mlp.parameters():
            params.requires_grad = False
        rules[k].rule_vec.requires_grad = True

    betaCE = globalBetaCE/1e5
    betaMSE = globalBetaMSE
    betaKL = globalBetaKL
    for params in m.rule_pred.parameters():
        params.requires_grad = False
    if slot_hidden == -1:
        m.Wq.requires_grad = True
        m.Wk.requires_grad = True
        m.Wqn.requires_grad = True
        m.Wkn.requires_grad = True
    else:
        for mlpmodel in [m.MLPq, m.MLPk, m.MLPqn, m.MLPkn]:
            for params in mlpmodel.parameters():
                params.requires_grad = True
            
    summselosses = 0.0
    sumbelosses = 0.0
    sumkllosses = 0.0
    sumrcelosses = 0.0
    cnt = 0
    shuffledData = list(range(len(data)))
    random.shuffle(shuffledData)
    for i in tqdm(shuffledData):
        cnt+=1
        inits, finals, object_size, initsobs, finalsobs, obstacle_size, obj, action1 = data[i]
#         print(inits, finals, object_size, initsobs, finalsobs, obstacle_size, obj, action1)
#         print("wowow")
        primarySlots = [slot(entity([inits[j],finals[j],object_size[j], action1]), slot_len = slot_len) for j in range(len(object_size))]
        secondarySlots = [slot(entity([initsobs[j],finalsobs[j],obstacle_size[j], action1]), slot_len = slot_len) for j in range(len(obstacle_size))]
        primary_slot, secondary_slot, rule_mask, predicted_output, all_predicted_outputs, predicted_collision = m(primarySlots, secondarySlots, rules)
#         print("start")
#         print(predicted_output)
#         print("========================\n",all_predicted_outputs)
#         print("--------")

#         print(primary_slot, secondary_slot, rule_mask, predicted_output, all_predicted_outputs )
#         print("--------")
#         print(predicted_output, predicted_collision[0], all_predicted_collision,torch.tensor(float(int(p))))
        KLdivs = torch.tensor(0.0)
        _n = 0
        if KLtype == 'output':
            for i in range(len(all_predicted_outputs)):
                for j in range(i):
                    _n+=1
                    KLdivs += (jsdloss(F.softmax(all_predicted_outputs[i]),F.softmax(all_predicted_outputs[j])))
        if KLtype == 'rulevec':
            for i in range(len(rules)):
                for j in range(i):
                    _n+=1
                    KLdivs += (jsdloss(F.softmax(rules[j].rule_vec),F.softmax(rules[i].rule_vec)))
        if KLtype == 'slotvec':
            for i in range(len(primarySlots)):
                for j in range(i):
                    _n+=1
                    KLdivs += (jsdloss(F.softmax(primarySlots[j].data),F.softmax(primarySlots[i].data)))
            for i in range(len(secondarySlots)):
                for j in range(i):
                    _n+=1
                    KLdivs += (jsdloss(F.softmax(secondarySlots[j].data),F.softmax(secondarySlots[i].data)))
        if KLtype == 'model':
            for i in range(len(rules)):
                for j in range(i):
                    _n+=1
                    #TODO made into torch.cat instead of torch.concat due to old torch version?
                    KLdivs += (jsdloss(F.softmax(torch.cat((rules[j].rule_mlp.linear.weight.flatten(),rules[j].rule_mlp.linear2.weight.flatten(),rules[j].rule_mlp.linear.bias.flatten(),rules[j].rule_mlp.linear2.bias.flatten()))),F.softmax(torch.cat((rules[i].rule_mlp.linear.weight.flatten(),rules[i].rule_mlp.linear2.weight.flatten(),rules[i].rule_mlp.linear.bias.flatten(),rules[i].rule_mlp.linear2.bias.flatten())))))
        try:
            KLdivs /= (_n*(_n-1)/2)
        except:
            pass
        p = (finals[obj] == endstate) #TODO currently this is just for 1 datapoint at a time.
        if (epoch == epoch_num - 1) and (len(XColl) < len(data)):
            XColl.append([*(primary_slot.detach().clone().numpy()[:2]), *(secondary_slot.detach().clone().numpy()[:2]), *(rule_mask.detach().clone().numpy())])
            yColl.append(int(p))
        loss = betaMSE*(1-int(p))*mse(predicted_output, torch.tensor(finals[obj]))/(maxMSE)
#         print("-----")
#         print(finals[obj])
        loss1 = betaCE*F.binary_cross_entropy(F.sigmoid(predicted_collision),torch.tensor(float(int(p)))) 
#         print(F.sigmoid(predicted_collision))
        loss2 = betaKL*KLdivs 
        #TODO added a new loss function here
        # print(rule_mask)
        # print(torch.tensor(action1))
        #TODO loss3 = betaRuleCE*F.cross_entropy(rule_mask, torch.tensor(action1))
        loss3 = (1-int(p))*betaRuleCE*nn.CrossEntropyLoss()(rule_mask.view(1,-1), torch.tensor([action1]))
#         print(F.cross_entropy(rule_mask, torch.tensor(action1)), torch.argmax(rule_mask.view(1,-1)), action1)
        loss += loss3 + loss1
        summselosses += (1-int(p))*mse(predicted_output, torch.tensor(finals[obj]))
        sumbelosses += F.binary_cross_entropy(F.sigmoid(predicted_collision),torch.tensor(float(int(p))))
        sumkllosses += KLdivs
        sumrcelosses += (1-int(p))*F.cross_entropy(rule_mask.view(1,-1), torch.tensor([action1]))
        loss.backward()
#         print(m.rule_pred.linear.bias.grad)
        if (SGD is True) or (cnt % (len(data)//batches) == 0) or (cnt == len(data) - 1):
            optimiser.step()
            optimiser.zero_grad()
            # gc.collect()
            # print("yay")
    betaKL *= betaKLMulti
    if -1*betaKL > maxBetaKL:
        betaKL = -1*maxBetaKL
    if -1*betaKL < minBetaKL:
        betaKL = -1*minBetaKL
    scheduler.step()
    mselosses.append(summselosses)
    belosses.append(sumbelosses)
    kllosses.append(sumkllosses)
    rcelosses.append(sumrcelosses)
    print(f"MSE Loss = {loss - loss1 - loss2 - loss3}, BCE Loss = {loss1}, KL Loss = {loss2}, RuleCE Loss = {loss3}")
    m.eval() 
    for k in range(rule_num):
        rules[k].eval()
#         print(k,rules[k].rule_vec)
    totalmae = 0
    for i in tqdm(range(len(datatest))):
        inits, finals, object_size, initsobs, finalsobs, obstacle_size, obj, action1 = datatest[i]
        primarySlots = [slot(entity([inits[j],finals[j],object_size[j],action1]), slot_len = slot_len) for j in range(len(object_size))]
        secondarySlots = [slot(entity([initsobs[j],finalsobs[j],obstacle_size[j],action1]), slot_len = slot_len) for j in range(len(obstacle_size))]
#         slots = [slot(entity(datatest[i]), slot_len = slot_len) for _ in range(slot_num)]
        primary_slot, secondary_slot, rule_mask, predicted_output, all_predicted_outputs, predicted_collision = m(primarySlots, secondarySlots, rules)
        rule_index = torch.argmax(rule_mask)
        p = (finals[obj] == endstate)
        if (epoch == epoch_num - 1) and (len(XColltest) < len(datatest)):
            XColltest.append([*(primary_slot.detach().clone().numpy()[:2]), *(secondary_slot.detach().clone().numpy()[:2]), *(rule_mask.detach().clone().numpy())])
            yColltest.append(int(p))
            if int(p) == 1:
                # print(int(rule_index), action1)
                pass
        if finals[obj] == endstate:
            if predicted_collision > 0:
                pAccuracy += 1
            pTotal += 1
        else:
            if predicted_collision > 0:
                pFalsePos += 1
            operation_to_rule[action1][int(rule_index)] += 1
        if predicted_collision > 0:
            totalmae += 100*(1-int((finals[obj] == endstate)))
        else:
            totalmae += np.sum(np.abs(predicted_output.detach().numpy()-np.array(finals[obj]))) #TODO made this :-1, shouldn't hard code it like this
    totalmae /= len(datatest)
    print(f"Total MAE = {totalmae}")
    # print('------------------')
    # print(rules[0].rule_mlp)
    # print('------------------')
    for k in operation_to_rule:
        print(k, end = ' ')
        print(operation_to_rule[k])
    print('Accuracy: {}/{} = {}'.format(np.sum([operation_to_rule[k][k] for k in range(len(operation_to_rule))]),np.sum(np.sum([operation_to_rule[k][j] for j in range(len(operation_to_rule)) for k in range(len(operation_to_rule))])),np.sum([operation_to_rule[k][k] for k in range(len(operation_to_rule))])/np.sum(np.sum([operation_to_rule[k][j] for j in range(len(operation_to_rule)) for k in range(len(operation_to_rule))]))))
    print('Recall of unreachable: {}/{} = {} with {} false positives'.format(pAccuracy, pTotal, pAccuracy/pTotal, pFalsePos))
    print('========================================================')
    
# 0 {0: 0, 1: 0, 2: 264, 3: 0}
# 1 {0: 0, 1: 0, 2: 0, 3: 261}
# 2 {0: 0, 1: 228, 2: 0, 3: 0}
# 3 {0: 247, 1: 0, 2: 0, 3: 0}
# 97.5 accuracy

# exit the python script


pickle.dump((XColl,yColl,XColltest,yColltest),open(f"demo_data/XYXtYtCollv{version}.pkl","wb"))

allrules = nn.ModuleList()
for k in range(rule_num):
    allrules.append(rules[k].rule_mlp)
torch.save(allrules, f'models/RulesMLPv{version}.pth')
allrules = []
for k in range(rule_num):
    allrules.append(rules[k].rule_vec)
f = open(f'models/RulesVecv{version}.pkl', 'wb')
pickle.dump(allrules, f)
torch.save(m, f'models/Modelv{version}.pth')

exit()


pmodel = se.RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
pmodel.fit(XColl,yColl)




print("pmodel")
print(pmodel.score(XColl,yColl),pmodel.score(XColltest,yColltest))
# find f1 score as pmodel on xcoll and xcolltest
from sklearn.metrics import classification_report
target_names = ['feasible', 'not feasible']
print(classification_report(yColl, pmodel.predict(XColl),target_names = target_names))
print(classification_report(yColltest, pmodel.predict(XColltest),target_names = target_names))
exit()
# # import joblib
# # joblib.dump(pmodel, "../learnt/pmodelv15.2.joblib")
# import pickle
# import joblib


# In[77]:


# tmselosses = [a.item() for a in mselosses]
# tbelosses = [a.item() for a in belosses]
# tkllosses = [a.item() for a in kllosses]
# # trcelosses = [a.item() for a in rcelosses]
# plt.plot(range(len(tmselosses)+1)[1:],[20*a for a in tmselosses],label="mean squared error loss")
# plt.plot(range(len(tbelosses)+1)[1:], tbelosses, label = "binary cross entropy loss")
# plt.plot(range(len(tkllosses)+1)[1:], tkllosses, label = "KL divergence loss")
# # plt.plot(range(len(trcelosses)+1)[1:], trcelosses, label = "rule cross entropy loss")
# plt.title("Losses vs Epoch")
# plt.legend()
# pass


# In[78]:


# datacheck = datatest[:20]
# # rules = [rule(rule_len, mlp_input, mlp_output, mlp_hidden, dropout) for _ in range(rule_num)]
# m.eval()
# for k in range(rule_num):
#     rules[k].eval()
# for i in tqdm(range(len(datacheck))):
# #     slots = [slot(entity(datacheck[i]), slot_len = slot_len) for _ in range(slot_num)]
#     inits, finals, object_size, initsobs, finalsobs, obstacle_size, obj, action1 = datacheck[i]
#     primarySlots = [slot(entity([inits[j],finals[j],object_size[j],action1]), slot_len = slot_len) for j in range(len(object_size))]
#     secondarySlots = [slot(entity([initsobs[j],finalsobs[j],obstacle_size[j],action1]), slot_len = slot_len) for j in range(len(obstacle_size))]
#     primary_slot, secondary_slot, rule_mask, predicted_output, all_predicted_outputs, predicted_collision = m(primarySlots, secondarySlots, rules)
# #     primary_slot, secondary_slot, rule_mask, predicted_output = m(slots, rules)
#     print(predicted_output.detach().numpy(), inits[obj], finals[obj], action1, round(np.sum(np.abs(predicted_output.detach().numpy()-np.array(finals[obj]))),3)) #TODO made this :-1, shouldn't hard code it like this
    


# In[86]:


#TODO
# Maybe start saving whenever MAE is lowest ngl
allrules = nn.ModuleList()
for k in range(rule_num):
    allrules.append(rules[k].rule_mlp)
torch.save(allrules, '../learnt/RulesMLPv15.3.pth')
allrules = []
for k in range(rule_num):
    allrules.append(rules[k].rule_vec)
import pickle
f = open('../learnt/RulesVecv15.3.pkl', 'wb')
pickle.dump(allrules, f)
torch.save(m, '../learnt/Modelv15.3.pth')
import joblib
joblib.dump(pmodel, "../learnt/pmodelv15.3.joblib")


# In[ ]:




