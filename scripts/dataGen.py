import argparse
import os

argparser = argparse.ArgumentParser()
argparser.add_argument('--seed', default=1, type=int, help='Random Seed, 1 for test 0 for train')
argparser.add_argument('--dir', default='demo_data/', type=str, help='Directory to save data')
argparser.add_argument('--filename', default='dataset', type=str, help='File name to save data without extension')
argparser.add_argument('-V','--version', default=-1, type=int, help='Version of data')
argparser.add_argument('--num_data', default=10000, type=int, help='Number of data points to generate')
argparser.add_argument('--length', default=1, type=float, help='Length of table')
argparser.add_argument('--width', default=1, type=float, help='Width of table')
argparser.add_argument('--infcoord', default=100.0, type=float, help='Infinite coordinate')
argparser.add_argument('--action_list', default=['up', 'down', 'left', 'right'], type=list, help='List of actions')
argparser.add_argument('--object_size', default=[(0.1,0.1)], type=list, help='List of object sizes')
argparser.add_argument('--obstacle_size', default=[(0.1,0.1)], type=list, help='List of obstacle sizes')
argparser.add_argument('--step_size', default=0.1, type=float, help='Step size')
argparser.add_argument('--noise_percent', default=0.01, type=float, help='Noise percent')
argparser.add_argument('--num_col', default=1000, type=int, help='Number of extra collisions (12 times the datapoints)')

args = argparser.parse_args()
seed = args.seed
dir = args.dir
filename = args.filename
num_data = args.num_data
length = args.length
width = args.width
infcoord = args.infcoord
action_list = args.action_list
object_size = args.object_size
obstacle_size = args.obstacle_size
step_size = args.step_size
noise_percent = args.noise_percent
num_col = args.num_col
version = args.version

if version >= 0:
    filename = filename + 'v' + str(version) + '.npy'
else:
    filename = filename + '.npy'

filepath = os.path.join(dir, filename)


import numpy as np
if seed is not None:
    np.random.seed(seed)
from tqdm import tqdm

def create_coordinate(length = 1, width = 1):
    '''
    Create (x,y) coordinates within table dimensions
    '''
    return np.random.uniform(-length/2, length/2), np.random.uniform(-width/2, width/2)

def random_action(action_list = ['up', 'down', 'left', 'right']):
    '''
    Randomly select an action from a list of actions
    '''
    return np.random.choice(action_list)
    
    
def new_coordinate(coordinate, action, step_size = 0.1, noise_percent = 0.1):
    '''
    Update the (x,y) coordinate based on the action
    '''
    noise = noise_percent * step_size
    if action == 'up':
        return coordinate[0] + np.random.randn() * noise, coordinate[1] + step_size + np.random.randn() * noise
    elif action == 'down':
        return coordinate[0] + np.random.randn() * noise, coordinate[1] - step_size + np.random.randn() * noise
    elif action == 'left':
        return coordinate[0] - step_size + np.random.randn() * noise, coordinate[1] + np.random.randn() * noise
    elif action == 'right':
        return coordinate[0] + step_size + np.random.randn() * noise, coordinate[1] + np.random.randn() * noise
    return coordinate

def origin_coordinate(length = 1, width = 1, step_size = 0.1, noise_percent = 0.1):
    return np.random.randn() * noise_percent * length / 100, np.random.randn() * noise_percent * width / 100

def validate_coordinate(coordinates, length = 1, width = 1):
    '''
    Check if the (x,y) coordinate is within the table dimensions
    '''
    for coordinate in coordinates:
        if not (-length/2 <= coordinate[0] <= length/2 and -width/2 <= coordinate[1] <= width/2):
            return False
    return True

def validate_overlap(objCoord, obsCoord, object_size, obstacle_size):
    allCoord = objCoord[:]
    allCoord.extend(obsCoord)
    allSize = object_size[:]
    allSize.extend(obstacle_size)
    for obj in range(len(allCoord)):
        objEdges = (allCoord[obj][0]+(allSize[obj][0])/2, allCoord[obj][0]-(allSize[obj][0])/2, allCoord[obj][1]+(allSize[obj][1])/2, allCoord[obj][1]-(allSize[obj][1])/2)
        for rest in range(len(allCoord[obj+1:])):
            rest+= obj+1
            currEdges = (allCoord[rest][0]+(allSize[rest][0])/2, allCoord[rest][0]-(allSize[rest][0])/2, allCoord[rest][1]+(allSize[rest][1])/2, allCoord[rest][1]-(allSize[rest][1])/2)
            if not (objEdges[0]<currEdges[1] or objEdges[1]>currEdges[0] or objEdges[2]<currEdges[3] or objEdges[3]>currEdges[2]):
                return False
    return True

datapoints = []
for d in tqdm(range(num_data)):
    inits = []
    finals = []
    for objs in range(len(object_size)):
        inits.append(create_coordinate(length = length, width = width)) 
        while not validate_overlap(inits, [], object_size, obstacle_size):
            inits[-1] = create_coordinate(length = length, width = width)
        #origin_coordinate(length = length, width = width, step_size = step_size, noise_percent = noise_percent)
        finals.append(inits[-1])
#     break
    initsobs = []
    finalsobs = []
    for obs in range(len(obstacle_size)):
        initsobs.append(create_coordinate(length = length, width = width))
        while not validate_overlap(inits, initsobs, object_size, obstacle_size):
            initsobs[-1] = create_coordinate(length = length, width = width)
        finalsobs.append(initsobs[-1])
    action = random_action(action_list = action_list)
    obj = np.random.choice(range(len(object_size)))
    finals[obj] = new_coordinate(inits[obj], action, step_size = step_size, noise_percent = noise_percent)
#     break
    while not validate_coordinate(finals, length = length, width = width):
        action = random_action(action_list = action_list)
        finals[obj] = new_coordinate(inits[obj], action, step_size = step_size, noise_percent = noise_percent)
    if not validate_overlap(finals, initsobs, object_size, obstacle_size):
        finals[obj] = (infcoord, infcoord)
    if action == 'up':
        action1 = 0
    elif action == 'down':
        action1 = 1
    elif action == 'left':
        action1 = 2
    elif action == 'right':
        action1 = 3
#     inits.extend(initsobs)
#     finals.extend(finalsobs)
    datapoints.append([inits, finals, object_size, initsobs, finalsobs, obstacle_size, obj, action1])
#     print(datapoints)
#     break


for d in tqdm(range(num_col)):
    for action in action_list:
        inits = []
        finals = []
        initsobs = []
        finalsobs = []
        inits.append(create_coordinate(length = length, width = width))
        finals.append((infcoord, infcoord))
        initsobs.append(new_coordinate(inits[0], action, step_size = step_size, noise_percent = noise_percent))
        finalsobs.append(initsobs[-1])
        for objs in range(len(object_size)-1):
            inits.append(create_coordinate(length = length, width = width)) 
            while not validate_overlap(inits, initsobs, object_size, obstacle_size):
                inits[-1] = create_coordinate(length = length, width = width)
            #origin_coordinate(length = length, width = width, step_size = step_size, noise_percent = noise_percent)
            finals.append(inits[-1])
    #     break
        
        for obs in range(len(obstacle_size)-1):
            initsobs.append(create_coordinate(length = length, width = width))
            while not validate_overlap(inits, initsobs, object_size, obstacle_size):
                initsobs[-1] = create_coordinate(length = length, width = width)
            finalsobs.append(initsobs[-1])
        #TODO here the obj is always 0, maybe i wanna change that in the future?
        obj = 0
        if action == 'up':
            action1 = 0
        elif action == 'down':
            action1 = 1
        elif action == 'left':
            action1 = 2
        elif action == 'right':
            action1 = 3
    #     inits.extend(initsobs)
    #     finals.extend(finalsobs)
        datapoints.append([inits, finals, object_size, initsobs, finalsobs, obstacle_size, obj, action1])
    #     print(datapoints)
        #
        
        # randomly choose one action from action_list that is not action
        temp_action_list = action_list[:]
        temp_action_list.remove(action)
        action2 = random_action(action_list = temp_action_list)
        
        # for all elements in action_list other than action
       
            
        inits = []
        finals = []
        initsobs = []
        finalsobs = []
        inits.append(create_coordinate(length = length, width = width))
        finals.append(new_coordinate(inits[0], action2, step_size = step_size, noise_percent = noise_percent))
        initsobs.append(new_coordinate(inits[0], action, step_size = step_size, noise_percent = noise_percent))
        finalsobs.append(initsobs[-1])
        for objs in range(len(object_size)-1):
            inits.append(create_coordinate(length = length, width = width)) 
            while not validate_overlap(inits, initsobs, object_size, obstacle_size):
                inits[-1] = create_coordinate(length = length, width = width)
            #origin_coordinate(length = length, width = width, step_size = step_size, noise_percent = noise_percent)
            finals.append(inits[-1])
    #     break
        
        for obs in range(len(obstacle_size)-1):
            initsobs.append(create_coordinate(length = length, width = width))
            while not validate_overlap(inits, initsobs, object_size, obstacle_size):
                initsobs[-1] = create_coordinate(length = length, width = width)
            finalsobs.append(initsobs[-1])
        #TODO here the obj is always 0, maybe i wanna change that in the future?
        obj = 0
        if action2 == 'up':
            action1 = 0
        elif action2 == 'down':
            action1 = 1
        elif action2 == 'left':
            action1 = 2
        elif action2 == 'right':
            action1 = 3
    #     inits.extend(initsobs)
    #     finals.extend(finalsobs)
        datapoints.append([inits, finals, object_size, initsobs, finalsobs, obstacle_size, obj, action1])
    #     print(datapoints)
        
        
        
        inits = []
        finals = []
        initsobs = []
        finalsobs = []
        inits.append(create_coordinate(length = length, width = width))
        finals.append(new_coordinate(inits[0], action, step_size = step_size, noise_percent = noise_percent))
        initsobs.append(new_coordinate(finals[0], action, step_size = step_size, noise_percent = noise_percent))
        finalsobs.append(initsobs[-1])
        for objs in range(len(object_size)-1):
            inits.append(create_coordinate(length = length, width = width)) 
            while not validate_overlap(inits, initsobs, object_size, obstacle_size):
                inits[-1] = create_coordinate(length = length, width = width)
            #origin_coordinate(length = length, width = width, step_size = step_size, noise_percent = noise_percent)
            finals.append(inits[-1])
    #     break
        
        for obs in range(len(obstacle_size)-1):
            initsobs.append(create_coordinate(length = length, width = width))
            while not validate_overlap(inits, initsobs, object_size, obstacle_size):
                initsobs[-1] = create_coordinate(length = length, width = width)
            finalsobs.append(initsobs[-1])
        #TODO here the obj is always 0, maybe i wanna change that in the future?
        obj = 0
        if action == 'up':
            action1 = 0
        elif action == 'down':
            action1 = 1
        elif action == 'left':
            action1 = 2
        elif action == 'right':
            action1 = 3
    #     inits.extend(initsobs)
    #     finals.extend(finalsobs)
        datapoints.append([inits, finals, object_size, initsobs, finalsobs, obstacle_size, obj, action1])
    #     break


datapoints = np.array(datapoints, dtype = object)
np.random.seed(seed=seed)
np.random.shuffle(datapoints)
np.save(filepath, datapoints)

