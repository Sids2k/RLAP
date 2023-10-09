import argparse
import os
import ast
import json
import pprint
from shapely.geometry import box
import numpy as np


parser = argparse.ArgumentParser()


parser.add_argument('--root_dir', default=os.getcwd(), metavar='DIR', help='Root Directory', type=str)
parser.add_argument('--file_name', default="v13copy.o3506608", metavar='DIR', help='File Name', type=str)
# add argument to check if 3 objects or 2 objects
parser.add_argument('--num_objects', default=3, type=int, help='Number of Objects')
checkingtechnique = 2

args = parser.parse_args()
objects = args.num_objects
file = open(os.path.join(args.root_dir, args.file_name), "r")
if args.file_name[-4:] == "dtxt":
    checkingtechnique = 3
objcoords = []
if checkingtechnique == 1:
    i = 0
    flag = False
    checkNext = False
    for line in file:
        if line[:2] == "AC":
            break
        if line[0] == "[" and line[1] == "(":
            obstacles_str = line
            # make a list of [(-0.5, -0.15), (-0.45, -0.15)]
            obstacles_list = ast.literal_eval(obstacles_str)
            # print(len(obstacles_list))
            checkNext = True
        elif checkNext:
            objcoords_str = line
            objcoords_list = objcoords_str.replace("(", "").replace(")", "").replace(",","").split()[:]
            objcoords.append([(float(objcoords_list[2*i]), float(objcoords_list[2*i+1])) for i in range(objects)])
            checkNext = False
            continue
        elif line[0] == '(' and flag is False:
    #         print(line)
            coords_str = line
            coords_list = coords_str.replace("(", "").replace(")", "").replace(",","").split()[2:]
            print(coords_list)
            objcoords.append([(float(objcoords_list[2*i]), float(objcoords_list[2*i+1])) for i in range(objects)])
        elif line[0] == '(' and flag is True:
            flag = False
        if line[:4]=="GOOD":
            i+=1
        elif line[:3]=="BAD":
    #         print(i,objcoords[:i])
    #         break
            objcoords = objcoords[:i]
            flag = True
if checkingtechnique == 2: #@sids2k with actual states and over states flag
    startCheck = False
    for line in file:
        if line[0] == "[" and line[1] == "(":
            obstacles_str = line
            # make a list of [(-0.5, -0.15), (-0.45, -0.15)]
            obstacles_list = ast.literal_eval(obstacles_str)
        if line[:2] == "AC":
            startCheck = True
            continue
        if line[:2] == "OV":
            startCheck = False
            continue
        if startCheck:
            objcoords_str = line
            objcoords_list = objcoords_str.replace("(", "").replace(")", "").replace(",","").split()[:]
            objcoords.append([(float(objcoords_list[2*i]), float(objcoords_list[2*i+1])) for i in range(objects)])
if checkingtechnique == 3: #@sids2k directly from list of tuples
    obsBool = False
    obstacles_list = []
    for line in file:
        if line[:3] == "OBS":
            obsBool = True
            continue
        if obsBool == True:
            obstacles_str = line
            obstacles_list = ast.literal_eval(obstacles_str)
            obsBool = False
            continue
        if obsBool == False:
            objcoords_str = line
            objcoords_list = objcoords_str.replace("(", "").replace(")", "").replace(",","").replace("[","").replace("]","").split()[:]
            if objcoords_list == []:
                continue
            objcoords.append([(float(objcoords_list[2*i]), float(objcoords_list[2*i+1])) for i in range(objects)])

a = objcoords
intial = [[x[0],x[1],0.64] for x in a[0]]
intial.extend([[x[0],x[1],0.64] for x in a[-1]])
#check if obstacle_list has overlapping squares (each square is 0.1x0.1) and remove them
# for i in range(len(obstacles_list)-1):
#     for j in range(i+1,len(obstacles_list)):
#         if(
#                 obstacles_list[i][0] < obstacles_list[j][0] + 0.1
#                 and obstacles_list[i][0] + 0.1 > obstacles_list[j][0]
#                 and obstacles_list[i][1] < obstacles_list[j][1] + 0.1
#                 and obstacles_list[i][1] + 0.1 > obstacles_list[j][1]
#             ):
#             obstacles_list.pop(j)
#             break
# def check_and_remove_overlapping_squares(obstacle_list):
#     # List to store non-overlapping squares
#     non_overlapping_list = []
#     removed_list = set()
#     # Iterate through each square in the obstacle list
#     for i in range(len(obstacle_list)):
#         current_square = obstacle_list[i]
#         is_overlapping = False

#         # Check for overlap with other squares
#         for j in range(i + 1, len(obstacle_list)):
            
#             other_square = obstacle_list[j]

#             # Check if the squares overlap
#             if (
#                 current_square[0] < other_square[0] + 0.1
#                 and current_square[0] + 0.1 > other_square[0]
#                 and current_square[1] < other_square[1] + 0.1
#                 and current_square[1] + 0.1 > other_square[1]
#             ):
#                 is_overlapping = True
#                 break

#         # If no overlap is found, add the square to the non-overlapping list
#         if not is_overlapping:
#             non_overlapping_list.append(current_square)
#         else:
#             removed_list.add(current_square)
#     return non_overlapping_list
# obstacles_list = check_and_remove_overlapping_squares(obstacles_list)


def remove_overlapping_obstacles(obstacle_list):
    obstacle_boxes = []  # List to store Shapely boxes representing the obstacles

    # Create Shapely boxes for each obstacle
    for obstacle in obstacle_list:
        x, y = obstacle
        obstacle_box = box(x - 0.05, y - 0.05, x + 0.05, y + 0.05)
        obstacle_boxes.append(obstacle_box)

    # Calculate the number of overlaps for each obstacle
    overlaps = [0] * len(obstacle_boxes)
    for i in range(len(obstacle_boxes)):
        for j in range(i + 1, len(obstacle_boxes)):
            if obstacle_boxes[i].intersects(obstacle_boxes[j]):
                overlaps[i] += 1
                overlaps[j] += 1

    # Remove obstacles with maximum overlaps iteratively
    non_overlapping_list = obstacle_list[:]
    while True:
        max_overlaps = max(overlaps)
        if max_overlaps <= 0:
            break

        max_index = overlaps.index(max_overlaps)
        non_overlapping_list.remove(obstacle_list[max_index])
        overlaps[max_index] = -1

        for i in range(len(obstacle_boxes)):
            if obstacle_boxes[max_index].intersects(obstacle_boxes[i]):
                overlaps[i] -= 1

    return non_overlapping_list

# obstacles_list = remove_overlapping_obstacles(obstacles_list)
# add obstacle_list to intial
for index,obs in enumerate(obstacles_list):
    # if index == 0:
    #     intial.append([*obs,0.64])
    # else:
    #     # check if new obstacle is within 0.05 distance of new obstacle. if so, don't add
    #     if abs(obs[0] - intial[-1][0]) > 0.05 or abs(obs[1] - intial[-1][1]) > 0.05:
    #         intial.append([*obs,0.64])
    intial.append([*obs,0.64])
    
borderobstacles = []
# make the border from -0.5,0.5 to 0.5,0.5 square
borderobstacles.extend([(-0.6, i) for i in np.arange(-0.6, 0.6, 0.1)])
borderobstacles.extend([(i, 0.6) for i in np.arange(-0.6, 0.6, 0.1)])
borderobstacles.extend([(0.6, i) for i in np.arange(-0.6, 0.6, 0.1)])
borderobstacles.extend([(i, -0.6) for i in np.arange(-0.6, 0.6, 0.1)])
for index,obs in enumerate(borderobstacles):
    intial.append([*obs,0.64])

plan = []
direction = []
for i in range(len(a)-1):
    # find which element changed from previous to next
    for j in range(len(a[i])):
        if a[i][j] != a[i+1][j]:
            # find the index of the element that changed
            index = j
            # find in which direction did the element change
            if a[i][j][0] < a[i+1][j][0] and abs(a[i][j][0] - a[i+1][j][0]) > 0.07:
                plan.append([index, [*a[i+1][index], 0.64]])
                direction.append("right")
            elif a[i][j][0] > a[i+1][j][0] and abs(a[i][j][0] - a[i+1][j][0]) > 0.07:
                plan.append([index, [*a[i+1][index], 0.64]])
                direction.append("left")
            elif a[i][j][1] < a[i+1][j][1] and abs(a[i][j][1] - a[i+1][j][1]) > 0.07:
                plan.append([index, [*a[i+1][index], 0.64]])
                direction.append("up")
            elif a[i][j][1] > a[i+1][j][1] and abs(a[i][j][1] - a[i+1][j][1]) > 0.07:
                plan.append([index, [*a[i+1][index], 0.64]])
                direction.append("down")
            else:
                print("no change")
                print(a[i][j], a[i+1][j])
    # add index of element that changed along with the new position to plan
    # print(a[i+1][index])
    # plan.append([index, [*a[i+1][index], 0.64]])
    # print(plan[-1])
    # if same index and same action, remove the earlier entry from plan
    for j in range(len(plan)-1):
        if plan[j][0] == plan[j+1][0] and direction[j] == direction[j+1]:
            plan.pop(j)
            direction.pop(j)
            break
# divide all values in inital by 2
for i in range(len(intial)):
    for j in range(len(intial[i])-1):
        intial[i][j] = intial[i][j]/2
# divide all coordinate values in plan by 2
for i in range(len(plan)):
    for j in range(len(plan[i][1])-1):
        plan[i][1][j] = plan[i][1][j]/2
        
colours = list(range(1,1+objects))
colours.extend([0 for i in range(objects)])
# add the value 7 to colours for each obstacle
for i in range(len(intial)-2*objects):
    colours.append(7)
# print(intial)
# print(plan)

# "TYPE[0] == 'Cube'", 
types = [f"TYPE[{i}] == 'Cube'" for i in range(objects)]
types.extend([f"TYPE[{i}] == 'Goal'" for i in range(objects, 2*objects)])
types.extend(["" for i in range(len(intial)-2*objects)])

jsondict = {
    "INITIAL_POS": intial,
    "FINAL_POS": plan,
    "Instructions": ["Planning time"],
    "Program": [f"{args.file_name}"],
    "num_objects": len(intial),
    "Colors": [f"COLOR[{i}] == {c}" for i,c in enumerate(colours)],
    "Types": types,
}
# open a json file to write the dictoinary
with open(f"{args.root_dir}/{args.file_name}.json", 'w') as write_file:
    json.dump(jsondict, write_file, indent=4, separators=(", ", ": "), ensure_ascii=False)

with open("curriculum.json", 'r+') as read_file:
    curriculum = json.load(read_file)
    print(curriculum["categories"])
    if f"{args.file_name}.json" not in [curriculum["categories"][i]["template"] for i in range(len(curriculum["categories"]))]:
        curriculum["categories"].append({"template":f"{args.file_name}.json", "count": 1})
    print(curriculum["categories"])
    #save the updated curriculum
    # delete the old json file contents
    read_file.seek(0)
    json.dump(curriculum, read_file, indent=4, separators=(", ", ": "), ensure_ascii=False)