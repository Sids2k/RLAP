import math
import random
import numpy as np
import json

from settings import *
from objects import Cube, Dice, Tray, Lego, Mask, Goal #@sids2k


# Panda Arm Position/Orientaion
PANDA_NUM_DOF = 7
END_EFFECTOR_INDEX = 8 
PANDA_POSITION = np.array([0, 0.5, 0.5]) # @sids2k, changed from [0, 0.5, 0.5]; need more change i suppose hmmm

# Inverse Kinematics
lower_limit = [-7]*PANDA_NUM_DOF
upper_limit = [7]*PANDA_NUM_DOF
joint_range = [7]*PANDA_NUM_DOF
rest_position = [0, 0, 0, -2.24, -0.30, 2.66, 2.32, 0, 0]

# Gripping
TABLE_OFFSET = 0.64
MOVE_HEIGHT = TABLE_OFFSET + 0.23 #0.2 or 0.4 is too high up (more than 1 block height); 0.1 is too low where gripper collides with blocks; 0.15 is perfect @sids2k
GRASP_WIDTH = 0.01
RELEASE_WIDTH = 0.05
GRASP_HEIGHT = 0.10
ALPHA = 0.99
DROP_MARGIN = 0.001

TIME_STEP = [1.25, 2.0, 1.5, 0.5, 1.5, 2.0, 1.5, 0.5, 1.5]
TIME_STEP = [1.25, 2.0, 2.0, 1.5, 2.0, 2.0, 1.5, 0.5, 1.5]
# TIME_STEP = [0.2, 0.8, 0.5, 0.3, 0.8, 0.2, 0.3, 0.2, 0.1]
# TIME_STEP = [0.2, 3.0, 0.5, 0.5, 0.5, 0.2, 0.5, 0.5, 0.5]
#@sids2k change time for each of the below actions (as shown in class PandaState)

TrayPositions = [
	[-0.5,  0.15, TABLE_OFFSET],
	[-0.5, -0.15, TABLE_OFFSET],
	[0.5,  0.15, TABLE_OFFSET],
	[0.5, -0.15, TABLE_OFFSET],
]

class PandaState(Enum):
	INIT = 0
	MOVE = 1
	PRE_GRASP = 2
	GRASP = 3
	POST_GRASP = 4
	MOVE_BW = 5
	PRE_RELEASE = 6
	RELEASE = 7
	POST_RELEASE = 8
	IDLE = 9

MAX_TRIES_RANDOM_POS = 100

class PandaWorld(object):
	def __init__(self, bullet_client, offset, height, width, template):
		# Client
		self.bullet_client = bullet_client
		self.offset = np.array(offset)
		self.flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
		flags = self.flags

		self.height = height
		self.width = width

		# Panda ARM
		panda_orn = self.bullet_client.getQuaternionFromEuler([0, 0, -math.pi/2])
		self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", self.offset + PANDA_POSITION, panda_orn, useFixedBase=True, flags=flags)
		finger_constraint = self.bullet_client.createConstraint(
			self.panda, 9, self.panda, 10,
			jointType=self.bullet_client.JOINT_GEAR,
			jointAxis=[1, 0, 0],
			parentFramePosition=[0, 0, 0],
			childFramePosition=[0, 0, 0])
		self.bullet_client.changeConstraint(finger_constraint, gearRatio=-1, erp=0.1, maxForce=50)
		self.panda_visible = False

		# Joint Panda Frame
		index = 0
		num_panda_joints = self.bullet_client.getNumJoints(self.panda)
		for joint_idx in range(num_panda_joints):
			self.bullet_client.changeDynamics(self.panda, joint_idx, linearDamping=0, angularDamping=0)
			info = self.bullet_client.getJointInfo(self.panda, joint_idx)
			jointType = info[2]
			if (jointType == self.bullet_client.JOINT_PRISMATIC) or (jointType == self.bullet_client.JOINT_REVOLUTE):
				self.bullet_client.resetJointState(self.panda, joint_idx, rest_position[index])
				index = index+1

		# Static Objects
		self.bullet_client.loadURDF("plane.urdf", offset, flags=flags)
		table = self.bullet_client.loadURDF("table/table.urdf", offset, flags=flags) # @sids2k, globalScaling=0.5
		self.bullet_client.changeVisualShape(table, -1, rgbaColor=[0.588, 0.435, 0.2, 1])

		# Maintain a list of all Objects
		self.objects = list()
		self.object_type = []

		self.PANDA_FINGER_HORIZONTAL = np.array([0, math.pi, math.pi/2])
		self.PANDA_FINGER_VERTICAL = np.array([0, math.pi, 0])

		# Initialize Variables
		self.t = 0
		self.control_dt = 1./240.
		self.state = PandaState.INIT
		self.finger_width = RELEASE_WIDTH
		self.gripper_height = MOVE_HEIGHT
		self.finder_orientation = self.PANDA_FINGER_VERTICAL
		self.block_position = None
		self.target_position = None
		self.pos = None

		# list of objects according to color and types
		if 'demo_json_path' in template:
			with open(template['demo_json_path'], 'r') as f:
				demo_data = json.load(f)
			for obj in demo_data['objects']:
				self.obj_orn = obj['rotation'] # assuming same rotation for all objects
				object_color_idx = obj['color'][0]
				self.object_type.append(obj['type'])
				if obj['type'] == 'Cube':
					self.objects.append(Cube(self.bullet_client, flags, obj['position'], object_color_idx, orn = obj['rotation']))
				elif obj['type'] == 'Dice':
					self.objects.append(Dice(self.bullet_client, flags, obj['position'], object_color_idx, orn = obj['rotation']))
				elif obj['type'] == 'Lego':
					self.objects.append(Lego(self.bullet_client, flags, obj['position'], object_color_idx, orn = obj['rotation']))
				elif obj['type'] == 'Tray':
					self.objects.append(Tray(self.bullet_client, flags, obj['position'], object_color_idx))
				elif obj['type'] == 'Mask':
					self.objects.append(Mask(self.bullet_client, flags, obj['position'], object_color_idx, orn = obj['rotation']))
				elif obj['type'] == 'Goal':
					self.objects.append(Goal(self.bullet_client, flags, obj['position'], object_color_idx, orn = obj['rotation']))
		else:
			COLOR = [-1 for _ in range(template['num_objects'])]
			TYPE = ['' for _ in range(template['num_objects'])]
			FINAL_POS = [self.offset + np.array([0, 0, TABLE_OFFSET]) for _ in range(template['num_objects'])]
			self.obj_orn = [0,0,0,1]

			self.status = True
			for i in range(template['num_objects']):
				# color_list = np.random.permutation(len(ColorList))
				color_list = [(x+i)%len(ColorList) for x in range(len(ColorList))]
				type_list = ['Mask', 'Cube', 'Goal']
				# random.shuffle(type_list)

				# Assign color
				color_found = False
				for color in color_list:
					COLOR[i] = int(color)
					color_expr = template['Colors'][i]
					if color_expr=='' or eval(color_expr):
						color_found = True
						break
				if not color_found:
					print('Colors not found')
					self.status = False
					return

				# Assign type
				type_found = False
				for type in type_list:
					TYPE[i] = type
					type_expr = template['Types'][i]
					if type_expr=='' or eval(type_expr):
						type_found = True
						break
				if not type_found:
					print('Types not found')
					self.status = False
					return
				self.object_type.append(TYPE[i])

				# Assign final position 
				# if i>0: # first object is always at the center so change that @sids2k
				# 	relation, base_obj_id = template['Relations'][i]
				# 	final_pos = self.relative_target_pos(relation, base_obj_id, i, FINAL_POS)
				# 	FINAL_POS[i] = final_pos

			# Check final positions non-overlapping
			# if not self.check_blocks_not_overlapping(FINAL_POS):
			# 	print('Overlapping final blocks')
			# 	self.status = False
			# 	return
			
			# Assign initial positions not overlapping with final positions
			# INITIAL_POS = []
			# for _ in range(template['num_objects']):
			# 	found_pos = False
			# 	for _ in range(MAX_TRIES_RANDOM_POS):
			# 		pos = self.get_random_table_position()
			# 		if self.check_new_block_not_overlapping(pos, INITIAL_POS + FINAL_POS):
			# 			INITIAL_POS.append(pos)
			# 			found_pos = True
			# 			break
			# 	if not found_pos:
			# 		print('No initial pos found')
			# 		self.status = False
			# 		return
						
			# Save objects info
			# FINAL_POS[:-2] = INITIAL_POS[:-2] # @sids2k
			# FINAL_POS[-1] = INITIAL_POS[-1] # @sids2k
			# INITIAL_POS = [[0,0,0.64],[0.11,0,0.64], [0.22,0,0.64], [0.11,0.11,0.64], [0.11,-0.11,1.5], [0.11,-0.11,0.64]] # @sids2k
			# INITIAL_POS = [[0,0,0.64],[0.11,0,0.64], [0.22,0,0.64], [0.11,0.11,0.64], [0.11,-0.22,0.64], [0.11,-0.11,0.64]] # @sids2k
			# FINAL_POS = [[0,0,0.64],[0,0.11,0.64], [0.22,0.22,0.64]] # @sids2k
			# FINAL_POS = [(0,[0.11,0,0.64]),(0,[0,0.11,0.64]), (1,[0.22,0.22,0.64])] # @sids2k
			INITIAL_POS = template["INITIAL_POS"]
			FINAL_POS = template["FINAL_POS"]
			self.final_positions = FINAL_POS
			
			for i in range(template['num_objects']):
				Object_class = eval(TYPE[i])
				self.objects.append(Object_class(self.bullet_client, flags, INITIAL_POS[i], COLOR[i], orn=self.obj_orn))
				print(i, self.objects[i].object_idx, COLOR[i], TYPE[i], INITIAL_POS[i])

	## -------------------------------------------------------------------------
	## Allocate Positions 
	## -------------------------------------------------------------------------

	def get_random_table_position(self):
		var_xy = WORK_AREA_SIDE*(np.random.rand(2) + np.array([-0.5, -0.5]))
		return self.offset + np.array([var_xy[0], var_xy[1], TABLE_OFFSET])

	def check_blocks_not_overlapping(self, block_positions):
		# TODO: currently thresholding is incorrect as block dimensions can be different. Make robust to different object dimensions
		return True
		num_blocks = len(block_positions)
		for i in range(num_blocks):
			for j in range(i + 1, num_blocks):
				x, y, _ = block_positions[i]-block_positions[j]
				max_d = max(abs(x), abs(y))
				if max_d < 0.075: return False
		return True
	
	def check_new_block_not_overlapping(self, new_block_position, block_positions):
		# TODO: currently thresholding is incorrect as block dimensions can be different. Make robust to different object dimensions
		# For current use case this works
		num_blocks = len(block_positions)
		for i in range(num_blocks):
			x, y, _ = block_positions[i]-new_block_position
			max_d = max(abs(x), abs(y))
			if max_d < 0.075: return False
		return True
	
	## -------------------------------------------------------------------------
	## Get Positions W.R.T. Objects
	## -------------------------------------------------------------------------

	def get_object_dim(self, object_target_id):
		'''
		Inputs:
			object_target_id: The id of the object in the pandaworld. Note that this not the object id w.r.t bullet client
		Return:(float) The object size. Currently, we are having cubes only. So a single number is enough.  
		'''
		obj_type = self.object_type[object_target_id]
		if obj_type in object_dimensions.keys():
			try:
				return object_dimensions[obj_type][-1]
			except:
				return object_dimensions[obj_type]
		else:
			print(f"Unrecognized Object Type in ConstructWorld:get_object_dim {obj_type}")
			raise TypeError()
		
	def relative_target_pos(self, relation, base_obj_id, move_obj_id, all_block_positions):
		handlers = {
			'TOP': self.top_target_pos,
			'LEFT': self.left_target_pos,
			'RIGHT': self.right_target_pos,
			'FRONT': self.front_target_pos
		}
		handler = handlers[relation]
		return handler(base_obj_id, move_obj_id, all_block_positions)
	

	def top_target_pos(self, base_obj_id:int, move_obj_id:int, all_block_positions:list):
		'''
			Inputs: (Note that the indices are not the object id w.r.t bullet client. The bullet may have additional objects like table and plane.)
				base_obj_id(int): Index of the id w.r.t the action is performed
				move_obj_id(int): The index of the object being moved
				all_block_positions:(list of 3-tuples) The list of positions of all objects in the PandaWorld
			Description: The target z co-ordinate = The left-bottom corner of base_obj + the height of the base object.
		'''
		base_pos = all_block_positions[base_obj_id]
		base_dim = self.get_object_dim(base_obj_id)
		move_z_pos = base_pos[2] + base_dim
		target_pos = [base_pos[0], base_pos[1], move_z_pos]
		return target_pos

	def left_target_pos(self, base_obj_id, move_obj_id, all_block_positions):
		'''
		Refer top_target_pos for input/output specs
		Description: The target x-cordiante = base_obj x co-ordinate - the length of the object being moved - MARGIN(to avoid overlapping). 
					 Here the length is same as that of the shape since the objects are cube. Refer panda.setting for more details
		'''
		base_pos = all_block_positions[base_obj_id]
		move_dim = self.get_object_dim(move_obj_id)
		move_left_pos = base_pos[0] - move_dim - MARGIN
		target_pos = [move_left_pos, base_pos[1], base_pos[2]]
		return target_pos

	def right_target_pos(self, base_obj_id, move_obj_id, all_block_positions):
		'''
		Refer top_target_pos for input/output specs
		Description: The target x-cordiante = base_obj x co-ordinate + the length of the base object - MARGIN(to avoid overlapping). 
					 Here the length is same as that of the shape since the objects are cube. Refer panda.setting for more details
		'''
		base_pos = all_block_positions[base_obj_id]
		base_dim = self.get_object_dim(base_obj_id)
		move_right_pos = base_pos[0] + base_dim + MARGIN
		target_pos = [move_right_pos, base_pos[1], base_pos[2]]
		return target_pos
	
	def front_target_pos(self, base_obj_id, move_obj_id, all_block_positions):
		'''
		Refer top_target_pos for input/output specs
		Description: The target x-cordiante = base_obj x co-ordinate + the length of the base object - MARGIN(to avoid overlapping). 
					 Here the length is same as that of the shape since the objects are cube. Refer panda.setting for more details
		'''
		base_pos = all_block_positions[base_obj_id]
		move_dim = self.get_object_dim(move_obj_id)
		move_front_pos = base_pos[1] - move_dim - MARGIN
		target_pos = [base_pos[0], move_front_pos, base_pos[2]]
		return target_pos

	def update_state(self):
		""" 
			The Control Time of 1 Second For Each State of Panda Execution
			INIT -> Initialization State
			MOVE -> Source State, IDLE -> Terminal State
		"""
		if (self.state != PandaState.IDLE):
			self.t += self.control_dt
			if (self.t > TIME_STEP[self.state.value]):
				self.t = 0
				if self.state == PandaState.INIT:
					self.state = PandaState.IDLE
				else:
					self.state = PandaState(self.state.value+1)

	def pre_execute_command(self):
		pass

	def executeCommand(self, block_pos, target_pos, move_obj_idx):
		""" Execute Command / Initiliaze State, dt """
		if self.state == PandaState.IDLE:
			self.pre_execute_command()
			self.block_position = block_pos
			self.target_position = target_pos
			self.obj_dim = self.get_object_dim(move_obj_idx)
			self.state = PandaState.MOVE
			self.t = 0

	def isExecuting(self):
		""" If the Panda is in the middle of command Execution """
		return self.state != PandaState.IDLE

	def movePanda(self, pos):
		"""
			Given The Position of the Panda (End_Effector_Part)
			Move The Panda/Joints to reach the location 
		"""
		# Move Panda Body
		self.bullet_client.submitProfileTiming("IK")
		orn = self.bullet_client.getQuaternionFromEuler(self.finder_orientation)
		jointPoses = self.bullet_client.calculateInverseKinematics(
			self.panda, END_EFFECTOR_INDEX, pos, orn, lower_limit, upper_limit, joint_range, rest_position)
		self.bullet_client.submitProfileTiming()
		control_mode = self.bullet_client.POSITION_CONTROL
		for idx in range(PANDA_NUM_DOF):
			self.bullet_client.setJointMotorControl2(self.panda, idx, control_mode, jointPoses[idx], force=1200., maxVelocity=1.0)
		# Move Gripper 
		control_mode = self.bullet_client.POSITION_CONTROL
		self.bullet_client.setJointMotorControl2(self.panda, 9, control_mode, self.finger_width, force=100)
		self.bullet_client.setJointMotorControl2(self.panda, 10, control_mode, self.finger_width, force=100)

	def get_nearest_below(self):
		nearest_obj_id, least_dist = None, float('inf')
		for id, obj in enumerate(self.objects):
			# check if below
			obj_dim = self.get_object_dim(id)
			below = True
			for i in range(2):
				below = below and obj.position[i] <= self.pos[i] + self.obj_dim and obj.position[i] >= self.pos[i] - obj_dim
			if below:
				dist = sum([(self.pos[i] - obj.position[i])**2 for i in range(3)])
				if dist < least_dist:
					least_dist = dist
					nearest_obj_id = id
		return nearest_obj_id

	def fine_correct_release(self):
		# find nearest object
		nearest_obj_id = self.get_nearest_below()
		if nearest_obj_id is not None:
			dim = self.get_object_dim(nearest_obj_id)
			nearest_obj_pos = self.objects[nearest_obj_id].position
			# print(nearest_obj_id, nearest_obj_pos[2], self.pos[2])
			self.pos[2] = max(self.pos[2], nearest_obj_pos[2] + dim + GRASP_HEIGHT + DROP_MARGIN)
			if self.adjust_horizontal:
				self.pos[0] = nearest_obj_pos[0]
				self.pos[1] = nearest_obj_pos[1]

	def fine_correct_grip(self):
		# find nearest object
		nearest_obj_id = self.get_nearest_below()
		if nearest_obj_id is not None:
			nearest_obj_pos = self.objects[nearest_obj_id].position
			# print(nearest_obj_id, nearest_obj_pos[2], self.pos[2])
			self.pos[2] = max(self.pos[2], nearest_obj_pos[2] + GRASP_HEIGHT)
			if self.adjust_horizontal:
				self.pos[0] = nearest_obj_pos[0]
				self.pos[1] = nearest_obj_pos[1]

	def step(self, gripper_error=False):

		alpha_change_gripper_height = lambda x: ALPHA * self.gripper_height + (1.0-ALPHA)* x
		# alpha_change_gripper_height = lambda x: x
		
		if self.state == PandaState.INIT:
			self.pos = np.zeros(3)
			self.gripper_height = alpha_change_gripper_height(MOVE_HEIGHT)
			self.pos[2] = self.gripper_height
			self.movePanda(self.pos)

		if self.state == PandaState.MOVE:
			# print(self.block_position)
			self.pos = self.block_position.copy()
			self.gripper_height = alpha_change_gripper_height(MOVE_HEIGHT)
			self.pos[2] = self.gripper_height
			self.movePanda(self.pos)

		elif (self.state == PandaState.PRE_GRASP):
			self.pos = self.block_position.copy()
			self.gripper_height = alpha_change_gripper_height(GRASP_HEIGHT+self.pos[2])
			self.pos[2] = self.gripper_height
			self.fine_correct_grip()
			self.movePanda(self.pos)

		elif self.state == PandaState.GRASP:
			self.finger_width = GRASP_WIDTH
			self.movePanda(self.pos)

		elif self.state == PandaState.POST_GRASP:
			self.pos = self.block_position.copy()
			self.gripper_height = alpha_change_gripper_height(MOVE_HEIGHT)
			self.pos[2] = self.gripper_height
			self.movePanda(self.pos)

		elif self.state == PandaState.MOVE_BW:
			self.gripper_height = alpha_change_gripper_height(MOVE_HEIGHT)
			for i in range(2):
				self.pos[i] = ALPHA * self.pos[i] + (1.0-ALPHA)*self.target_position[i]
			# self.pos = self.target_position.copy()
			self.pos[2] = self.gripper_height
			if gripper_error and random.random() < 0.01:
				self.finger_width = RELEASE_WIDTH
			self.movePanda(self.pos)

		elif self.state == PandaState.PRE_RELEASE:
			self.pos = self.target_position.copy()
			self.gripper_height = alpha_change_gripper_height(GRASP_HEIGHT + self.pos[2])
			self.pos[2] = self.gripper_height
			self.fine_correct_release()
			self.movePanda(self.pos)
			return self.pos[2] - GRASP_HEIGHT - DROP_MARGIN

		elif self.state == PandaState.RELEASE:
			self.finger_width = RELEASE_WIDTH
			self.movePanda(self.pos)

		elif self.state == PandaState.POST_RELEASE:
			self.pos = self.target_position.copy()
			self.gripper_height = alpha_change_gripper_height(MOVE_HEIGHT)
			self.pos[2] = self.gripper_height
			self.fine_correct_release()
			self.movePanda(self.pos)

		elif self.state == PandaState.IDLE:
			pass

		return None


