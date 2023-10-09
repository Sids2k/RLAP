import os
import cv2
import numpy as np
from PIL import Image
from world import PandaWorld
from objects import Cube, Dice, Lego
from world import PandaState
from settings import camera_settings
import time

class ConstructBase(PandaWorld):
	def __init__(self, bullet_client, offset, height, width, template, instance_dir, set_hide_panda_body=True):
		super().__init__(bullet_client, offset, height, width, template)
		self.instance_dir = instance_dir
		self.mapping = list()
		if set_hide_panda_body:
			self.hide_panda_body()
		self.position_list = list()

		while self.state == PandaState.INIT:
			self.update_state()
			self.step()
			bullet_client.stepSimulation()

	def hide_panda_body(self):
		for link_idx in range(-1, 11):
			self.bullet_client.changeVisualShape(self.panda, link_idx, rgbaColor=[0, 0, 0, 0])
		self.panda_visible = False
	
	def show_panda_body(self):
		for link_idx in range(-1, 11):
			self.bullet_client.changeVisualShape(self.panda, link_idx, rgbaColor=[1., 1., 1., 1.])
		self.panda_visible = True

	## -------------------------------------------------------------------------
	## Saving Instances and Demonstration Info
	## -------------------------------------------------------------------------
	def depth_color_map(self, depth):
		depth = (depth - 0.90)*10*255
		depth_colormap = cv2.convertScaleAbs(depth)
		return depth_colormap

	def save_instance(self, use_panda):
		w = self.width
		h = self.height
		instance_dir = self.instance_dir

		# Get Instance Folder
		demo_states = [os.path.join(instance_dir, f) for f in os.listdir(instance_dir)]
		demo_states = [s for s in demo_states if os.path.isdir(s)]
		instance_folder = "S" + "{0:0=2d}".format(len(demo_states))
		os.mkdir(os.path.join(instance_dir, instance_folder))

		# for view in ['user', 'top', 'right', 'robot']:
		# @sids2k change view
		# for view in ['top']:
		for view in ['user']:
			if view == 'user':
				self.bullet_client.resetDebugVisualizerCamera(**camera_settings['small_table_view'])
			elif view == 'top':
				self.bullet_client.resetDebugVisualizerCamera(**camera_settings['top_view'])
			elif view == 'right':
				self.bullet_client.resetDebugVisualizerCamera(**camera_settings['right_view'])
			elif view == 'robot':
				self.bullet_client.resetDebugVisualizerCamera(**camera_settings['robot_view'])

			# for panda_visible in [True, False]:
			for panda_visible in [True]:
				if (not use_panda) and panda_visible:
					continue
				if not panda_visible:
					self.hide_panda_body()
				else:
					self.show_panda_body()

				_, _, rgba, depth, mask = self.bullet_client.getCameraImage(width = w, height = h)

				type_folder = view + '-panda' if panda_visible else view
				folder = os.path.join(instance_dir, instance_folder, type_folder)
				os.mkdir(folder)
				img_filepath = os.path.join(folder, f"rgba.png")
				depth_filepath = os.path.join(folder, f"depth.png")
				mask_filepath = os.path.join(folder, f"mask.png")

				# Get ColorMaps
				mask[mask < 3] = 0
				mask = cv2.convertScaleAbs(mask)
				depth = self.depth_color_map(depth)

				# Save Images
				Image.fromarray(rgba, 'RGBA').save(img_filepath)
				Image.fromarray(depth, 'L').save(depth_filepath)
				Image.fromarray(mask, 'L').save(mask_filepath)

		if use_panda:
			self.show_panda_body()
		# @sids2k change view
		# self.bullet_client.resetDebugVisualizerCamera(**camera_settings['top_view'])
		self.bullet_client.resetDebugVisualizerCamera(**camera_settings['small_table_view'])

	
	def get_scene_info(self):
		info = {}
		info['objects'] = []
		for idx,obj in enumerate(self.objects):
			current_object = {}
			current_object["type"] = obj.type
			current_object["object_idx"] = idx 
			current_object["color"] = obj.color
			current_object['position'] = self.position_list[0][idx]
			current_object["rotation"] = obj.rotation # here rotation is measured in Quaternions
			info['objects'].append(current_object)
		info['object_color'] = [o.color[0] for o in self.objects]
		info['object_type'] = [o.type for o in self.objects]
		info['object_positions'] = self.position_list
		return info
		
	def save_position_info(self):
		l = [o.position for o in self.objects]
		self.position_list.append(l)

	## -------------------------------------------------------------------------
	## Applying Programs :
	## -------------------------------------------------------------------------

	def is_clear(self, object_positions, target_position, skip):
		for i in range(0, len(object_positions)):
			if i == skip: continue
			assert type(object_positions) == list and type(target_position) == list
			x, y, z = list(np.array(object_positions[i]) - np.array(target_position))
			max_d = max(abs(x), abs(y), abs(z))
			if max_d < 0.075: return False
		return True

	def move_object(self, move_obj_idx, target_pos, use_panda = False, adjust_horizontal=False):
		""" Moves Object (By Index) To The Target Position
			Inputs: 
				move_obj_id(int): The index of the object being moved
				target_pos(3-tuple): The target position of the moved object 
				use_panda(bool): If False, the object will be moved by deleting and creating a new object in the target position. 
								 If True, the panda robot will be simulated to move the object to the target. 	
		"""
		if use_panda == True:
			# print(move_obj_idx, target_pos, use_panda, timeStep, initial_pos, adjust_horizontal, gripper_error)
			m_object = self.objects[move_obj_idx]
			obj_id = m_object.object_idx
			pos_init, _ = self.bullet_client.getBasePositionAndOrientation(obj_id)
			m_object.position = list(pos_init)
			# @sids2k ; I have added this condition to check if the object is already at the target position
			if abs(pos_init[0] - target_pos[0]) < 1e-2 and abs(pos_init[1] - target_pos[1]) < 1e-2 and abs(pos_init[2] - target_pos[2]) < 1e-2:
				print("At target position ",move_obj_idx)
				return True, False
			self.executeCommand(m_object.position, target_pos, move_obj_idx)
			self.adjust_horizontal = adjust_horizontal
			stuck = False
			reachable = False
			while self.state != PandaState.IDLE:
				# print(self.state)
				self.update_state()
				self.step()
				if self.state == PandaState.POST_GRASP:
					pos, _ = self.bullet_client.getBasePositionAndOrientation(obj_id)
					if pos[2] > 0.8:
						reachable = True
				if self.state == PandaState.POST_RELEASE:
					pos, _ = self.bullet_client.getBasePositionAndOrientation(obj_id)
					if pos[2] > 0.8:
						stuck = True
				self.bullet_client.stepSimulation()
				time.sleep(self.control_dt)
			
			pos, _ = self.bullet_client.getBasePositionAndOrientation(obj_id)

			for i in range(len(pos)):
				if abs(m_object.position[i] - pos[i]) > 1e-2:
					reachable = True
					break
			m_object.position = list(pos) 
			return reachable, stuck
		else:
			m_object = self.objects[move_obj_idx]
			self.bullet_client.removeBody(m_object.object_idx)
			color_idx = m_object.color[0]
			if m_object.type == 'Cube':
				self.objects[move_obj_idx] = Cube(self.bullet_client, self.flags, target_pos, color_idx,orn = self.obj_orn)
			elif m_object.type == 'Dice':
				self.objects[move_obj_idx] = Dice(self.bullet_client, self.flags, target_pos, color_idx, orn = self.obj_orn)
			elif m_object.type == 'Lego':
				self.objects[move_obj_idx] = Lego(self.bullet_client, self.flags, target_pos, color_idx, orn = self.obj_orn)
			else:
				print(f"No Implementation for Object Type:{m_object.type} in move_object")
				raise NotImplementedError()
			return True, False
