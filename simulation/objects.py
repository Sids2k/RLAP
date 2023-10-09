import math
import random
import numpy as np
import os
import pybullet_data
from settings import Objects, Color, DiceUrdf, ColorList, object_dimensions

# class Mask():
#   def __init__(self, bullet, flags, position, color_idx = 1, orn = None) -> None:
#     orn = bullet.getQuaternionFromEuler([0, 0, np.random.normal(0, math.pi/24, 1)[0]]) if orn is None else orn
#     # objct = bullet.loadURDF(Objects.Block.value, position, orn, flags=flags, globalScaling=1)
#     # bullet.changeVisualShape(objct, -1, rgbaColor=ColorList[color_idx])
#     # Define the goal region dimensions and position
#     goal_dimensions = [0.2, 0.2, 0.02]  # Example dimensions [width, length, height]
#     goal_position = [0.5, 0.5, 0.02]  # Example position [x, y, z]

#     # Create the visual marker for the goal region
#     print("OKN TILL NOW")
#     markerId = bullet.createCollisionShape(bullet.GEOM_BOX, halfExtents=goal_dimensions)
#     markerVisualId = bullet.createMultiBody(baseCollisionShapeIndex=markerId, basePosition=goal_position)
#     bullet.changeVisualShape(markerVisualId, -1, rgbaColor=[1, 0, 0, 0.5])  # Set marker color to red (with transparency)

#     # Set collision properties for the marker
#     bullet.setCollisionFilterGroupMask(markerVisualId, -1, 0, 0)  # Disable collisions with all groups
#     bullet.setCollisionFilterGroupMask(markerVisualId, -1, 1, 1)  # Enable collisions with group 1 (cubes)

#     self.type = 'Mask'
#     self.object_idx = markerId
#     self.color = (color_idx, Color(color_idx).name)
#     self.position = list(position)
#     self.rotation = orn #rotation is measured in Quaternion
# @sids2k; increased weight of the mask and goal, and also fixed the orientation of it. did not reduce the collision shape but it works ok only imo
class Mask():
  def __init__(self, bullet, flags, position, color_idx = 1, orn = None) -> None:
    # @sids2k; hardcoded certain things like meshScale in colisionid and mass of the object
    orn = bullet.getQuaternionFromEuler([0, 0, 0]) if orn is None else orn
    visualid = bullet.createVisualShape(bullet.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(),'cube.obj'), meshScale=[object_dimensions['Mask']]*3, rgbaColor=ColorList[color_idx])
    colisionid = bullet.createCollisionShape(bullet.GEOM_MESH,fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[object_dimensions['Mask']/10]*3)
    objct = bullet.createMultiBody(baseMass=1e10,baseVisualShapeIndex=visualid,baseCollisionShapeIndex=colisionid,basePosition=position,baseOrientation=orn)
    # halfExtents=[0.005, 0.005, 0.005]
    # @sids2k; flags not used
    self.type = 'Mask'
    self.object_idx = objct
    self.color = (color_idx, Color(color_idx).name)
    self.position = list(position)
    self.rotation = orn #rotation is measured in Quaternion
    
class Goal():
  def __init__(self, bullet, flags, position, color_idx = 1, orn = None) -> None:
    # @sids2k; hardcoded certain things like meshScale in colisionid and mass of the object
    orn = bullet.getQuaternionFromEuler([0, 0, 0]) if orn is None else orn
    try:
      visualid = bullet.createVisualShape(bullet.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(),'cube.obj'), meshScale=[*object_dimensions['Goal']], rgbaColor=[*ColorList[color_idx][:-1],0.2])
      colisionid = bullet.createCollisionShape(bullet.GEOM_MESH,fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[x/10 for x in object_dimensions['Goal']])
    except:
      visualid = bullet.createVisualShape(bullet.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(),'cube.obj'), meshScale=[object_dimensions['Goal']]*3, rgbaColor=[*ColorList[color_idx][:-1],0.2])
      colisionid = bullet.createCollisionShape(bullet.GEOM_MESH,fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[object_dimensions['Goal']/10]*3)
    objct = bullet.createMultiBody(baseMass=1e10,baseVisualShapeIndex=visualid,baseCollisionShapeIndex=colisionid,basePosition=position,baseOrientation=orn)
    # halfExtents=[0.005, 0.005, 0.005]
    # @sids2k; flags not used
    self.type = 'Goal'
    self.object_idx = objct
    self.color = (color_idx, Color(color_idx).name)
    self.position = list(position)
    self.rotation = orn #rotation is measured in Quaternion
    
class Cube():
  def __init__(self, bullet, flags, position, color_idx = 1, orn = None) -> None:
    orn = bullet.getQuaternionFromEuler([0, 0, np.random.normal(0, math.pi/24, 1)[0]]) if orn is None else orn
    objct = bullet.loadURDF(Objects.Block.value, position, orn, flags=flags, globalScaling=1)
    bullet.changeVisualShape(objct, -1, rgbaColor=ColorList[color_idx])
    self.type = 'Cube'
    self.object_idx = objct
    self.color = (color_idx, Color(color_idx).name)
    self.position = list(position)
    self.rotation = orn #rotation is measured in Quaternion


class Dice():
  def __init__(self, bullet, flags, position, color_idx = 1, orn = None) -> None:
    orn = bullet.getQuaternionFromEuler([0, 0, np.random.normal(0, math.pi/24, 1)[0]]) if orn is None else orn
    objct = bullet.loadURDF(DiceUrdf[color_idx].value, position, orn, flags=flags, globalScaling=1)
    self.type = 'Dice'
    self.object_idx = objct
    self.color = (color_idx, Color(color_idx).name)
    self.position = list(position)
    self.rotation = orn #rotation is measured in Quaternion


class Lego():
  def __init__(self, bullet, flags, position, color_idx = 1, orn = None) -> None:
    orn = bullet.getQuaternionFromEuler([0, 0, np.random.normal(0, math.pi/24, 1)[0]]) if orn is None else orn
    objct = bullet.loadURDF(Objects.Lego.value, position, orn, flags=flags, globalScaling=2.10) ## 1.25*(31/18.5)
    bullet.changeVisualShape(objct, -1, rgbaColor=ColorList[color_idx])
    self.type = 'Lego'
    self.object_idx = objct
    self.color = (color_idx, Color(color_idx).name)
    self.position = list(position)
    self.rotation = orn #rotation is measured in Quaternion

class Tray():
  def __init__(self, bullet, flags, position, color_idx = 1, orn = None) -> None:
    objct = bullet.loadURDF(Objects.Tray.value, position, flags=flags, globalScaling=0.5)
    bullet.changeVisualShape(objct, -1, rgbaColor=ColorList[color_idx])
    self.type = 'Tray'
    self.object_idx = objct
    self.color = (color_idx, Color(color_idx).name)
    self.position = list(position)
    
