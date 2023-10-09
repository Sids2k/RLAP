import os
import argparse
import pybullet as p
import pybullet_data as pd
import numpy as np
from settings import camera_settings, ColorList
from program_generator import ProgramGenerator
import traceback

def remove_sample(s):
	os.system(f"rm {s} -r")
	print(".......Sample Deleted: ", s,".......")

def str2bool(string):
    s = string.lower()
    if s in ['true', 'yes', 't', '1']:
        return True
    elif s in ['false', 'no', 'f', '0']:
        return False
    else:
        raise ValueError("Couldn't convert the string {string} into boolean")

def construct_main(template_file, fps=240., width=1024, height=768, dataset_dir='dir', use_panda=True):
  """
  :Parameters:
  fps: float, Dataset Directory From Root'
  width: Width of GUI Window
  height: Height of GUI Window
  dataset_dir: Relative path to the Dataset Directory
  objects: list, Types of objects required in the scene
  template_file: str, template file for generating program and instructions
  """

  #set parameters of GUI window
  timeStep = 1./fps

  if(not os.path.isdir(dataset_dir)):
    os.mkdir(dataset_dir)

  sample_no = len(os.listdir(dataset_dir))
  smpl_dir = os.path.join(dataset_dir, "{0:0=4d}".format(sample_no))
  os.mkdir(smpl_dir)
  
  
  Generator = DataConstructor(timeStep)
  p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "simulation_video.mp4")
  Generator.construct_data(height, width, smpl_dir, template_file, use_panda)
  p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)
  p.disconnect()


def init_bulletclient(timeStep, width=None, height=None, video_filename=None):
  #connection_mode GUI = graphical mode, DIRECT = non-graphical mode
  if video_filename is None:
    p.connect(p.GUI)
  else:
    p.connect(p.GUI, options = f"--minGraphicsUpdateTimeMs=0 --width={width} --height={height} --mp4=\"{video_filename}\" --mp4fps=48")
  #Add pybullet_data path to search path to load urdf files
  p.setAdditionalSearchPath(pd.getDataPath())
  # p.setAdditionalSearchPath("./urdf/")
  
  #visualizer and additional settings 
  p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
  p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
  p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
  p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
  p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
  p.setPhysicsEngineParameter(solverResidualThreshold=0)
  # @sids2k change view
  p.resetDebugVisualizerCamera(**camera_settings['small_table_view'])
  # p.resetDebugVisualizerCamera(**camera_settings['top_view'])
  p.setTimeStep(timeStep)
  p.setGravity(0, 0, -10.0)
  p.setRealTimeSimulation(0)

class DataConstructor(object):
  def __init__(self,timeStep):
    init_bulletclient(timeStep)

  def construct_data(self, height, width, smpl_dir, template_file, use_panda):
    construct = ProgramGenerator(p,[0,0,0],height,width,smpl_dir,template_file = template_file)
    if not construct.status:
      remove_sample(smpl_dir)
      return
    
    status = construct.move_objects_to_final_positions(use_panda)
    if not status:
      remove_sample(smpl_dir)
      return

    construct.save_demonstration_info()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fps', default=240., type=float, help='Dataset Directory From Root')
  parser.add_argument('--width', default=1024, help='Width of GUI Window')
  parser.add_argument('--height', default=768, help='Height of GUI Window')
  parser.add_argument('--root_dir', default=os.getcwd(), metavar='DIR', help='Root Directory')
  parser.add_argument('--dataset_dir', metavar='DIR', default='data', help='Relative path to the Dataset Directory')
  parser.add_argument('--template_file', required = True, type = str, help = "template file for generating program and instructions")
  parser.add_argument('--num_examples', type=int, required=True)
  parser.add_argument('--use_panda', type=str2bool, default=True, help='Whether to use Panda during data generation')
  args = parser.parse_args()

  for i in range(args.num_examples):
    try:
      construct_main(args.template_file, args.fps, args.width, args.height, args.dataset_dir, args.use_panda)
    except:
      traceback.print_exc()
