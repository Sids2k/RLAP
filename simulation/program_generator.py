
import json
from base import ConstructBase
import random

class ProgramGenerator(ConstructBase):
    def __init__(self, bullet_client, offset, height, width, instance_dir, template_file, set_hide_panda_body=False):
        self.template_file = template_file
        self.template = self.load_template(template_file)
        super().__init__(bullet_client, offset, height, width, self.template, instance_dir, set_hide_panda_body)
        self.save_position_info()

    def load_template(self,template_file):
        with open(template_file,'r') as f:
            template = json.load(f)
        return template
    
    def move_objects_to_final_positions(self, use_panda):
        if not use_panda:
            self.hide_panda_body()
        self.save_instance(use_panda) #save the initial scene
        
        # for i in range(len(self.objects)):
        #     reachable, stuck = self.move_object(i, self.final_positions[i], use_panda)
        #     if (not reachable) or stuck:
        #         return False
        #     self.save_instance(use_panda) # save intermediate scene
        #     self.save_position_info()
        # @sids2k
        for id, pos in self.final_positions:
            reachable, stuck = self.move_object(id, pos, use_panda)
            if (not reachable) or stuck:
                return False
            self.save_instance(use_panda) # save intermediate scene
            self.save_position_info()
            
        # check all objects at intended positions (geometrically for now)
        # @sids2k removed check for final positions due to difference in self.final_positions structure
        # for i in range(len(self.objects)):
        #     cur_pos = self.objects[i].position
        #     target_pos = self.final_positions[i]
        #     for j in range(len(cur_pos)):
        #         if abs(cur_pos[j] - target_pos[j]) > 1e-2:
        #             return False
                
        return True 
    
    def save_demonstration_info(self):
        info = dict()
        scene_info = self.get_scene_info()
        info.update(scene_info)
        info['template_json_filename'] = self.template_file
        info['instruction'] = random.sample(self.template['Instructions'], 1)
        info['program'] = self.template['Program']
        info['num_objects'] = self.template['num_objects']
        with open(f"{self.instance_dir}/demo.json", 'w') as write_file:
            json.dump(info, write_file)

