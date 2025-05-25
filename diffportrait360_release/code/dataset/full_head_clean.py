import json
import cv2
import numpy as np

from torch.utils.data import Dataset
import os 
import random
import torch
from torchvision import transforms as T
from PIL import Image, ImageFile
# read a image through PIL and transfer it to tensor
ImageFile.LOAD_TRUNCATED_IMAGES = True
print("using load_truncated_images, to avoid bug")
#import pdb;pdb.set_trace()
class full_head_clean_real_data_temporal(Dataset):
    def __init__(self, root_path, image_transform=None, sample_frame=8, flag='real', more_image_control = True):
        # super()
        self.root = root_path
        self.flag = flag
        self.potential_face_list = [ '10', '12', '13','14', '15', '16' ,'17','18', '19','20', '21', '22', '23', '24', '25', '26', 
                                    '27','28', '29', '30', '31','32','33', '34' ,'35','36' '37','38', '54', '55', '56','57'
                                      ]
        self.potential_face_list_PH = ['00', '01', '02', '03', '04', '05','06','07','08','09','10','11',
                                       '25','26','27','28','29', '30', '31', '32', '33', '34', '35']
        self.potential_NS_fine_face_list = ['16','18','19','25','26','28','31','55','56' ]
        self.potential_NS_fine_back_list = ['59','50','49','48','46','45','01','00','02']
        self.potential_PH_fine_face_list = [str(i) for i in range(0,9)]  + [str(j) for j in range(63,71)]
        #self.potential_PH_fine_face_list([])##['00','01','02','03','04','05','06','32','33','34','35' ]
        self.potential_PH_fine_back_list = [str(i) for i in range(29,46)]#['16','17','18','19','20','21']
        self.id_list = []
        self.more_image_control = more_image_control
        self.id_driving_list_PH = []
        # for id_name in sorted(os.listdir(self.root)):
        #     if id_name.startswith('PH_'):
        #         camera_folder = os.path.join(self.root, id_name, 'condition')
        #     else:
        #         camera_folder = os.path.join(self.root, id_name, 'camera')
        #     if not os.path.isdir(os.path.join(self.root, id_name, 'image')):
        #         continue
        #     # if image folder doenst have image, skip
        #     elif not any(file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) for file in os.listdir(camera_folder)):
        #         continue
        #     elif id_name.startswith('seed'): # remove the PanoHead stupid shit !
        #         continue
        #     else:
        #         self.id_list.append(id_name)
        for id_name in sorted(os.listdir(self.root)):
            if id_name.startswith('PHsup_'): # remove the
                self.id_driving_list_PH.append(id_name)  
            if id_name.startswith('PHsup_'): #or id_name.startswith('0') or id_name.startswith('i'): # remove the
            #if id_name.startswith('seed'):
                # if id_name.startswith("i"):
                #     # check the subfolder if it had a subfolder called images
                #     image_folder = os.path.join(self.root, id_name,'images')
                #     if not any(file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) for file in os.listdir(image_folder)):
                #         print("not have a png folder in:", image_folder)
                #         continue
                    # elif int(id_name.split("_")[1])>1000:
                    #     continue 
                    # else:
                    #    self.id_list.append(id_name)
                #else:
                    self.id_list.append(id_name)
            else:
                continue
        self.transform = image_transform
        self.sample_frame = sample_frame
    def __len__(self):
        
        return 100000
    
    def __getitem__(self, idx):
        if self.sample_frame == 1:
            raise KeyError("sample_frame should be at least larger than 1")
            
        len_id_list = len(self.id_list)
        #print('len_id_list totall:', len_id_list)
        idx = random.randint(0, len_id_list-1)
        extra_appearance_list = []
        id_name = self.id_list[idx]
        #
        len_id_drive_PH = len(self.id_driving_list_PH)
        idx_drive = random.randint(0, len_id_drive_PH-1)
        id_drive_name_PH = self.id_driving_list_PH[idx_drive]    
        if id_name.startswith('PHsup_'):
            gt_path_ = os.path.join(self.root, id_name, "image")
            valid_face_list, valid_all_list = self.check_face(gt_path_, 'PH', potential_face_list = self.potential_PH_fine_face_list)
            valid_no_face_list, _ = self.check_face(gt_path_, 'PH', potential_face_list =self.potential_PH_fine_back_list)
            condition_image_path_ = os.path.join(self.root, id_drive_name_PH, "image")
            
        elif id_name.startswith('0'): # for nersemble dataset
            gt_path_ = os.path.join(self.root, id_name, "image_seg")
            valid_face_list, valid_all_list = self.check_face(gt_path_, 'real', potential_face_list =self.potential_NS_fine_face_list)
            valid_no_face_list, _ = self.check_face(gt_path_, 'real', potential_face_list =self.potential_NS_fine_back_list)
            condition_image_path_ = os.path.join(self.root, id_name, "camera")
            
        elif id_name.startswith('i'): # stylization dataset
            valid_face_list = ['0.png']
            valid_all_list = ['0.png','1.png','2.png','3.png']
            condition_dict = {'0.png':'0000000.png','1.png':'0000008.png','2.png':'0000018.png','3.png':'0000027.png'}
            valid_no_face_list = ['2.png']
            gt_path_ = os.path.join(self.root, id_name, "images")
            condition_image_path_ = os.path.join(self.root, id_drive_name_PH, "condition")
        random.shuffle(valid_all_list)
        random.shuffle(valid_face_list)
        random.shuffle(valid_no_face_list)
        # appearance frame idx should be randomly picked from 0-9 27-35 
        if id_name.startswith('i'):
            appearance_frame_idx = '0.png'
            if self.more_image_control:
                more_appearance_frame_idx = '2.png'
            frames_list = valid_all_list
        else:
            valid_exlcude_face_list = list(set(valid_all_list) - set(valid_face_list))
            random.shuffle(valid_exlcude_face_list)
            appearance_frame_idx = valid_face_list[0].zfill(7)+'.png'
            frames_list = valid_all_list[:self.sample_frame]
            #print(id_name, " ")
            #print(valid_face_list[0])
            if self.more_image_control:
                if len(valid_no_face_list) == 0:
                    more_appearance_frame_idx = valid_exlcude_face_list[-1].zfill(7)+'.png'
                    print('idName:', id_name, 'appearance:', appearance_frame_idx, ', doesnt have no back image')
                more_appearance_frame_idx = valid_no_face_list[0].zfill(7)+'.png'
        # gt _path_
        
        apperance  = self.read_image(os.path.join(gt_path_, appearance_frame_idx))
        targets = []
        conditions = []
        interval = random.randint(2, 6)
        frames_list = self.pick_frames(num_frames=self.sample_frame, total_frames=72, interval=interval)
        #
        print('frame_list:', frames_list[0:4], 'interval:', interval)
        for frame in frames_list:
            if not id_name.startswith('i'):
                target_frame_idx = str(frame).zfill(7)+'.png' 
                #source_path = camera_folders[1]
                condition = self.read_image(os.path.join(condition_image_path_, target_frame_idx))         
            else:
                target_frame_idx = frame  
                condition = self.read_image(os.path.join(condition_image_path_, condition_dict[target_frame_idx]))         
            target = self.read_image(os.path.join(gt_path_, target_frame_idx))            
   
            # mask condition here
            if self.transform is not None:
                target = self.transform(target)  
                condition = self.transform(condition)
            #condition = np.ones_like(target).astype(np.uint8)
            targets.append(target)
            conditions.append(condition)
        targets = torch.stack(targets)
        conditions = torch.stack(conditions)
        prompt = ''
        if self.transform is not None: 
            condition_image = self.transform(apperance)      
        
        # mask condition here
        
        #condition = np.ones_like(target).astype(np.uint8) 
        
        res = {'image': targets, 'condition_image': condition_image, 'condition': conditions,'text_bg':prompt, 'text_blip': prompt}
        if self.more_image_control:
            more_appearance_frame = self.read_image(os.path.join(gt_path_, more_appearance_frame_idx))
            if self.transform is not None:
                more_appearance_frame = self.transform(more_appearance_frame) # we only consider
            #extra_appearance_list.append(more_appearance_frame)
            res['extra_appearance'] = more_appearance_frame        
        return res
    def read_image(self, path):
        #print("using_path:", path)
        # Open the image file
        image = Image.open(path)
        
        # Ensure the image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize the image to 512x512
        image = image.resize((512, 512))
        
        # Convert the PIL image to a NumPy array
        image_np = np.array(image)
        
        return image_np
    def check_face(self, id_path, flags, potential_face_list=None):  
        valid_face = []
        for face_id in potential_face_list:
            image_path = os.path.join(id_path, face_id.zfill(7)+'.png')
            if os.path.isfile(image_path):
                valid_face.append(face_id.zfill(7))
        valid_all = []
        for face_id in os.listdir(id_path):
            if face_id.endswith('.png'):
                valid_all.append(face_id.split('.')[0])
        return valid_face, valid_all  
    def pick_frames(self, num_frames, total_frames=72, interval=None):
        # 如果没有指定间隔，则默认为1（即连续帧）
        if interval is None:
            interval = 1
        
        # 随机选择一个起始帧
        start_frame = random.randint(0, total_frames - 1)
        
        # 计算连续四帧
        selected_frames = [(start_frame + i * interval) % total_frames for i in range(num_frames)]
        
        return selected_frames

class back_head_generation(Dataset):
    def __init__(self, inference_image_dataset, condition_path, image_transform=None):
        self.root = inference_image_dataset
        self.condition_path = condition_path
        self.id_list = []
        for id_name in sorted(os.listdir(self.root)):
            if id_name.endswith('.png') or id_name.endswith('.jpg'):
                self.id_list.append(id_name)
        self.transform = image_transform
        
    def __len__(self):
        
        return len(self.id_list)
    
    def __getitem__(self, idx):
        id_name = self.id_list[idx]
        #target_frame_idx = str().zfill(7)+'.png'
        #gt_path_ = os.path.join(self.root, id_name)
        condition_image_path_ = os.path.join(self.condition_path, "0000016.png")
        condition = self.read_image(condition_image_path_)
        #condition = self.read_image(os.path.join(self.condition_path, id_name))#self.read_image(condition_image_path_)
        appearance_path = os.path.join(self.root, id_name)
        appearance = self.read_image(appearance_path)

        prompt = ''
        if self.transform is not None:
            #target = self.transform(target)  
            condition_image = self.transform(appearance)
            condition = self.transform(condition)        
        fea_condition = condition_image
        res = {'image': condition_image, 'condition_image': condition_image, 'condition': condition,'text_bg':prompt, 'text_blip': prompt, "image_name": id_name, 'fea_condition': fea_condition}
        
        return res
    def read_image(self, path):
        #print("using_path:", path)
        # Open the image file
        image = Image.open(path)
        
        # Ensure the image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize the image to 512x512
        image = image.resize((512, 512))
        
        # Convert the PIL image to a NumPy array
        image_np = np.array(image)
        
        return image_np
    

class full_head_clean_real_data_single_frame(Dataset):
    def __init__(self, root_path, image_transform=None, flag='real', more_image_control = False):
        self.root = root_path
        self.flag = flag
        self.potential_face_list = [ '10', '12', '13','14', '15', '16' ,'17','18', '19','20', '21', '22', '23', '24', '25', '26', 
                                    '27','28', '29', '30', '31','32','33', '34' ,'35','36' '37','38', '54', '55', '56','57'
                                      ]
        self.potential_face_list_PH = ['00', '01', '02', '03', '04', '05','06','07','08','09','10','11',
                                       '25','26','27','28','29', '30', '31', '32', '33', '34', '35']
        self.potential_PH_fine_face_list = ['00','01','02','03','04','05','06','32','33','34','35' ]
        self.potential_PH_fine_back_list = ['15','16','17','18','19','20','21']
        self.id_list = []
        self.id_driving_list_PH = []
        for id_name in sorted(os.listdir(self.root)):
            #if id_name.startswith('PH_'):
            #    camera_folder = os.path.join(self.root, id_name, "condition")
            #else:
            #    camera_folder = os.path.join(self.root, id_name, "camera")
                
            #if not os.path.isdir(os.path.join(self.root, id_name,'image')):
            #    continue

            #elif not any(file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) for file in os.listdir(camera_folder)):
            #    continue
            if id_name.startswith('PH_'): # remove the
                self.id_driving_list_PH.append(id_name)  
            #if id_name.startswith('seed') or id_name.startswith('0') or id_name.startswith('i'): 
            #if id_name.startswith('seed'):
            if id_name.startswith('seed'): # remove the styliazation
                self.id_list.append(id_name)
            else:
                continue
        # if appearance inside of the potential_face_list, then generate 

        self.transform = image_transform
        self.more_image_control = more_image_control
        #import pdb;pdb.set_trace() 
    def __len__(self):
        
        return 100000
    def check_face(self, id_path, potential_face_list=None):  
        valid_face = []
        for face_id in potential_face_list:
            image_path = os.path.join(id_path, face_id.zfill(7)+'.png')
            if os.path.isfile(image_path):
                valid_face.append(face_id.zfill(7))
        valid_all = []
        for face_id in os.listdir(id_path):
            if face_id.endswith('.png') and not face_id.endswith('_parsing.png'):
                valid_all.append(face_id.split('.')[0])
        return valid_face, valid_all
    def __getitem__(self, idx):
        #id_name = self.id_list[idx]
        len_id_list = len(self.id_list)
        idx = random.randint(0, len_id_list-1)

        #extra_appearance_list = []
        id_name = self.id_list[idx]
        len_id_drive_PH = len(self.id_driving_list_PH)
        idx_drive = random.randint(0, len_id_drive_PH-1)
        id_drive_name_PH = self.id_driving_list_PH[idx_drive]
        
        # for nersemble

        if id_name.startswith('seed'):
            gt_path_ = os.path.join(self.root, id_name)
            valid_face_list, valid_all_list = self.check_face(gt_path_, potential_face_list = self.potential_PH_fine_face_list)
            valid_no_face_list, _ = self.check_face(gt_path_, potential_face_list =self.potential_PH_fine_back_list)
            #id_drive_name_PH = 'seed0010'
            condition_image_path_ = os.path.join(self.root, id_drive_name_PH, "image")
        elif id_name.startswith('0'):
            gt_path_ = os.path.join(self.root, id_name,'image_seg')
            valid_face_list, valid_all_list = self.check_face(gt_path_, potential_face_list =self.potential_NS_fine_face_list)
            valid_no_face_list, _ = self.check_face(gt_path_, potential_face_list =self.potential_NS_fine_back_list)
            #condition_image_path_ = os.path.join(self.root, id_drive_name_NS, "image")
            condition_image_path_ = os.path.join(self.root, id_name,'camera')
            #check with condition_image_path if the for driving
        elif id_name.startswith('i'):
            gt_path_ = os.path.join(self.root, id_name, "images")
            condition_image_path_ = os.path.join(self.root, id_drive_name_PH, "image")
            #valid_face_list = ['0']

        if id_name.startswith('i'):
            appearance_frame_idx = '0.png'
            target_frame_idx = '2.png'
            condition = self.read_image(os.path.join(condition_image_path_, '0000018.png'))
        else:
            random.shuffle(valid_all_list)
            random.shuffle(valid_face_list)
            random.shuffle(valid_no_face_list)
            #valid_exlcude_face_list = list(set(valid_all_list) - set(valid_face_list))
            #random.shuffle(valid_exlcude_face_list)
            if len(valid_face_list) == 0:
                print('idName:', id_name,  ', doesnt have face image')
            appearance_frame_idx = valid_face_list[0].zfill(7)+'.png'
            target_frame_idx = valid_no_face_list[-1].zfill(7)+'.png'
            # gt _path_
            condition = self.read_image(os.path.join(condition_image_path_, target_frame_idx))
        
        apperance  = self.read_image(os.path.join(gt_path_, appearance_frame_idx))

        target = self.read_image(os.path.join(gt_path_, target_frame_idx))
    
        prompt = ''
        if self.transform is not None:
            target = self.transform(target)  
            condition_image = self.transform(apperance)
            condition = self.transform(condition)        
        res = {'image': target, 'condition_image': condition_image, 'condition': condition,'text_bg':prompt, 'text_blip': prompt}
        return res
    def read_image(self, path):
        #print("using_path:", path)
        # Open the image file
        image = Image.open(path)
        
        # Ensure the image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize the image to 512x512
        image = image.resize((512, 512))
        
        # Convert the PIL image to a NumPy array
        image_np = np.array(image)
        
        return image_np






