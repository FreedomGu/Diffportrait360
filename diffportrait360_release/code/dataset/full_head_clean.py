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
        condition_image_path_ = os.path.join(self.condition_path, "0000018.png")
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
    

class full_head_clean_inference_final_face(Dataset):
    def __init__(self, condition_path, image_transform=None, inference_image_dataset= None, initial_image_path = None):
        self.condition_path = condition_path
        self.inference_image_dataset = inference_image_dataset
        self.initial_image_path = initial_image_path
        #CHECK IF IT IS A INFERENCE BASED 
        if self.inference_image_dataset is not None:
            self.id_list = []
            self.inference_image_dataset_back_hair = os.path.join(self.inference_image_dataset, 'Back_Head')
            for img in sorted(os.listdir(self.inference_image_dataset_back_hair)):
                if img.endswith('.png') or img.endswith('.jpg'):
                    self.id_list.append(img)
        #else:      
        #    self.id_list = []
        #    for id_name in sorted(os.listdir(self.root+"_test")):
        #        self.id_list.append(id_name)
        self.transform = image_transform
        self.frame_rate = frame_rate = len([i for i in os.listdir(self.condition_path)])
    def __len__(self):
        return len(self.id_list)
    def __getitem__(self, idx):
        print("using_id:", self.id_list[idx])
        id_name = self.id_list[idx]
        extra_appearance_list = []
        frame_rate = self.frame_rate
        #gt_path_ = os.path.join(self.inference_image_dataset, id_name, "image")
        if self.inference_image_dataset is not None:
            # Check the appearance flag and read the required number of images
            appearance = self.read_image(os.path.join(self.inference_image_dataset, 'input_image', id_name))
            if self.transform is not None:
                    condition_image = self.transform(appearance)
        condition_image_path_ = self.condition_path #os.path.join(self.root, 'conditions', 'sphere')#os.path.join(self.root, '0000000', "gt")
        gt_img_list = []
        drive_img_list = []
        
        if self.initial_image_path is not None:
            fea_condition = []
            fea_condition_path = os.path.join(self.initial_image_path, id_name.split(".")[0], 'condition'+str(frame_rate)+'.mp4')
            cap = cv2.VideoCapture(fea_condition_path)
            frame_number = 0
            # Check if the video has been opened successfully
            if not cap.isOpened():
                print("Error opening video file")
            else:
                # Loop through each frame in the video
                while True:
                    ret, frame = cap.read()  # Read the frame
                    if not ret:
                        break  # Break the loop if there are no frames left to rea
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    inital_image = self.transform(image)
                    fea_condition.append(inital_image)
                    # Increment the frame number
                    frame_number += 1

                # Release the video capture object
                cap.release()
            
            # Print out the total number of frames saved
            print(f"Frames extracted: {frame_number}")
            #import pdb;pdb.set_trace()
        else:
            print(f"Not using feature Condition")
            fea_condition = None
        #self.initial_image_path  = '/mnt/motherfucker/yuming/Nerf_noise/pti'
   
        for i in range(frame_rate):               
            target_frame_idx = str(i).zfill(7)+'.png'
            #source_path = camera_folders[1]
            condition = self.read_image(os.path.join(condition_image_path_, target_frame_idx))
            
            if self.transform is not None:
                #target = self.transform(target)  
                condition = self.transform(condition)
            gt_img_list.append(condition)
            drive_img_list.append(condition)
            
            # if self.initial_image_path is not None:
            #     initial_image = condition

            #     id_name_ = id_name.split('.')[0]
            #     initial_image = self.read_image(os.path.join(self.initial_image_path, target_frame_idx))
            #     # if self.inference_image_dataset.split('_')[-2].endswith('sphere'):
            #     #     initial_image = self.read_image(os.path.join(self.initial_image_path, id_name_, target_frame_idx))
            #     # else:
            #     #     # if sequence 
            #     #     if i < 10 or i > 25:
            #     #         initial_image = self.read_image(os.path.join(self.inference_image_dataset, 'Front_Face', id_name))
            #     #     else:
            #     #         initial_image = self.read_image(os.path.join(self.inference_image_dataset, 'Generated_Face', id_name))
            #     print('reaqding noise from ',fea_condition_path )
            #     if self.transform is not None:
            #         initial_image = self.transform(initial_image)
            #     fea_condition.append(initial_image)
                
            # mask condition here
            
            #condition = np.ones_like(target).astype(np.uint8)
        prompt = '' #smile, open mouth, high quality'
        if fea_condition is not None:
            fea_condition = torch.stack(fea_condition)
        else:
            fea_condition = []
        res = {'image': gt_img_list, 'condition_image': condition_image, 'condition': drive_img_list,'text_bg':prompt, 'text_blip': prompt, 'fea_condition': fea_condition, "image_name": id_name}
        #if self.more_image_control:
        #if id_name.endswith('.jpg'):
        #    id_name= id_name.split(".")[0]+".png"
        
        more_appearance_frame = self.read_image(os.path.join(self.inference_image_dataset, 'Back_Head', id_name))
        #assert more_appearance_frame os.path.isfile(fname), f"File not found: {fname}"
        assert os.path.isfile(os.path.join(self.inference_image_dataset, 'Back_Head', id_name)), f"File not found: {os.path.join(self.inference_image_dataset, 'Back_Head', id_name)}"
        if self.transform is not None:
            more_appearance_frame = self.transform(more_appearance_frame) # we only consider
            extra_appearance_list.append(more_appearance_frame)
            #print("using edxtra_appera ce", os.path.join(self.inference_image_dataset, 'Back_Head', id_name))
        res['extra_appearance'] = more_appearance_frame       
        return res            
        
    def read_image(self, path):
        #print("validation_image_path:", path)
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

