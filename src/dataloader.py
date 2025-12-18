from torch.utils.data import Dataset
import os
import numpy as np
import torch
import cv2

class YoloDataset(Dataset):
    def __init__(self, data_dir, config):
        self.S = config['S']
        self.B = config['B']
        self.C = config['C']
        self.data_dir = data_dir
        self.files_list = [f.split('.jpg')[0] for f in os.listdir(data_dir) if f.endswith('.jpg')]
        self.files_list.sort()
    
    def __len__(self):
        return len(self.files_list)
    
    def read_label(self, file_path):
        # reads all the txt files in the folder
        # reads each line in the txt file
        # extracts the (label, x, y, w, h)
        # returns a list of objects 
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            objs = []
        for line in lines:
            values = line.strip().split()
            y = int(values[0])
            x_pos = float(values[1])
            y_pos = float(values[2])
            w = float(values[3])
            h = float(values[4])
            objs.append([y, x_pos, y_pos, w, h])
      
        return objs
    
    def label2tensor(self, objs):
        tensor = np.zeros((self.S, self.S, self.C + self.B*5))
        
        for obj in objs:
            y_class, x_pos, y_pos, w, h = obj
            cell_row = int(self.S * y_pos)
            cell_col = int(self.S * x_pos)
            
            # Keep the first object only
            if tensor[cell_row, cell_col, 4] == 0:
                x_cell = self.S * x_pos - cell_col
                y_cell = self.S * y_pos - cell_row
            
                for b in range(self.B):
                    tensor[cell_row, cell_col, b*5:b*5+5] = [x_cell, y_cell, w, h, 1]
            
                tensor[cell_row, cell_col, self.B*5 + y_class] = 1

        return torch.tensor(tensor, dtype=torch.float32)
    
    def __getitem__(self, idx):
        # Data loading function for torch.utils.data Dataset
        
        # Labels
        label_file = self.data_dir + '/' + self.files_list[idx] + '.txt'
        objs = self.read_label(label_file)
        label_tensor = self.label2tensor(objs)
        
        # Image
        image_file = self.data_dir + '/' + self.files_list[idx] + '.jpg'
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (448, 448))
        image_norm = image / 255.0
        image_tensor = torch.tensor(image_norm, dtype=torch.float32).permute(2, 0, 1)

        return image_tensor, label_tensor