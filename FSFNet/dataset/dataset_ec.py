import torch.utils.data as data
import cv2
import numpy as np
import pandas as pd
import os
import torch



class ECdataset(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform

        NAME_COLUMN = 0
        LABEL_COLUMN = 1

        if self.train:
            dataset = pd.read_csv(os.path.join(self.root, 'tacnn/next_speaker/labels/next_speaker_train.csv'), sep=',')
        else:
            dataset = pd.read_csv(os.path.join(self.root, 'tacnn/next_speaker/labels/next_speaker_val.csv'), sep=',')

        if self.dataidxs is not None:
            file_names = np.array(dataset.iloc[:, NAME_COLUMN].values)[self.dataidxs]
            target = np.array(dataset.iloc[:,LABEL_COLUMN].values)[self.dataidxs]
        else:
            file_names = dataset.iloc[:, NAME_COLUMN].values
            target = dataset.iloc[:,LABEL_COLUMN].values

        self.file_paths = []
        self.targets = []



        for f,t in zip(file_names, target):

            self.tmp_file_paths = []
            self.tmp_targets = []
            for i in range(1,5):
                path = os.path.join(self.root, 'tacnn/next_speaker/last_frames_out/'+ f[:-4] + str(i) + "_video_aligned_" +  "frame_det_00_000300.bmp")
                if os.path.exists(path):
                    self.tmp_file_paths.append(path)
                    self.tmp_targets.append(t)
                else:
                    print("Error: " + f[:-4] + str(i))

            self.file_paths.append(self.tmp_file_paths)
            self.targets.append(self.tmp_targets)

    def __len__(self):
        return len(self.file_paths)

    def get_labels(self):
        return self.targets

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        imgs = []
        for subpath in path:
            image = cv2.imread(subpath)
            image = image[:, :, ::-1]  # BGR to RGB
            target = self.targets[idx]

            if self.transform is not None:
                image = self.transform(image)
            
            imgs.append(image)
        
        imgs = torch.tensor(np.array(imgs))
        lbs = torch.tensor(np.array(target))
        
        return imgs, lbs