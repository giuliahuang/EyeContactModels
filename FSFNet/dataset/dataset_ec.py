import torch.utils.data as data
import cv2
import numpy as np
import pandas as pd
import os
import torch



class ECdataset(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, data_path=None, data_df=None, transform=None, csv_name=None, feature_columns=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.data_path = data_path
        self.data_df = data_df
        self.cols4featuresets = {
            'pose_of':['pose_Tx','pose_Ty','pose_Tz','pose_Rx','pose_Ry','pose_Rz'],
            'gaze_of':['gaze_0_x','gaze_0_y','gaze_0_z','gaze_1_x','gaze_1_y','gaze_1_z'],
            'poseT_of':['pose_Tx','pose_Ty','pose_Tz'],
            'poseR_of':['pose_Rx','pose_Ry','pose_Rz'],
            'speaker_info':['speaker_0','speaker_1','speaker_2','speaker_3','speaker_4'],
            'is_speaker_info': ['is_speaker'],
        }

        NAME_COLUMN = 0
        LABEL_COLUMN = 1

        if self.data_df is not None:
            dataset = self.data_df
        else:
            df = pd.read_csv(os.path.join(self.root, self.data_path), sep=',')
            df = df[df['data_subset'] == 'train'] if self.train else df[df['data_subset'] != 'train']

        if self.dataidxs is not None:
            file_names = np.array(dataset.iloc[:, NAME_COLUMN].values)[self.dataidxs]
            target = np.array(dataset.iloc[:,LABEL_COLUMN].values)[self.dataidxs]
        else:
            file_names = df['sample_index'].values
            target = df['ec_relative'].values.astype(int)

        self.feature_columns = self.get_feature_cols(feature_columns) if feature_columns is not None else None

        # Extract structured features
        if self.feature_columns is None:
            # Default: all columns after 2nd
            # self.structured_features = df.iloc[:, 2:].values.astype(np.float32)
            self.structured_features = [None for _ in range(len(file_names))]
        else:
            self.structured_features = df[self.feature_columns].values.astype(np.float32)

        self.file_paths = []
        self.targets = []
        self.features = []

        for f,feature, t in zip(file_names, self.structured_features, target):
            path = os.path.join(self.root, 'frames/'+ f +'.bmp')
            if os.path.exists(path):
                self.file_paths.append(path)
                self.targets.append(t)
                self.features.append(feature)
            else:
                print("Error: " + f)

    def get_feature_cols(self, featuresets):
        featuresets = sorted(featuresets)
        feature_cols = []
        for featureset in featuresets:
            feature_cols += self.cols4featuresets[featureset]
        return feature_cols

    def __len__(self):
        return len(self.file_paths)

    def get_labels(self):
        return self.targets

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        feature = self.features[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        target = self.targets[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, target