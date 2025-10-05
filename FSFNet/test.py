import warnings
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
warnings.filterwarnings("ignore")
import numpy as np
from torchvision import transforms
import os
from sklearn.svm import SVC

import torch
import argparse
from dataset.dataset_ec import ECdataset

from sklearn.metrics import f1_score
from time import time
from utils.load_weights import *
from utils.sam import SAM
import torch.nn as nn

from models.FSFNet import FSFNet
from dataset.randomaug import RandAugment

# log
import datetime
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H:%M]-")

import random
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_random_seed(42)

# It should contain in folder /frames all the face crops in '.bmp' format
DATASET_PATH = '../Dataset/'
log_txt_path = 'log/' + time_str + 'set' + '-log.txt'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])


data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    RandAugment(1,5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(scale=(0.02, 0.1)),
])

data_transforms_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


if __name__ == "__main__":

    """
    Evaluation script for FSFNet and ensemble with ML classifiers.
    The script loads pretrained models, evaluates them on the test dataset,
    and reports performance metrics (Accuracy, Macro F1, Composite Score).
    """

    google_drive_path = '/Users/giulia_huang/Library/CloudStorage/GoogleDrive-ruoxingiuliahuang@gmail.com/My Drive/eye_contact/intermediate/models'
    pretrained_ml_model_path = '../MLmodels/models'
    
     # Here you should insert tuples of (model_path, feature_set)
    # [('combined-dataset-spos-all_testonval-True_feats-gaze_of',['gaze_of'])]
    # model_paths = []

    ml_model_path = None # Model name
    feature_column = None # Feature set

    # Parameters
    model_weight = 0.4
    to_print = []

    # Load dataset
    test_dataset = ECdataset(
        root=DATASET_PATH,
        data_path="test_final.csv",
        feature_columns=feature_column,
        train=False,
        transform=data_transforms_val,
    )

    print('Test set size:', test_dataset.__len__())

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        num_workers=2,
        shuffle=False,
        pin_memory=True
    )
    

    # Variable initialization
    pre_labels, gt_labels, ml_pre_labels, combined_pre_labels = [], [], [], []
    fsfnet_prob_list, ml_prob_list = [], []

    use_gpu = False
    finetuned = False
    device = 'cuda' if use_gpu else 'cpu'

    test_loss = 0.0
    correct_predictions = 0
    total_samples = len(test_dataset)
    CE_criterion = nn.CrossEntropyLoss()

    # Load FSFNet model
    model_path = (
        "./models/finetuned/best_finetuned_4_adjusted.pth"
        if finetuned
        else "./models/pretrained/best.pth"
    )
    model = FSFNet(img_size=112, num_classes=4, type='large')
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("FSFNet Model loaded.")

    # Load ML model
    scaler = joblib.load(os.path.join(pretrained_ml_model_path,"scaler.joblib"))
    ml_model = joblib.load(os.path.join(pretrained_ml_model_path,'best_estimator_ec.joblib'))
    print(f"{type(ml_model)} Model loaded")
    to_print.append(f"model type: {type(ml_model)}")
    

    for imgs, features, targets in test_loader:

        imgs, targets = imgs.to(device), targets.to(device)

        # FSFNet predictions
        outputs, _ = model(imgs)
        loss = CE_criterion(outputs, targets)
        test_loss += loss.item()

        fsfnet_preds = torch.argmax(outputs, dim=1)  # More efficient
        fsfnet_probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        correct_predictions += torch.eq(fsfnet_preds, targets).sum().item()

        if ml_model_path:

            # ML model predictions
            features_scaled = scaler.transform(features.numpy())
            ml_preds = ml_model.predict(features_scaled)
            ml_probs = ml_model.predict_proba(features_scaled)

            # Combined predictions
            avg_probs = (1 - model_weight) * fsfnet_probs + model_weight * ml_probs
            combined_preds = np.argmax(avg_probs, axis=1)

            # Store labels
            ml_pre_labels.extend(ml_preds.tolist())
            ml_prob_list.append(ml_probs)
            combined_pre_labels.extend(combined_preds.tolist())

        # Store labels
        gt_labels.extend(targets.tolist())
        pre_labels.extend(fsfnet_preds.tolist())
        fsfnet_prob_list.append(fsfnet_probs)


    # FSFNet results
    test_loss /= total_samples
    test_acc = correct_predictions / total_samples
    fsfnet_f1 = f1_score(gt_labels, pre_labels, average="macro")
    fsfnet_score = 0.67 * fsfnet_f1 + 0.33 * test_acc

    to_print.append("\n[FSFNet Results]")
    to_print.append(classification_report(gt_labels, pre_labels, digits=4))
    to_print.append(
        f"Accuracy: {test_acc:.4f}, Loss: {test_loss:.3f}, "
        f"Macro F1: {fsfnet_f1:.4f}, Score: {fsfnet_score:.4f}"
    )

    if ml_model_path:

        # ML results
        ml_acc = accuracy_score(gt_labels, ml_pre_labels)
        ml_f1 = f1_score(gt_labels, ml_pre_labels, average="macro")
        ml_score = 0.67 * ml_f1 + 0.33 * ml_acc

        to_print.append("\n[ML Results]")
        to_print.append(classification_report(gt_labels, ml_pre_labels, digits=4))
        to_print.append(
            f"Accuracy: {ml_acc:.4f}, Macro F1: {ml_f1:.4f}, Score: {ml_score:.4f}"
        )

        # Combined results
        combined_acc = accuracy_score(gt_labels, combined_pre_labels)
        combined_f1 = f1_score(gt_labels, combined_pre_labels, average="macro")
        combined_score = 0.67 * combined_f1 + 0.33 * combined_acc

        to_print.append("\n[Combined FSFNet + ML Results]")
        to_print.append(f"Ensemble weight: {model_weight} (ML) + {1-model_weight} (FSFNet)")
        to_print.append(classification_report(gt_labels, combined_pre_labels, digits=4))
        to_print.append(
            f"Accuracy: {combined_acc:.4f}, Macro F1: {combined_f1:.4f}, "
            f"Score: {combined_score:.4f}"
        )

    for line in to_print:
        print(line)
        # with open("./best_finetuned_ensemble.log", "a") as f:
        #     f.write(line + "\n")

