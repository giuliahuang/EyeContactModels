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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ec', help='dataset')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=128, help='Batch size for validation.')
    parser.add_argument('--modeltype', type=str, default='large', help='small or base or large')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.000004, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=300, help='Total training epochs.')
    parser.add_argument('--gpu', type=str, default='0', help='assign multi-gpus by comma concat')
    return parser.parse_args()


import random
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

args = parse_args()
set_random_seed(42)

log_txt_path = 'log/' + time_str + 'set' + str(args.dataset) + '-log.txt'

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
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


num_classes = 4
datapath = '/Volumes/照片/GiuliaDataset/Annotations/'
# datapath = '/Users/giulia_huang/Library/CloudStorage/GoogleDrive-ruoxingiuliahuang@gmail.com/My Drive/eye_contact/intermediate'
#datapath = '/Users/giulia_huang/Desktop/models/baseline/eye_contact/intermediate'
# train_dataset = ECdataset(datapath, train=True, transform=data_transforms)


# Correct
if __name__ == "__main__":

    google_drive_path = '/Users/giulia_huang/Library/CloudStorage/GoogleDrive-ruoxingiuliahuang@gmail.com/My Drive/eye_contact/intermediate/models'
    
    # model_paths=[
                # ('combined-dataset-spos-all_testonval-True_feats-gaze_of',['gaze_of']),
                # ('combined-dataset-spos-all_testonval-True_feats-gaze_of-is_speaker_info',['gaze_of','is_speaker_info']),
                # ('combined-dataset-spos-all_testonval-True_feats-gaze_of-is_speaker_info-pose_of',['gaze_of','pose_of','is_speaker_info'])]
                # ('combined-dataset-spos-all_testonval-True_feats-gaze_of-is_speaker_info-pose_of-speaker_info',['gaze_of','pose_of','is_speaker_info','speaker_info']),
                # ('combined-dataset-spos-all_testonval-True_feats-gaze_of-is_speaker_info-speaker_info',['gaze_of','pose_of','speaker_info']),
                # ('combined-dataset-spos-all_testonval-True_feats-gaze_of-pose_of',['pose_of','gaze_of']),
                # ('combined-dataset-spos-all_testonval-True_feats-gaze_of-pose_of-speaker_info',['gaze_of','pose_of','speaker_info']),
                # ('combined-dataset-spos-all_testonval-True_feats-gaze_of-speaker_info',['gaze_of','speaker_info']),
                # ('combined-dataset-spos-all_testonval-True_feats-is_speaker_info-pose_of',['pose_of','is_speaker_info']),
                # ('combined-dataset-spos-all_testonval-True_feats-is_speaker_info-pose_of-speaker_info',['pose_of','is_speaker_info','speaker_info']),
                # ('combined-dataset-spos-all_testonval-True_feats-pose_of',['pose_of'])]
                # ('combined-dataset-spos-all_testonval-True_feats-pose_of-speaker_info',['pose_of','speaker_info'])]

    model_paths = [
        # ('combined-svc-dataset-spos-all_testonval-True_feats-gaze_of',['gaze_of']),
        # ('combined-svc-dataset-spos-all_testonval-True_feats-gaze_of-is_speaker_info-speaker_info',['gaze_of','is_speaker_info','speaker_info']),
        ('combined-dataset-spos-all_testonval-True_feats-gaze_of-is_speaker_info-speaker_info',['gaze_of','is_speaker_info','speaker_info'])
    ]

    for xgb_model_path, feature_column in model_paths:

        # for w in [0.1,0.2,0.3,0.4,0.5]:
        w = 0.4
        to_print = []
        feature_columns = feature_column
        val_dataset = ECdataset(root=datapath, data_path = 'test_final.csv', csv_name='test_final.csv', feature_columns=feature_columns, train=False, transform=data_transforms_val)
        # val_dataset = ECdataset(root=datapath, data_path = 'combined_data.csv', csv_name='combined_data.csv', feature_columns=feature_columns, train=False, transform=data_transforms_val)

        #model = FSFNet(img_size=112, num_classes=num_classes, type=args.modeltype)

        val_num = val_dataset.__len__()
        # print('Train set size:', train_dataset.__len__())
        print('Validation set size:', val_dataset.__len__())

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=args.val_batch_size,
                                                num_workers=args.workers,
                                                shuffle=False,
                                                pin_memory=True)

        pre_labels = []
        gt_labels = []
        xgb_pre_labels = []
        # xgb_gt_labels = []
        combined_pre_labels = []
        # combined_gt_labels = []

        # Placeholder lists for testing logic
        fsfnet_prob_list = []
        xgb_prob_list = []
        meta_targets = []

        test_loss = 0.0
        correct_predictions = 0
        total_samples = len(val_dataset)
        CE_criterion = nn.CrossEntropyLoss()
        use_gpu = False
        device = 'cuda' if use_gpu else 'cpu'

        model = FSFNet(img_size=112, num_classes=4, type='large')

        model_path = './models/pretrained/best.pth' # original FSFNet
        # model_path = './models/finetuned/best_finetuned_4_adjusted.pth' # best finetuned FSFNet


        print("Loading model")

        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])

        print("FSFNet Model loaded")

        # Load trained XGBoost model
        #root_path = '/Users/giulia_huang/Desktop/models/baseline/eye_contact/intermediate/models/testing-spos-all_testonval-True_feats-gaze_of-pose_of/'
        #root_path = './models/SVC'

        root_path = os.path.join(google_drive_path,xgb_model_path)
        scaler = joblib.load(os.path.join(root_path,"scaler.joblib"))
        xgb_model = joblib.load(os.path.join(root_path,'best_estimator_ec.joblib'))
        to_print.append(f"model type: {type(xgb_model)}")
        model.eval()


        # Accumulate features and labels
        # all_features = []
        # all_targets = []

        # for batch_i, (imgs, feature, targets) in enumerate(val_loader):
        #     all_features.append(feature.numpy())        # Assuming 'feature' is a torch.Tensor
        #     all_targets.append(targets.numpy())

        # Stack into final training arrays
        # X_train = np.concatenate(all_features, axis=0)
        # y_train = np.concatenate(all_targets, axis=0)

        # Train SVC with probability support
        # xgb_model = SVC(probability=True)
        # xgb_model.fit(X_train, y_train)

        for batch_i, (imgs, feature, targets) in enumerate(val_loader):
            print(batch_i)
            imgs, targets = imgs.to(device), targets.to(device)  # Move to CPU
            outputs, features = model(imgs)  # Get predictions
            loss = CE_criterion(outputs, targets)
            test_loss += loss.item()

            # Predictions
            predicts = torch.argmax(outputs, dim=1)  # More efficient
            fsfnet_probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            correct_predictions += torch.eq(predicts, targets).sum().item()

            # Store labels for F1 score calculation
            pre_labels.extend(predicts.tolist())
            gt_labels.extend(targets.tolist())

            # XGBoost predictions
            features_np = feature.numpy()
            features_scaled = scaler.transform(features_np)
            xgb_preds = xgb_model.predict(features_scaled)
            xgb_probs = xgb_model.predict_proba(features_scaled)

            xgb_pre_labels.extend(xgb_preds.tolist())
            # xgb_gt_labels.extend(targets.tolist())

            # Average predictions
            # avg_probs = (fsfnet_probs + xgb_probs) / 2.0
            avg_probs = (1-w) * fsfnet_probs + w * xgb_probs
            combined_preds = np.argmax(avg_probs, axis=1)

            combined_pre_labels.extend(combined_preds.tolist())
            # combined_gt_labels.extend(targets.tolist())

            fsfnet_prob_list.append(fsfnet_probs)         # From torch.softmax(outputs, dim=1)
            xgb_prob_list.append(xgb_probs)               # From xgb_model.predict_proba(...)
            meta_targets.extend(targets.tolist())         # Ground truth labels


        # Compute final metrics for deep model
        to_print.append(f"{feature_column}")
        test_loss /= total_samples
        test_acc = correct_predictions / total_samples
        f1 = f1_score(gt_labels, pre_labels, average="macro")
        total_score = 0.67 * f1 + 0.33 * test_acc

        to_print.append(f"\n[FSFNet Results]")
        to_print.append(classification_report(gt_labels, pre_labels, digits=4))
        to_print.append(f"Accuracy: {test_acc:.4f}, Loss: {test_loss:.3f}, F1: {f1:.4f}, Score: {total_score:.4f}")

        # Evaluate SVC model
        xgb_acc = accuracy_score(gt_labels, xgb_pre_labels)
        xgb_f1 = f1_score(gt_labels, xgb_pre_labels, average="macro")
        xgb_total_score = 0.67 * xgb_f1 + 0.33 * xgb_acc

        to_print.append(f"\n[XGBoost Results]")
        to_print.append(classification_report(gt_labels, xgb_pre_labels, digits=4))
        to_print.append(f"Accuracy: {xgb_acc:.4f}, Macro F1: {xgb_f1:.4f}, Score: {xgb_total_score:.4f}")

        # Evaluate combined model
        combined_acc = accuracy_score(gt_labels, combined_pre_labels)
        combined_f1 = f1_score(gt_labels, combined_pre_labels, average="macro")
        combined_total_score = 0.67 * combined_f1 + 0.33 * combined_acc

        to_print.append(f"\n[Combined FSFNet + XGBoost]")
        to_print.append(f"\n {w} * XGBoost + {1-w} * FSFNet")
        to_print.append(classification_report(gt_labels, combined_pre_labels, digits=4))
        to_print.append(f"Accuracy: {combined_acc:.4f}, Macro F1: {combined_f1:.4f}, Score: {combined_total_score:.4f}")

        # Combine all batches
        fsfnet_probs_all = np.concatenate(fsfnet_prob_list, axis=0)
        xgb_probs_all = np.concatenate(xgb_prob_list, axis=0)
        meta_features = np.concatenate([fsfnet_probs_all, xgb_probs_all], axis=1)

        # TODO: Retrain logistic model
        # meta_model = joblib.load('logistic_model_last_{w}.pkl')
        # meta_model = joblib.load('logistic_model_xgboost_best_{w}.pkl')
        # meta_model = joblib.load('logistic_model_svc_best_{w}.pkl')

        # meta_preds = meta_model.predict(meta_features)

        # stacked_acc = accuracy_score(meta_targets, meta_preds)
        # stacked_f1 = f1_score(meta_targets, meta_preds, average="macro")
        # stacked_total_score = 0.67 * stacked_f1 + 0.33 * stacked_acc

        # to_print.append(f"\n[Stacked FSFNet + ML]")
        # to_print.append(classification_report(meta_targets, meta_preds, digits=4))
        # to_print.append(f"Accuracy: {stacked_acc:.4f}, Macro F1: {stacked_f1:.4f}, Score: {stacked_total_score:.4f}")

        ###

        # from sklearn.model_selection import train_test_split

        # X_train, X_test, y_train, y_test = train_test_split(
        #     meta_features, meta_targets, test_size=0.3, random_state=42)

        # meta_model = LogisticRegression(max_iter=1000)
        # meta_model.fit(X_train, y_train)
        # joblib.dump(meta_model, './logistic_model_svc_best_{w}.pkl')

        # meta_preds = meta_model.predict(X_test)

        # stacked_acc = accuracy_score(y_test, meta_preds)
        # stacked_f1 = f1_score(y_test, meta_preds, average="macro")
        # stacked_total_score = 0.67 * stacked_f1 + 0.33 * stacked_acc

        # to_print.append(f"\n[Stacked FSFNet + XGBoost]")
        # to_print.append(classification_report(y_test, meta_preds, digits=4))
        # to_print.append(f"Accuracy: {stacked_acc:.4f}, Macro F1: {stacked_f1:.4f}, Score: {stacked_total_score:.4f}")
    
        for line in to_print:
            print(line)
            # with open("./best_finetuned_ensemble.log", "a") as f:
            #     f.write(line + "\n")

        data = np.column_stack((gt_labels, combined_pre_labels, pre_labels, xgb_pre_labels))
        np.savetxt("./prediction_labels.csv", data, delimiter =",", header="gt_labels, combined, fsfnet, xgboost")
        print("Saved: ", data)

        def mcnemar_table(y_true, y_pred_A, y_pred_B):
            # Ensure numpy arrays
            y_true = np.array(y_true)
            y_pred_A = np.array(y_pred_A)
            y_pred_B = np.array(y_pred_B)
            
            # Correctness arrays (True = correct, False = wrong)
            correct_A = (y_pred_A == y_true)
            correct_B = (y_pred_B == y_true)
            
            # Counts for the McNemar table
            n11 = np.sum(correct_A & correct_B)   # both correct
            n10 = np.sum(correct_A & ~correct_B)  # A correct, B wrong
            n01 = np.sum(~correct_A & correct_B)  # A wrong, B correct
            n00 = np.sum(~correct_A & ~correct_B) # both wrong
            
            table = np.array([[n11, n10],
                            [n01, n00]])
            return table

        from statsmodels.stats.contingency_tables import mcnemar

        table_fsfnet = mcnemar_table(gt_labels, combined_pre_labels, pre_labels)
        table_xgboost = mcnemar_table(gt_labels, combined_pre_labels, xgb_pre_labels)

        print("table_fsfnet", table_fsfnet)
        print("table_xgboost", table_xgboost)

        result = mcnemar(table_fsfnet, exact=False, correction=True)
        print("Statistic FSFNet:", result.statistic, "p-value:", result.pvalue)

        result = mcnemar(table_xgboost, exact=False, correction=True)
        print("Statistic XGBoost:", result.statistic, "p-value:", result.pvalue)

