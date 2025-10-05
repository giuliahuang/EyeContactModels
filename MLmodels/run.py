import numpy as np, pandas as pd
import os
import subprocess
import cv2
import pdb
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score, f1_score
from sklearn import preprocessing
import joblib
from tqdm import tqdm
import xgboost as xgb


# Project paths
ABS_PATH = "./"
DATASET_PATH = "../Dataset/"
INTERMEDIATE_PATH = os.path.join(ABS_PATH, "intermediate")

def check_dir(directory):
    """
    Ensure that a directory exists; create it if missing.

    Parameters
    ----------
    directory : str
        Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    else:
        print(f"Directory already exists: {directory}")


# Initialize required directories
check_dir(INTERMEDIATE_PATH)

def train(test_on_val: bool = True, featuresets: list = ["gaze_of", "pose_of", "speaker_info"], skip_training: bool = False, use_gpu: bool = False, data_path: str = None):

    """
    Train and evaluate an ML model (XGBoost or SVM) for eye contact classification.

    Parameters
    ----------
    test_on_val : bool, default=True
        If True, use the validation set for testing. 
        If False, train on train+val (final submission setup).
    featuresets : list, default=["gaze_of", "pose_of", "speaker_info"]
        List of feature groups to use for training.
    skip_training : bool, default=False
        If True, skip training and load pretrained models.
    use_gpu : bool, default=False
        Whether to use GPU-accelerated XGBoost.
    data_path : str
        Path to CSV file containing dataset with features and labels.

    Returns
    -------
    output_train_log : list of str
        Log of training and evaluation results.
    """
    
    output_train_log = []

    featuresets = sorted(featuresets)
    
    current_model = 'XGBoost' if use_gpu else 'SVC'
    
    model_id = current_model + '-testonval-' +str(test_on_val) + '-'.join(['_feats']+featuresets)
    
    # Define output paths
    model_out_path = os.path.join(INTERMEDIATE_PATH, "models", model_id)
    pred_out_path = os.path.join(INTERMEDIATE_PATH, "preds", model_id)
    for path in [model_out_path, pred_out_path]:
        os.makedirs(path, exist_ok=True)

    # Validate input data
    if not data_path:
        print("No data path provided.")
        return

    data = pd.read_csv(data_path)
    feature_cols = get_feature_cols(featuresets)
    label_col = "ec_relative"

    if test_on_val:
        train_indicator = data['data_subset']=='train'
    else:
        train_indicator = np.ones(data.shape[0]).astype(bool)

    # Prepare training data
    df_train = data.loc[train_indicator, feature_cols + [label_col]]
    print(f"# training samples (with NaN): {df_train.shape[0]}")
    df_train = df_train.dropna()
    print(f"# training samples (without NaN): {df_train.shape[0]}")

    x_train = df_train[feature_cols].values
    y_train = df_train[label_col].values

    # Training
    if not skip_training:
        # Normalize features
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)

        # Balance classes
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)


        if use_gpu:
            print("Training with GPU-accelerated XGBoost.")
            estimator = xgb.XGBClassifier(
                tree_method="hist",
                device="cuda",
                eval_metric="logloss",
            )
            param_grid = {
                "max_depth": [3, 4, 5],
                "learning_rate": [0.01, 0.1, 0.2],
                "n_estimators": [100, 200],
                "gamma": [0, 1, 5],
            }
        else:
            print("Training with SVM.")
            estimator = SVC(kernel="rbf", probability=True)
            param_grid = {
                "C": 2.0 ** np.arange(-7, 8),
                "gamma": 2.0 ** np.arange(-7, 8),
            }

        # Hyperparameter tuning
        gridSearch = GridSearchCV(estimator,param_grid)
        gridSearch.fit(x_train, y_train, **{"sample_weight": sample_weights})
        best_estimator = gridSearch.best_estimator_

        # Save models
        joblib.dump(best_estimator,os.path.join(model_out_path,'best_estimator_ec.joblib'))
        joblib.dump(scaler,os.path.join(model_out_path,'scaler.joblib'))
    else:
        print("Skipping training: loading pretrained model.")
        best_estimator = joblib.load(os.path.join(model_out_path,'best_estimator_ec.joblib'))
        scaler = joblib.load(os.path.join(model_out_path,'scaler.joblib'))

    # performance on train set
    if x_train.any():
        preds_train = best_estimator.predict(x_train)
        report = classification_report(y_train,preds_train)
        print("Training performance:\n", report)
        output_train_log.append(f"Training performance:\n{report}")

    
    # Evaluate on validation set (if applicable)
    if test_on_val:
        df_val = data.loc[~train_indicator, :]
        x_val = df_val[feature_cols].fillna(df_val[feature_cols].mean()).values
        x_val = scaler.transform(x_val)
        y_val = np.nan_to_num(df_val[label_col], nan=0.0)

        preds_val = best_estimator.predict(x_val)
        report_val = classification_report(y_val, preds_val)
        print("Validation performance:\n", report_val)
        output_train_log.append(f"Validation performance:\n{report_val}")

        # Compute summary metrics
        val_acc = accuracy_score(y_val, preds_val)
        val_f1 = f1_score(y_val, preds_val, average="macro")
        total_score = 0.67 * val_f1 + 0.33 * val_acc

        summary_lines = [
            f"Features: {featuresets}",
            f"Validation Accuracy: {val_acc:.4f}",
            f"Validation Macro F1: {val_f1:.4f}",
            f"Combined Score: {total_score:.4f}\n",
        ]

        for line in summary_lines:
            print(line)

    return



cols4featuresets = {
    'pose_of':['pose_Tx','pose_Ty','pose_Tz','pose_Rx','pose_Ry','pose_Rz'],
    'gaze_of':['gaze_0_x','gaze_0_y','gaze_0_z','gaze_1_x','gaze_1_y','gaze_1_z'],
    'poseT_of':['pose_Tx','pose_Ty','pose_Tz'],
    'poseR_of':['pose_Rx','pose_Ry','pose_Rz'],
    'speaker_info':['speaker_0','speaker_1','speaker_2','speaker_3','speaker_4'],
    'is_speaker_info': ['is_speaker'],
}

def get_feature_cols(featuresets):
    """
    Expand feature set identifiers into actual feature column names.

    Parameters
    ----------
    featuresets : list of str
        Keys corresponding to feature groups (e.g., "gaze_of", "pose_of").

    Returns
    -------
    feature_cols : list of str
        Flattened list of feature column names.
    """
    featuresets = sorted(featuresets)
    feature_cols = []
    for featureset in featuresets:
        feature_cols.extend(cols4featuresets[featureset])
    return feature_cols

if __name__ == "__main__":
    """
    Script to train or evaluate models on different feature combinations.
    """

    # File paths
    test_file_path = os.path.join(DATASET_PATH, "test.csv")
    train_file_path = os.path.join(DATASET_PATH, "train_val.csv")

    # Toggle between training and evaluation
    training_mode = True
    gpu = False

    # Validate dataset availability
    if (training_mode and not os.path.exists(train_file_path)) or (
        not training_mode and not os.path.exists(test_file_path)
    ):
        raise FileNotFoundError(
            f"Required dataset not found in {DATASET_PATH}. "
            "Please extract features first and save them to this directory."
        )

    # List of models and corresponding feature sets
    features_set_list = [
        ["gaze_of"],
        ["gaze_of", "is_speaker_info"],
        ["gaze_of", "is_speaker_info", "pose_of"],
        ["is_speaker_info", "pose_of"],
        ["gaze_of", "pose_of"],
        ["gaze_of", "is_speaker_info", "pose_of", "speaker_info"],
        ["gaze_of", "is_speaker_info", "speaker_info"],
        ["gaze_of", "pose_of", "speaker_info"],
        ["gaze_of", "speaker_info"],
        ["is_speaker_info", "pose_of", "speaker_info"],
        ["pose_of", "speaker_info"],
        ["pose_of"]
    ]

    # Train or evaluate models
    for featuresets in features_set_list:
        if training_mode:
            train(
                test_on_val=True,
                featuresets=featuresets,
                use_gpu=gpu,
                skip_training=False,
                data_path=train_file_path,
            )
        else:
            train(
                test_on_val=True,
                featuresets=featuresets,
                use_gpu=gpu,
                skip_training=True,
                data_path=test_file_path,
            )

    print("All experiments completed.")