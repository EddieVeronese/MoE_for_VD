import json
from autogluon.core.utils.loaders import load_pd
from datasets import load_dataset
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
import uuid
import torch
from sklearn.model_selection import train_test_split
import sklearn
import numpy as np
import os
from autogluon.core.metrics import make_scorer
import sys
import argparse
from imblearn.over_sampling import RandomOverSampler


def clean_func(func):
    """Remove empty lines and trim leading/trailing whitespace from a function string."""
    lines = func.split("\n")
    new_lines = []
    for line in lines:
        # trim leading and trailing whitespaces
        line = line.strip()
        # remove empty lines
        if line:
            new_lines.append(line)
    return "\n".join(new_lines)

def trainer(train_pd, val_pd, test_pd, args):
    """
    Train or load a MultiModalPredictor for sequence classification.
    Supports focal loss weighting or random oversampling (ROS).
    """
    weights = []
    for type in train_pd['class_CWE'].value_counts().keys():
        class_data = train_pd[train_pd.class_CWE == type]
        weights.append(1 / (class_data.shape[0] / train_pd.shape[0]))
        print(f"class {type}: num samples {len(class_data)}")
    weights = list(np.array(weights) / np.sum(weights))
    
    print(weights)
    
    selected_model = args.model_name

    epochs = args.epochs
    batch_size = args.batch_size
    gamma = 1
    # Determine model path
    if args.ros:
        model_path = f"output_{selected_model.split('/')[-1]}_seed{args.seed}/ros_ep{epochs}_bs{batch_size}_eval_f1_macro"
    else:
        model_path = f"output_{selected_model.split('/')[-1]}_seed{args.seed}/focal_ep{epochs}_bs{batch_size}_eval_f1_macro_gamma{gamma}"
    
    # Load model if it exists
    if os.path.exists(model_path):
        predictor = MultiModalPredictor.load(model_path)
        
    else:
        # Set hyperparameters
        param_dicts = {
            "model.hf_text.checkpoint_name": selected_model,
            "env.precision": "bf16-mixed",
            "env.per_gpu_batch_size": 32,
            "env.batch_size": batch_size,
            "optimization.max_epochs": epochs
        }
        
        if not args.ros:
            # Use focal loss
            print('use focal loss')
            print(weights)
            param_dicts["optimization.loss_function"] = "focal_loss"
            param_dicts["optimization.focal_loss.alpha"] = weights
            param_dicts["optimization.focal_loss.gamma"] = 1.0
            param_dicts["optimization.focal_loss.reduction"] = "sum"
        else:
            # Apply Random Oversampling
            print('use ros')
            ros = RandomOverSampler(random_state=0)
            train_pd, y_resampled = ros.fit_resample(train_pd, train_pd['class_CWE'])
            train_pd = train_pd.reset_index(drop=True)
            print(train_pd.class_CWE.value_counts().to_dict())
        
        # Initialize predictor
        predictor = MultiModalPredictor(
            label='class_CWE', path=model_path,eval_metric ='f1_macro'
        )
        predictor.fit(
        train_data=train_pd,
        tuning_data=val_pd,
        hyperparameters=param_dicts,
        )

        # Evaluate on test set
        eval_result = predictor.evaluate(test_pd,metrics = ['f1_macro','f1_micro','f1_weighted','accuracy','mcc'])
        print(eval_result)
        with open(f"{model_path}/multicls_eval_result.json", "w") as f:
            json.dump(eval_result, f)

    # Inference on test set
    inference_test = True
    if inference_test:
        test_pd = pd.read_csv(args.test_file)
        test_pd['processed_func'] = test_pd['processed_func'].apply(clean_func)
        test_dataset = test_pd[["processed_func", "class_CWE"]]
        test_result = predictor.predict_proba(test_dataset[["processed_func"]])
        test_result.to_pickle(f"{model_path}/testset_pred_proba.pkl")

    # Inference on training set (optional)
    inference_train = False
    if inference_train:
        train_pd = pd.read_csv(args.train_file)
        train_pd['processed_func'] = train_pd['processed_func'].apply(clean_func)
        train_dataset = train_pd[["processed_func", "class_CWE"]]
        train_result = predictor.predict_proba(train_dataset[["processed_func"]])
        train_result.to_pickle(f"{model_path}/trainset_pred_proba.pkl")

    # Inference on validation set
    inference_val = True
    if inference_val:
        val_pd = pd.read_csv(args.val_file)
        val_pd['processed_func'] = val_pd['processed_func'].apply(clean_func)
        val_dataset = val_pd[["processed_func", "class_CWE"]]
        val_result = predictor.predict_proba(val_dataset[["processed_func"]])
        val_result.to_pickle(f"{model_path}/valset_pred_proba.pkl")


def parse_args():
    """Parse command line arguments for training and inference."""
    parser = argparse.ArgumentParser(
        description="sequence classification task")
    parser.add_argument(
        "--cwe",
        type=str,
        default="binary",
        help="cwe name",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/codebert-base",
        help="model name",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="batch_size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="ag train seed",
    )
    parser.add_argument(
        "--ros",
        type=bool,
        default=False,
        help="whether to do ros or focalloss",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="datasets/p_processed_train_only_vuln.csv",
        help="train file",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="datasets/p_processed_test_only_vuln.csv",
        help="test file",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="datasets/p_processed_val_only_vuln.csv",
        help="val file",
    )
    args = parser.parse_args()

    return args


def main():
    # Load and clean datasets
    train_pd = pd.read_csv(args.train_file)
    train_pd['processed_func'] = train_pd['processed_func'].apply(clean_func)
    train_pd = train_pd[train_pd['target'] == 1]
    
    val_pd = pd.read_csv(args.val_file)
    val_pd['processed_func'] = val_pd['processed_func'].apply(clean_func)
    val_pd = val_pd[val_pd['target'] == 1]
    
    test_pd = pd.read_csv(args.test_file)
    test_pd['processed_func'] = test_pd['processed_func'].apply(clean_func)
    test_pd = test_pd[test_pd['target'] == 1]
    
    def find_cwes(df,level):
        """Return the list of CWE classes in the dataset."""
        cwe_list = list(df[level].value_counts().keys())
        cwe_list = [x for x in cwe_list if str(x) != 'nan']
        return cwe_list
    
    lv1_cwes = find_cwes(train_pd, 'class_CWE')
    
    # Filter datasets to include only known CWEs
    train_lv1 = train_pd[train_pd['class_CWE'].isin(lv1_cwes)]
    val_lv1 = val_pd[val_pd['class_CWE'].isin(lv1_cwes)]
    test_lv1 = test_pd[test_pd['class_CWE'].isin(lv1_cwes)]
    

    train_pd = train_lv1[["processed_func", "class_CWE"]]
    val_pd = val_lv1[["processed_func", "class_CWE"]]
    test_pd = test_lv1[["processed_func", "class_CWE"]]
    
    # Train or load predictor
    trainer(train_pd, val_pd, test_pd, args)
    
    
if __name__ == "__main__":
    args = parse_args()
    main()
