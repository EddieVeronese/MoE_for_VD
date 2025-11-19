import os
import subprocess
import torch
import pandas as pd
from autogluon.multimodal import MultiModalPredictor


def load_router_model():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.current_device()}")
    router_model_path = "models/output_codebert-base_seed45/focal_ep40_bs512_eval_f1_macro_gamma1"
    predictor = MultiModalPredictor.load(router_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model will be loaded on: {device}")
    return predictor


def run_expert_test(expert_class, test_data):
    model_name = expert_models[expert_class]
    command = [
        "python", "../LineVul/linevul/linevul_main.py",
        "--model_name", model_name,
        "--output_dir", "../LineVul/linevul/saved_models",
        "--model_type", "roberta",
        "--tokenizer_name", "microsoft/codebert-base",
        "--model_name_or_path", "microsoft/codebert-base",
        "--do_test",
        "--do_local_explanation",
        "--top_k_constant=10",
        "--save_line_scores",
        "--reasoning_method=all",
        "--train_data_file", "../data/big-vul_dataset/train.csv",
        "--eval_data_file", "../data/big-vul_dataset/val.csv",
        "--test_data_file", test_data,  
        "--block_size", "512",
        "--eval_batch_size", "512"
    ]
   
    result = subprocess.run(command, capture_output=True, text=True)
    if result.stderr:
        print(f"Error for expert {expert_class}:\n{result.stderr}")
    return result.stdout


def classify_with_expert(test_df, expert_class):
    expert_data = test_df[test_df['predicted_class'] == expert_class]
    if expert_data.empty:
        print(f"No data for expert {expert_class}. Skipping...")
        return None
    
    expert_data_path = f"./temp_{expert_class}_test.csv"
    expert_data.to_csv(expert_data_path, index=False)
    print(f"Running test for expert {expert_class}...")
    output = run_expert_test(expert_class, expert_data_path)
    
    
    os.remove(expert_data_path)
    return output


def load_dataset(test_data_path):
    test_df = pd.read_csv(test_data_path)
    print("First few rows of the dataset (showing 'class_CWE'):")
    print(test_df[['class_CWE']].head())
    print("\nUnique values in 'class_CWE' before prediction:")
    print(test_df['class_CWE'].unique())
    return test_df


expert_models = {
    'CWE-284': 'model_bert_284.bin',
    'CWE-664': 'model_bert_664.bin',
    'CWE-682': 'model_bert_682.bin',
    'CWE-691': 'model_bert_691.bin',
    'CWE-693': 'model_bert_693.bin',
    'CWE-703': 'model_bert_703.bin',
    'CWE-707': 'model_bert_707.bin',
    'CWE-710': 'model_bert_710.bin',
}


def main():
    
    predictor = load_router_model()

    
    test_data_path = "datasets/processed_test.csv"
    test_df = load_dataset(test_data_path)

    
    print("\nPredictions from router:")
    predictions = predictor.predict(test_df)
    test_df['predicted_class'] = predictions
    print("\nUnique values in 'predicted_class' after prediction:")
    print(test_df['predicted_class'].unique())

    
    test_df.to_csv("predicted_classes.csv", index=False)
    
    
    for expert_class in expert_models.keys():
        print(f"\nClassifying vulnerabilities for {expert_class}...")
        output = classify_with_expert(test_df, expert_class)
        
        
        if output:
            print(f"Results for expert {expert_class}:")
            print(output)
        else:
            print(f"No output for expert {expert_class}.")

if __name__ == "__main__":
    main()
