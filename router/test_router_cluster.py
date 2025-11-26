import torch
import pandas as pd
from autogluon.multimodal import MultiModalPredictor

# --- Load the trained router model ---
def load_router_model():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.current_device()}")
    router_model_path = "models/output_codebert-base_seed54/focal_ep40_bs512_eval_f1_macro_gamma1"
    predictor = MultiModalPredictor.load(router_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model will be loaded on: {device}\n")
    return predictor

# --- Evaluate routing accuracy and class-level performance ---
def evaluate_routing(predictor: MultiModalPredictor, test_data_path: str) -> None:
    # Load test data
    df = pd.read_csv(test_data_path)
    if 'new_class_CWE' not in df.columns:
        raise KeyError("Dataset must contain a 'new_class_CWE' column with ground truth labels.")

    # Predict with router
    df['predicted_class'] = predictor.predict(df)

    df['predicted_class'] = df['predicted_class'].astype(str)
    df['new_class_CWE'] = df['new_class_CWE'].astype(str)

    # Compute overall metrics
    total = len(df)
    correct = (df['predicted_class'] == df['new_class_CWE']).sum()
    incorrect = total - correct
    accuracy = correct / total * 100

    # Summary table
    summary_df = pd.DataFrame({
        'Metric': ['Total samples', 'Correct', 'Incorrect', 'Accuracy (%)'],
        'Value': [total, correct, incorrect, f"{accuracy:.2f}"]
    })
    print("\n=== Routing Summary ===")
    print(summary_df.to_string(index=False))

    # Per-class correct counts
    correct_df = df[df['predicted_class'] == df['new_class_CWE']]
    per_class = (
        correct_df
        .groupby('new_class_CWE')
        .size()
        .reset_index(name='correct_count')
        .sort_values('new_class_CWE')
    )
    print("\n=== Correct Predictions per CWE ===")
    print(per_class.to_string(index=False))

    # Detailed misclassification breakdown
    mis = df[df['predicted_class'] != df['new_class_CWE']]
    if mis.empty:
        print("\nNo misclassifications. All samples routed correctly.")
    else:
        counts = (
            mis
            .groupby(['new_class_CWE', 'predicted_class'])
            .size()
            .reset_index(name='count')
            .sort_values(['new_class_CWE', 'count'], ascending=[True, False])
        )
    print("\n=== Misclassification Breakdown ===")
    print(counts.to_string(index=False))

    # --- Full confusion matrix (non-normalized) ---
    conf_matrix = (
        df.groupby(['new_class_CWE', 'predicted_class'])
        .size()
        .unstack(fill_value=0)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    print("\n=== Confusion Matrix (Actual CWE x Predicted CWE) ===")
    print(conf_matrix)


if __name__ == '__main__':
    # Example usage
    test_csv = 'datasets/new_j_processed_test.csv'

    predictor = load_router_model()
    evaluate_routing(predictor, test_csv)