import os
import subprocess

# Mapping between CWE classes and expert model filenames
expert_models = {
    '284': 'j_model_284.bin',
    '435': 'j_model_435.bin',
    '664': 'j_model_664.bin',
    '693': 'j_model_693.bin',
    '703': 'j_model_703.bin',
    '707': 'j_model_707.bin',
}

def run_expert_test(model_name: str, test_data_path: str) -> str:
    """
    Run the test for a single expert model using its associated CSV dataset.
    Returns all command output (stdout + stderr), even if there are no errors.
    """
    command = [
        "python", "../LineVul/linevul/linevul_main.py",
        "--model_name", model_name,
        "--output_dir", "../LineVul/linevul/saved_models",
        "--model_type", "roberta",
        "--tokenizer_name", "microsoft/codebert-base",
        "--model_name_or_path", "microsoft/codebert-base",
        "--do_test",
        "--do_local_explanation",
        "--top_k_constant=5",
        "--reasoning_method=attention",
        "--train_data_file", "../data/big-vul_dataset/train.csv",
        "--eval_data_file", "../data/big-vul_dataset/val.csv",
        "--test_data_file", test_data_path,
        "--block_size", "512",
        "--eval_batch_size", "512"
    ]
    # Redirect stderr to stdout
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    # Print error code if the process failed
    if result.returncode != 0:
        print(f"[ERROR] Expert {model_name} (return code {result.returncode})")
    return result.stdout.strip()


def main():
    base_dir = "datasets"
    for suffix, model_name in expert_models.items():
        test_csv = os.path.join(base_dir, f"j_processed_test_CWE-{suffix}.csv")
        if not os.path.isfile(test_csv):
            print(f"[SKIP] File not found for CWE-{suffix}: {test_csv}")
            continue

        print(f"\n=== Running test for CWE-{suffix} with model {model_name} ===")
        output = run_expert_test(model_name, test_csv)
        if output:
            print(f"[OK] Results CWE-{suffix}:\n{output}")
        else:
            print(f"[WARN] No output captured for CWE-{suffix}.")

if __name__ == "__main__":
    main()
