import os
import subprocess

# Fixed model name to be used for all tests
global_model = 'model.bin'


def run_expert_test(model_name: str, test_data_path: str) -> str:
    
    command = [
        "python", "/home/eddie.veronese/Eddie/LineVul/linevul/linevul_main.py",
        "--model_name", model_name,
        "--output_dir", "/home/eddie.veronese/Eddie/LineVul/linevul/saved_models",
        "--model_type", "roberta",
        "--tokenizer_name=microsoft/codebert-base",
        "--model_name_or_path=microsoft/codebert-base",
        "--do_test",
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
        print(f"[ERROR] Model {model_name} (return code {result.returncode})")
    return result.stdout.strip()


def main():
    base_dir = "experts_datasets"
    # List of CWE suffixes to run tests for
    cwe_suffixes = ['284', '664', '682', '691', '693', '703', '707', '710']

    for suffix in cwe_suffixes:
        test_csv = os.path.join(base_dir, f"processed_test_CWE-{suffix}.csv")
        if not os.path.isfile(test_csv):
            print(f"[SKIP] File not found for CWE-{suffix}: {test_csv}")
            continue

        print(f"\n=== Running test for CWE-{suffix} with fixed model {global_model} ===")
        output = run_expert_test(global_model, test_csv)
        if output:
            print(f"[OK] Results CWE-{suffix}:\n{output}")
        else:
            print(f"[WARN] No output captured for CWE-{suffix}.")


if __name__ == "__main__":
    main()
