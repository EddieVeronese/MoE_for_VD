import os
import subprocess

# Mapping between CWE classes and expert model filenames
expert_models = {
    '1': 'j_model_rand_1.bin',
    '2': 'j_model_rand_2.bin',
    '3': 'j_model_rand_3.bin',
    '4': 'j_model_rand_4.bin',
    '5': 'j_model_rand_5.bin',
    '6': 'j_model_rand_6.bin',
    '7': 'j_model_rand_7.bin',
    '8': 'j_model_rand_8.bin',
}

def run_expert_test(model_name: str, test_data_path: str) -> str:
    """
    Runs the test for a single expert using its model and corresponding CSV dataset.
    Returns the complete output (stdout + stderr), even if no errors occur.
    """
    command = [
        "python", "../LineVul/linevul/linevul_main.py",
        "--model_name", model_name,
        "--output_dir", "../LineVul/linevul/saved_models",
        "--model_type", "roberta",
        "--tokenizer_name", "microsoft/codebert-base",
        "--model_name_or_path", "microsoft/codebert-base",
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
    # Print the error code if the process fails
    if result.returncode != 0:
        print(f"[ERROR] Expert {model_name} failed (return code {result.returncode})")
    return result.stdout.strip()


def main():
    base_dir = "datasets"
    for suffix, model_name in expert_models.items():
        test_csv = os.path.join(base_dir, f"j_equal_test_{suffix}.csv")
        if not os.path.isfile(test_csv):
            print(f"[SKIP] File not found for CWE-{suffix}: {test_csv}")
            continue

        print(f"\n=== Running test for CWE-{suffix} with model {model_name} ===")
        output = run_expert_test(model_name, test_csv)
        if output:
            print(f"[OK] Results for CWE-{suffix}:\n{output}")
        else:
            print(f"[WARN] No output captured for CWE-{suffix}.")

if __name__ == "__main__":
    main()
