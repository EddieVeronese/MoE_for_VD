## How to replicate 

### About the Environment Setup
First of all, create a python environment with the following commands:
```
python3 -m venv venv
source venv/bin/activate
```

Then install the python dependencies via the following command:
```
pip install -r requirements.txt
```

### About the Datasets
The datasets have a variable number of columns based on the language, but the three that are needed for training and using the models are
1. processed_func (str): The original function
2. target (int): The function-level label that determines whether a function is vulnerable or not
3. vul_func_with_fix (str): The fixed function with added in deleted lines labeled.
The three datasets used for the experiments are: BigVul (C/C++), ProjectKB (Java) and CVEFixes (Python).
 
    
#### How to test a model
  To test a model, download the model and corresponding datasets in the right folder, for example
  ```
  cd linevul
  cd saved_models
  cd checkpoint-best-f1
  gdown https://drive.google.com/uc?id=1cOwkzjsNYelSstx3CcMxVIuuHCislkEC
  cd ../../..
  cd data
  cd datasets
  gdown https://drive.google.com/uc?id=1J-XS_ftJKtu3qLYwSGm5TtQVy2d74bKW
  cd ../../..
  ```
  
  To reproduce the results run the following script which will produce the results at both function and line level:
  ```
  cd linevul
  bash run_test.sh
  ```
    
  This command executes:
  ```
  python linevul_main.py \
    --model_name=model_name.bin \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_test \
    --do_local_explanation \
    --top_k_constant=10 \
    --reasoning_method=all \
    --train_data_file=../data/datasets/train.csv \
    --eval_data_file=../data/datasets/val.csv \
    --test_data_file=../data/datasets/test.csv \
    --block_size 512 \
    --eval_batch_size 512
  ```

  You can change the model used with the --model_name flag, and the datasets used with the --train_data_file, --eval_data_file, and --test_data_file flags. You can also change the system used from codbert to uniixcoder by changing the --tokenizer_name and --model_name_or_path flags in unixcoder-base.


    
#### How to train a model
  
  To train a model, download the datasets, for example
  ```
  cd data
  cd datasets
  gdown https://drive.google.com/uc?id=1QpsbDZwhrV2HhpddOLOUDQu1MDCRGLuc
  gdown https://drive.google.com/uc?id=1GTAT2AepayFAZJXcwsasao-aXA-148hh
  cd ../../..
  ```
  
  To reproduce the results run the following script which will produce the model file inside the folder linevul/saved_models:
  ```
  cd linevul
  bash run_train.sh
  ```
    
  This command executes:
  ```
  python linevul/linevul_main.py \
    --output_dir=./linevul/saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --do_test \
    --train_data_file=./data/datasets/j/experts_datasets/j_processed_train.csv \
    --eval_data_file=./data/datasets/j/experts_datasets/j_processed_vel.csv \
    --test_data_file=./data/datasets/j/experts_datasets/j_processed_test.csv \
    --epochs 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
  ```
    

##### Acknowledgment:
###### We thank [LineVul](https://github.com/awsm-research/LineVul) for providing the source code of their project, which has served as a foundation for the current research project.
