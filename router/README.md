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

### Train

To train a router model based on CWE, you need to download the appropriate dataset into the ```\datasets``` folder, for example with the following commands:

```
cd datasets
gdown https://drive.google.com/uc?id=1QpsbDZwhrV2HhpddOLOUDQu1MDCRGLuc
gdown https://drive.google.com/uc?id=1GTAT2AepayFAZJXcwsasao-aXA-148hh
gdown https://drive.google.com/uc?id=1koqk0H2vmXwSaErq921R6SMT5uGHj4-z
cd ../
```
And then you need to change the dataset used in the script by going to the file ```train_router_CWE.py``` at line 179, 185 and 191.

You can then start the train with the command:

```
python train_router_CWE.py
```

The results will be stored in ```models/```

### Variations
To train the model on all the vulnerabilities present in the dataset and not just on the vulnerable ones, it is necessary to comment out the lines 203, 207 and 211, that is:
```python
train_pd = train_pd[train_pd['target'] == 1]
val_pd = val_pd[val_pd['target'] == 1]
test_pd = test_pd[test_pd['target'] == 1]
```

If, however, the model you want to train is based on a dataset created through clustering, you need to repeat all the previous steps, but with the script ```train_router_cluster.py``` and finally run:

```
cd datasets
gdown https://drive.google.com/uc?id=1rHIQxs-_sY9nv8SkVn952ixViiMTuzxY
gdown https://drive.google.com/uc?id=1VxbkI4wPYhdTnmsNWADPds49WfOQY5-7
gdown https://drive.google.com/uc?id=1iuuFjc2-kd30NU4r9yKTs4CdzuQx1zp0
cd ../
python train_router_cluster.py
```

### Test
Inside the main folder there are various files for testing both the router and the overall architecture

#### Router Test
To test CWE based router you must first have a model in the folder ```models``` and a test csv file in the folder ```datasets```, as shown by the commands:

```
cd datasets
gdown https://drive.google.com/uc?id=1koqk0H2vmXwSaErq921R6SMT5uGHj4-z
cd ../
cd models
gdown --folder https://drive.google.com/drive/folders/113bFITEqbz9NVR9s9isyAy8El_HdA8oc
cd ../
```

Then you need to change the name of the model used and the name of the test dataset used, respectively on lines 9 and 81 of the test file ```test_router_CWE.py```, i.e. the parameters:  ```router_model_path``` and ```test_csv```.

Finally run the script with the command:

```
python test_router_CWE.py
```

If, however, the model you want to test is based on a dataset created through clustering, you need to repeat all the previous steps, but with the script ```test_router_cluster.py``` and finally run:

```
cd datasets
gdown https://drive.google.com/uc?id=1iuuFjc2-kd30NU4r9yKTs4CdzuQx1zp0
cd ../
cd models
gdown --folder https://drive.google.com/drive/folders/1AkFpXztA5XE7fLoE_7EMtnjzyhTNsuDE
cd ../
python test_router_cluster.py
```

#### Architecture Test
The overall architecture tests start from this folder but pull the LineVul templates and files directly from the LineVul folder.
There are two main types of tests: those that consider the correct sorting of vulnerabilities, as if the router never made mistakes, and those that instead consider the router and its errors. Let's start with an explanation of the test with perfect sorting.

##### Test with Right Predictions
To run the test, you need to have the desired models in the ```LineVul/linevul/saved_models``` folder and set them in the dictionary contained in the ```expert_models``` parameter on line 5 of the ```test_bert_func_only_right``` file. Then you need to have the test files corresponding to the various models in the ```router/datasets``` folder and set them in the ```test_csv``` parameter on line 51 of the test file. It is important that there is a correspondence between the suffix assigned in the dictionary to each model and the identifying name of the various test datasets, as shown for example by these commands:
```
cd ..
cd LineVul
cd linevul
cd saved_models
cd checkpoint-best-f1
gdown https://drive.google.com/uc?id=1cOwkzjsNYelSstx3CcMxVIuuHCislkEC
gdown https://drive.google.com/uc?id=1_9E5nmLv1ZUjxOM81c-4gKFIcWJihglf
gdown https://drive.google.com/uc?id=1IC0R_k_czExOZtyqXhRPpGjKKdR_uFu2
gdown https://drive.google.com/uc?id=1ELY3psqKwZKcafYI2W68VSll3rV1IfSn
gdown https://drive.google.com/uc?id=1f57S9iQsooZgujS1Ry3Xysb7OxMpfeU2
gdown https://drive.google.com/uc?id=1fIty7YypU8X1DZtPt5FjH4sJ1LpgNKBL
cd ../../../../
cd router
cd dataset
gdown https://drive.google.com/uc?id=1J-XS_ftJKtu3qLYwSGm5TtQVy2d74bKW
gdown https://drive.google.com/uc?id=1OJaCi9EdHcG2qOaktxpqEwhePiBJgpBS
gdown https://drive.google.com/uc?id=11lP6mF2oBixWson4sUnZCbVYjfqhOwE5
gdown https://drive.google.com/uc?id=150NGePVyR5mcT4vlflCTHkzKUJxyAsZv
gdown https://drive.google.com/uc?id=1pNzwRnYaQUs1BrZ2cKBPygUPAV1jbuHT
gdown https://drive.google.com/uc?id=1FbdRwp-WIMwxXo4dg6t3NF25eJeYLmT8
cd ..
```

You can then start the test at function level with the command:

```
python test_bert_func_only_right.py
```

To instead run the test at the statement level, you need to repeat the steps above inside the ```test_bert_state_only_right.py``` file and then run it with:

```
python test_bert_state_only_right.py
```

In the latter you can change some parameters like ```top_k_constant``` and ```reasoning_method```, as described in the LineVul instructions

##### Test with Router
To run the test, you need to have the desired models in the ```LineVul/linevul/saved_models``` folder and set them in the dictionary contained in the ```expert_models``` parameter on line 67 of the ```test_bert_func``` file. Then you need to have the test dataset in the ```router/datasets``` folder and set them in the ```test_data_path``` parameter on line 84 of the test file. Additionally, you also need to enter the path to the model used by the router in the ```router_model_path``` parameter on line 11. It is important that there is a correspondence between the suffix assigned in the dictionary to each model and the label assigned by the router; an example is seen in the following commands:
```
cd ..
cd LineVul
cd linevul
cd saved_models
cd checkpoint-best-f1
gdown https://drive.google.com/uc?id=1cOwkzjsNYelSstx3CcMxVIuuHCislkEC
gdown https://drive.google.com/uc?id=1_9E5nmLv1ZUjxOM81c-4gKFIcWJihglf
gdown https://drive.google.com/uc?id=1IC0R_k_czExOZtyqXhRPpGjKKdR_uFu2
gdown https://drive.google.com/uc?id=1ELY3psqKwZKcafYI2W68VSll3rV1IfSn
gdown https://drive.google.com/uc?id=1f57S9iQsooZgujS1Ry3Xysb7OxMpfeU2
gdown https://drive.google.com/uc?id=1fIty7YypU8X1DZtPt5FjH4sJ1LpgNKBL
cd ../../../../
cd router
cd dataset
gdown https://drive.google.com/uc?id=1koqk0H2vmXwSaErq921R6SMT5uGHj4-z
cd ..
cd models 
gdown --folder https://drive.google.com/drive/folders/113bFITEqbz9NVR9s9isyAy8El_HdA8oc
cd ..
```

You can then start the test at function level with the command:

```
python test_bert_func.py
```

To instead run the test at the statement level, you need to repeat the steps above inside the ```test_bert_state.py``` file and then run it with:

```
python test_bert_state.py
```
