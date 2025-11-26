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

### Necessary Installations

To get started you need to install the correct version of Joern and perform the environment preparation, use the following commands:

```
chmod +x ./run.sh
./run.sh
./zrun/getjoern.sh
```

### Train

To train a model, you need to download the appropriate dataset into the ```\sourcescripts\storage\external\``` folder, for example with the following commands:

```
cd sourcescripts
cd storage
cd external
gdown https://drive.google.com/uc?id=1ld9rOOvArisajTrC6UARD0mRnyPVfkqZ
cd ../../..
```
And then you need to change the dataset in the script by going to the file ```sourcescripts/utils/preprocessdata.py``` at line 140. It is also necessary to change the name of some columns of the reference dataset by running the script ```change_name.py```, with:

```
python sourcescripts/storage/external/change_name.py 
```

You can then start the train with the command:

```
bash zrun/Process_train_test.sh
```

The results will be stored in ```storage/outputs/```. Once the train is completed you need to delete the workspace folder that was created in the root directory to run further trains.

### Test 
To replicate the test of the trained model you simply have to create a folder inside ```storage/archive```, for example ```model_1``` and put inside it the folders resulting from the train, that are ```checkpoints, codebert_method_level, Dataset, Dataset_Vvuldet_codebert_pdg+raw, minimal_datasets and processed```

Then you can then start the test with the command and the results will appear inside an ```output``` folder, for example:

```
cd sourcescripts
cd storage
cd archive
gdown https://drive.google.com/uc?id=1OUV6ZW6Qh1gQ_UptWQtaOMD6JYI5OAfx
unzip j_284.zip
cd ../../..
bash zrun/process_test.sh
```

Make sure you have entered the name of the folder you created above into the process_test.sh script parameters. You can also run multiple tests in the same way with the script ```./zrun/multiple_test.sh```.


### Using C/C++ or Python Datasets with the GNN Model

By default, running the scripts will use Java code for graph construction and model training. If you want to use a **C/C++ dataset** or **Python dataset**, follow these steps:

#### For C/C++ datasets
1. Open the project in VSCode.  
2. Search for `.java"` and replace it with `.c"`. You should get 7 occurrences to change in 5 files, that are `getgraphdata.py`, `allcwefeaturemain.py.py`, `mainfeaturemain.py`, `nodeedgesdata.py` and `preprocessdata.py`
3. Navigate to `sourcescripts/storage/external/` and open `get_func_graph.scala`.  
4. In line 2, change:  
   ```
   importCode.java(filename)
   ```
   to 
   ```
   importCode.c(filename)
   ```
#### For Python datasets
Perform the same replacement in VSCode, changing .java" to .py".
In get_func_graph.scala, change line 2 from:
```
importCode.java(filename)
```
to
```
importCode.python(filename)
```

##### Acknowledgment:
###### We thank [LineVD](https://github.com/davidhin/linevd) for providing the source code of their project, which has served as a foundation for the current research project.
