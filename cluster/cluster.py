import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch

# Step 1: Load the dataset
df = pd.read_csv("datasets/j_processed_train.csv")
functions = df["processed_func"].dropna().tolist()

# Step 2: Load the CodeBERT tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.eval()

# Step 2b: Function to get CLS token embedding
def get_embedding(code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[0][0]  # [CLS] token
        return embedding.numpy()

# Step 2c: Compute embeddings for all functions (with progress bar)
embeddings = []
for func in tqdm(functions, desc="Generating embeddings"):
    embeddings.append(get_embedding(func))

# Step 3: K-Means clustering
k = 8
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(embeddings)

# Step 4: Save the results
df = df.iloc[:len(labels)]
df["cluster"] = labels
df.to_csv("clustered.csv", index=False)

# Step 5: Display 3 examples per cluster
for i in range(8):
    print(f"\n Cluster {i}")
    cluster_funcs = df[df["cluster"] == i]["processed_func"]
    print(cluster_funcs.sample(min(3, len(cluster_funcs))).values)
