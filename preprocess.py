import os
import pandas as pd
from sklearn.datasets import load_iris

os.makedirs("data", exist_ok=True)

# check if dataset.csv exists
if not os.path.exists("data/dataset.csv"):
    print("⚠️ dataset.csv not found — generating sample data (Iris)")
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.to_csv("data/dataset.csv", index=False)

# proceed with preprocessing
data = pd.read_csv("data/dataset.csv")
data.to_csv("data/preprocessed.csv", index=False)
print("✅ Preprocessing completed successfully")
