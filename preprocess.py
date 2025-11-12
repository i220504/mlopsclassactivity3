import pandas as pd
from sklearn.datasets import load_iris
import os
import pandas as pd

# ensure data folder exists
os.makedirs("data", exist_ok=True)

# your code
data = pd.read_csv("data/dataset.csv")
data.to_csv("data/preprocessed.csv", index=False)

iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

data.to_csv("data/preprocessed.csv", index=False)
print("✅ Data preprocessed and saved to data/preprocessed.csv")
