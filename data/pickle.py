import pandas as pd

df = pd.read_pickle("validation_dataset_n10.pkl")
df.to_csv("data.csv", index=False)

print("Saved data.csv")