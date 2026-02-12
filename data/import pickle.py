import pickle

with open("validation_dataset_n10.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))