import pandas as pd
import pickle as pkl
import numpy as np
from tqdm import tqdm

get_dir = "../text_vecs/"
text_vec_list = []
dataset = pd.read_csv("data/dataset_dropna.csv")
for i in tqdm(dataset["audio_id"]):
    text_vec_list.append(np.load(get_dir + i + ".npy"))
    
with open("text_vec_list.p", "wb") as f:
    pkl.dump(text_vec_list, f)
    
files = os.listdir("../text_vecs_val")
dataset = pd.read_csv("data/dataset_val.csv")
get_dir = "../text_vecs_val/"
text_vec_list_val = []
for i in dataset["audio_id"]:
    text_vec_list_val.append(np.load(get_dir + i + ".npy"))
    
with open("text_vec_list_val.p", "wb") as f:
    pkl.dump(text_vec_list_val, f)