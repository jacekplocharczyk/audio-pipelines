from allennlp.commands.elmo import ElmoEmbedder
from tqdm import tqdm
import pandas as pd
import numpy as np

elmo_model = ElmoEmbedder("elmo/options.json", "elmo/weights.hdf5", cuda_device=0)

dataset = pd.read_csv('data/data/dataset_test.csv')
texts = dataset['hyps'].apply(lambda x : x.split(' '))
texts = list(texts)

SAVEPATH = "text_vecs_test/"

text_vecs = []
for i in tqdm(range(len(texts))):
    np.save(SAVEPATH + dataset["audio_id"][i] + '.npy', elmo_model.embed_sentence(texts[i]))