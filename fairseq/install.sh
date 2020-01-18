#!/bin/bash

apt-get install g++

git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable .
pip install torchaudio
# if there is a bug you can reinstall torchvision pip uninstall torchvision; pip install torchvision==0.3.0
cd ..

wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt