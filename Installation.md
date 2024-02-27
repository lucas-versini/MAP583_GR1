Si jamais ça aide, ce que j'ai fait pour installer tout ce qui est nécessaire c'est quelque chose comme ça:

```
conda create -n projet_583 python=3.9

conda activate projet_583

git clone https://github.com/yu4u/age-estimation-pytorch

conda install -c anaconda cmake

conda install -c conda-forge dlib

conda install numpy opencv pandas tensorboard tqdm yacs future imgaug

pip install better-exceptions

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

conda install pretrainedmodels
```

Et pour pas avoir de souci pour télécharger certains modèles, dans `train.py` il peut être utile de mettre ça au début:

```
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```
