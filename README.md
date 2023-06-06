# Contrastive_Learning_MM_FND

Code for the paper - "Identifying multimodal misinformation leveraging novelty detection and emotion recognition"

```environment.yml``` - conda environment for setup.

# Dataset

Sub-directories-

1. Source - Contains four .csv files ReCovery.csv, TICNN.csv, Reuters.csv  and Snopes.csv

2. Target - Contains four .csv files ReCovery.csv, TICNN.csv, Reuters.csv  and Snopes.csv

Each Source File contains attributes such as ID, number of Sources, Source url, Source Text, Source Image url and Source Reliability.

Each Target File contains attributes such as ID, Target url, Target Text, Target Image url and Label

Note:

Reuters.csv and Snopes.csv are two subfiles from fauxtography Dataset

Images can be downloaded by using image_url, which is provided in .csv files.

TICNN.csv file in source folder is uploaded in two subparts due to GitHub file size uploading constraints (TICNN_1.csv, TICNN_2.csv).

## Evaluation

### Extract Features

1. ResNet+BERT

```
python build_multimodal_rbert.py
```

2. Visual BERT

```
python vb_source.py
```
```
python vb_target.py
```

### Baselines

Run the ```baselines.ipynb``` python notebook.

### SOTA Comparisons

1. SAFE: We use [SAFE](https://github.com/Jindi0/SAFE) as one of our baselines. The code for running on our data is present in the Comparison_SOTA/SAFE folder.

2. EANN: We use [EANN](https://github.com/yaqingwang/EANN-KDD18) as another baseline. The code for running EANN on our data is present in Comparison_SOTA/EANN.

### Contrastive Learning based Novelty Model

Run the ```contrastive_ce.ipynb``` python notebook.

### Emotion Representations

1. Emotion Dataset extraction - Run ```Emotion/extract_mm_emotion.ipynb```

2. Emotion Level Pre-Training and Emotion Representations - Run ```Emotion/finetune_rf.ipynb```

### Proposed Model

Run the ```nv_em_ce.ipynb``` python notebook.
