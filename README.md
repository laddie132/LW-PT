# Label-Wised Quick-Thought (LW-QT)

## Preprocessing
``` bash
PYTHONHASHSEED=1 python preprocess.py -data=[DATA_NAME]
```

## Pre-Train
``` bash
python pretrain.py -out=[OUT_INFIX] -train -test
```

## Build Doc-Rep
``` bash
python build_doc_rep.py -in=[IN_INFIX]
```

## Train Classification
``` bash
python classification.py -in=[IN_INFIX] -out=[OUT_INFIX] -train -test
```

## Others

- make RMSC-V2 dataset: `tests/make_rmsc.py`
- visual document embeddings: `tests/visual_emb.py`
- visual labels F1 score: `tests/visual_label_f1.py`