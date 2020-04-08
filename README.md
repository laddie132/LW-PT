# DA-QT

## Preprocessing
``` python
PYTHONHASHSEED=1 python preprocess.py
```

## Pre-Train
``` python
python pretrain.py --out=[OUT_INFIX]
```

## Build Doc-Rep
``` python
python build_doc_rep.py --in=[IN_INFIX]
```

## Train Classification
``` python
python classification.py --out=[OUT_INFIX]
```
