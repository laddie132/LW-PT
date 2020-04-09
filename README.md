# DA-QT

## Preprocessing
``` bash
PYTHONHASHSEED=1 python preprocess.py
```

## Pre-Train
``` bash
python pretrain.py --out=[OUT_INFIX]
```

## Build Doc-Rep
``` bash
python build_doc_rep.py --in=[IN_INFIX]
```

## Train Classification
``` bash
python classification.py --out=[OUT_INFIX]
```
