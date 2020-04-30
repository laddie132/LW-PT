# Label-Wised Quick-Thought (LW-QT)
This is the code for 2020 paper [Label-Wised Document Pre-Training for Multi-Label Text Classiﬁcation](https://)

## Requirements

- Ubuntu 16.04
- Python >= 3.6.0
- PyTorch >= 1.3.0

## Reproducibility

- `--data` and `--outputs`

We provide the proprecessed RMSC and AAPD datasets and pretrained checkpoints of LW-LSTM+QT+FT model and HLWAN+QT+FT model to make sure reproducibility. Please download from the [link](https://) and decompress to the root directory of this repository.

```
--data
    |--aapd
    	|--label_test
    	|--label_train
    	...
    |--rmsc
    	|--rmsc.data.test.json
    	|--rmsc.data.train.json
    	|--rmsc.data.valid.json
    aapd_word2vec.model
    aapd_word2vec.model.wv.vectors.npy
    aapd.meta.json
    aapd.pkl
    rmsc_word2vec.model
    rmsc_word2vec.model.wv.vectors.npy
    rmsc.meta.json
    rmsc.pkl
--outputs
    |--aapd
    |--rmsc
```

> Note that the `data/aapd`and `data/rmsc` is the initial dataset. Here we provide a split of RMSC (i.e. RMSC-V2).

- Testing on AAPD
``` bash
python classification.py -config=aapd.yaml -in=aapd -gpuid [GPU_ID] -test
```

- Testing on RMSC
``` bash
python classification.py -config=rmsc.yaml -in=rmsc -gpuid [GPU_ID] -test
```

## Preprocessing
If you want to preprocess the dataset by yourself,  just run the following command with name of dataset (e.g. RMSC or AAPD).
``` bash
PYTHONHASHSEED=1 python preprocess.py -data=[RMSC/AAPD]
```
> Note that `PYTHONHASHSEED` is used in word2vec.

## Pre-Train

Pre-train the LW-QT model.

``` bash
python pretrain.py -config=[CONFIG_NAME] -out=[OUT_INFIX] -gpuid [GPU_ID] -train -test
```

- `CONFIG_NAME`: `aapd.yaml` or `rmsc.yaml`
- `OUT_INFIX`: infix of outputs directory contains logs and checkpoints

## MLTC Task

Train the downstream model for MLTC task.

``` bash
python classification.py -config=[CONFIG_NAME] -in=[IN_INFIX] -out=[OUT_INFIX] -gpuid [GPU_ID] -train -test
```

- `IN_INFIX`: infix of inputs directory contains pre-trained checkpoints

## Others

- build a static documents representation to facilitate downstream tasks
``` bash
python build_doc_rep.py -config=[CONFIG_NAME] -in=[IN_INFIX] -gpuid [GPU_ID]
```
> Not used unless necessary.

- make RMSC-V2 dataset: `tests/make_rmsc.py`
- visual document embeddings: `tests/visual_emb.py`
- visual labels F1 score: `tests/visual_label_f1.py`
- case study: `tests/case_study.py`

## Reference

If you consider our work useful, please cite the paper:

```
TODO
```
