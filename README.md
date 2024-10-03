# AtSpeed
This is the pytorch implementation of our paper
> Efficient Inference for Large Language Model-based Generative Recommendation

## Environment
- Anaconda 3
- Python 3.9.0
- pytorch 1.13.0
- transformers 4.41.0

## Usage
### Data
```bash
data/
├── beauty
├── games
```

### Train
```bash
code/
├── train_KD.py
├── KDModels.py
├── utils.py
├── data.py
├── generation_trie.py
├── collator.py
```

### Evaluation
```bash
code/
├── eval.py
├── KDModels.py
├── beamSD/
│   ├── beamSD.py
```