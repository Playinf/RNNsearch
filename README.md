# RNNsearch
An implementation of RNNsearch using Tensorflow, the model is the same
with [GroundHog](https://github.com/lisa-groundhog/GroundHog), a
[Theano version](https://github.com/XMUNLP/RNNsearch) is also available


## Usage

### Data Preprocessing
Preprocessing scripts can be found at [here]
(https://github.com/XMUNLP/RNNsearch)

1. Build vocabulary
  * Build source vocabulary
  ```
  python scripts/buildvocab.py --corpus zh.txt --output vocab.zh.pkl
                               --limit 30000 --groundhog
  ```
  * Build target vocabulary
  ```
  python scripts/buildvocab.py --corpus en.txt --output vocab.en.pkl
                               --limit 30000 --groundhog
  ```
2. Shuffle corpus (Optional)
```
python scripts/shuffle.py --corpus zh.txt en.txt
```

### Training
```
  python rnnsearch.py train --model nmt --corpus zh.txt.shuf en.txt.shuf
      --vocab zh.vocab.pkl en.vocab.pkl --embdim 620 --hidden 1000
      --attention 1000 --alpha 5e-4 --norm 1.0 --batch 128 --maxepoch 5
      --seed 1234 --freq 1000 --vfreq 1500 --dfreq 50 --sort 20
      --references nist02.ref0 nist02.ref1 nist02.ref2 nist02.ref3
      --validation nist02.src
```
### Decoding
```
  python rnnsearch.py translate --model nmt.best.pkl < input > translation
```
