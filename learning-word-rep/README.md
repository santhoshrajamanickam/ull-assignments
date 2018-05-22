## Learning Word Representations

We implemented the following models using Python 3.6 and the PyTorch package. After training, each script saves the PyTorch model state dictionary to disk, together with a serialized object (via pickle) containing a dictionary that maps words (`str`) to indices (`int`) for the given corpus used during training.

### Skip-gram

Example usage:

```
python skipgram_train.py -v 10000 -d 300 -w 5 -c data/europarl/training.en
```
This trains a skip-gram model using a vocabulary of 10.000 words, 300 dimensions, a window of 5 words and the specified corpus.

### Bayesian skip-gram

Example usage:

```
python bayesian_train.py -v 10000 -d 300 -w 5 -c data/europarl/training.en
```
This trains a Bayesian skip-gram using a vocabulary of 10.000 words, 300 dimensions, a window of 5 words and the specified corpus.

### Embed-align

Example usage:

```
python embedalign_train.py -v 10000 -d 300 -c1 data/europarl/training.en -c2 data/europarl/training.fr
```
This trains an embed-align model using a vocabulary of 10.000 words, 300 dimensions, and the two specified corpora.

### Evaluation

For the evaluation on the lexical substitution and alignment tasks we used the following two files:

- [evaluation.py](evaluation.py)
- [test_evaluation.py](test_evaluation.py)
