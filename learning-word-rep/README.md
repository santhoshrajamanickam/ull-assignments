## Learning Word Representations

We implemented the following models using Python 3.6 and the PyTorch package.

### Skip-gram embeddings

Example usage:

```
python skipgram_embeddings.py -v 10000 -d 300 -w 5 -c data/europarl/training.en
```
This trains word embeddings using a vocabulary of 10.000 words, 300 dimensions, a window of 5 words and the specified corpus. The embeddings are saved to a text file with one line per word, and values separated by spaces.
