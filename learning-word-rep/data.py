import string
from collections import Counter, defaultdict
import numpy as np
import time


class SkipgramCorpus:
    def __init__(self, corpus_path, vocab_size):
        sentences = []
        token_count = 0
        # Get word frequency first
        word_counts = Counter()
        with open(corpus_path) as file:
            for line in file:
                # Remove punctuation
                line_words = line.translate(str.maketrans('', '', string.punctuation))
                # Split into tokens
                sentence = line_words.strip().split()
                # Update counts
                word_counts.update(sentence)
                token_count += len(sentence)
                # Store sentence
                sentences.append(sentence)

        # Get V most common tokens
        word2idx = defaultdict(lambda: len(word2idx))
        idx2word = {}
        for word_count in word_counts.most_common(vocab_size):
            word = word_count[0]
            idx2word[word2idx[word]] = word
        # Add special symbols
        UNK = '<unk>'
        EOS = '<s>'
        idx2word[word2idx[UNK]] = UNK
        idx2word[word2idx[EOS]] = EOS
        # Lock word2idx
        word2idx = dict(word2idx)

        # Transform words to indices and drop uncommon words
        self.sentences = [list(map(lambda word: word2idx.get(word, word2idx[UNK]), sent)) for sent in sentences]

        self.UNK = UNK
        self.EOS = EOS
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.token_count = token_count

    def next_word_context_pair(self, window, batch_size):
        context_idx = np.delete(np.arange(-window, window+1), window)
        processed = 0
        start = time.time()
        for batch_idx in range(0, len(self.sentences), batch_size):
            if batch_idx - processed >= 10000:
                end = time.time() - start
                print('processed batch {:d}, {:d} sentences in {:.1f} seconds'.format(batch_idx, batch_idx - processed, end))
                start = time.time()
                processed = batch_idx
            batch_sents = self.sentences[batch_idx:batch_idx + batch_size]
            target_idx = 0
            batch_done = False
            while not batch_done:
                targets = []
                contexts = []
                for sentence in batch_sents:
                    if target_idx >= len(sentence):
                        continue

                    context_words = []
                    # Collect words within the window.
                    # Out of sentence indices are replaced by the EOS symbol
                    for i in target_idx + context_idx:
                        if i < 0 or i >= len(sentence):
                            context_words.append(self.word2idx[self.EOS])
                        else:
                            context_words.append(sentence[i])
                    targets.append(sentence[target_idx])
                    contexts.append(context_words)
                if len(targets) > 0:
                    yield targets, contexts
                    target_idx += 1
                else:
                    batch_done = True
