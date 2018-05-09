import string
from collections import Counter, defaultdict
import numpy as np
import time

class SkipgramCorpus:
    """
    A class with methods to read and generate batches from a corpus
    to train a skip-gram model.
    Args:
        - corpus_path (str): the path of the input text file
        - vocab_size (int): the size of the vocabulary
        - window (int): number of words used as context in the skip-gram
    """
    def __init__(self, corpus_path, vocab_size, window):
        token_count = 0
        line_count = 0
        # First pass through the corpus: get statistics and vocabulary
        word_counts = Counter()
        max_sent_length = 0
        with open(corpus_path) as file:
            for line in file:
                line_count += 1
                # Remove punctuation
                line_words = line.translate(str.maketrans('', '', string.punctuation))
                # Split into tokens
                sentence = line_words.strip().split()
                # Update counts
                word_counts.update(sentence)
                token_count += len(sentence)
                if len(sentence) > max_sent_length:
                    max_sent_length = len(sentence)

        # Get V most common tokens
        vocab_size = min(vocab_size, len(word_counts))
        word2idx = defaultdict(lambda: len(word2idx))
        idx2word = {}
        for word, count in word_counts.most_common(vocab_size):
            idx2word[word2idx[word]] = word
        # Calculate unigram probabilities to the 3/4 power
        self.unigram = np.array([word_counts[word] for word in word2idx]) ** 0.75
        self.unigram /= np.sum(self.unigram)
        self.word_indices = np.arange(vocab_size)

        # Add special symbols
        UNK = '<unk>'
        EOS = '<s>'
        idx2word[word2idx[UNK]] = UNK
        idx2word[word2idx[EOS]] = EOS
        # Lock word2idx
        word2idx = dict(word2idx)

        # Second pass: store corpus in numpy array
        self.sentences = np.empty((line_count, 2 * window + max_sent_length), dtype=int)
        # Pad with EOS where needed
        self.sentences[:, :window] = word2idx[EOS]
        self.sent_lengths = np.empty(line_count, dtype=int)
        with open(corpus_path) as file:
            for i, line in enumerate(file):
                line_words = line.translate(str.maketrans('', '', string.punctuation))
                sent = line_words.strip().split()
                self.sent_lengths[i] = len(sent)
                self.sentences[i, window:window + len(sent)] = list(map(lambda word: word2idx.get(word, word2idx[UNK]), sent))
                self.sentences[i, window + len(sent):] = word2idx[EOS]

        # Keep useful data
        self.window = window
        self.UNK = UNK
        self.EOS = EOS
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.max_sent_length = max_sent_length
        self.token_count = token_count

    def next_batch(self, batch_size):
        """
        A generator of batches for training a skip-gram.
        Args:
            - batch_size (int): size of the batch
        Returns:
            - targets (list): containing batch_size words
            - pos_contexts, neg_contexts (lists): containing batch_size lists of length 2 * window
        """
        # An array to select columns for contexts
        context_idx = np.delete(np.arange(-self.window, self.window + 1), self.window)

        processed = 0
        start = time.time()
        for batch_idx in range(0, len(self.sentences), batch_size):
            if batch_idx - processed >= 10000:
                end = time.time() - start
                print('processed batch {:d}, {:d} sentences in {:.1f} seconds'.format(batch_idx, batch_idx - processed, end))
                start = time.time()
                processed = batch_idx

            # Select sentences for the batch
            batch_sentences = self.sentences[batch_idx:batch_idx + batch_size]
            # Compute maximum length in the batch for efficiency
            batch_lengths = self.sent_lengths[batch_idx:batch_idx + batch_size]
            max_batch_length = np.max(batch_lengths)
            for target_idx in range(self.window, self.window + max_batch_length):
                # Valid sentences have length less than or equal to the target index
                valid_lengths_mask = batch_lengths >= target_idx - self.window
                # Given valid sentences, get targets and positive and negative contexts
                targets = batch_sentences[valid_lengths_mask, target_idx]
                pos_contexts = batch_sentences[valid_lengths_mask][:, context_idx + target_idx]

                yield targets, pos_contexts

    def next_batch_neg_sampling(self, batch_size):
        """
        A generator of batches for training a skip-gram.
        Args:
            - batch_size (int): size of the batch
        Returns:
            - targets (list): containing batch_size words
            - pos_contexts, neg_contexts (lists): containing batch_size lists of length 2 * window
        """
        # An array to select columns for contexts
        context_idx = np.delete(np.arange(-self.window, self.window + 1), self.window)

        processed = 0
        start = time.time()
        for batch_idx in range(0, len(self.sentences), batch_size):
            if batch_idx - processed >= 10000:
                end = time.time() - start
                print('processed batch {:d}, {:d} sentences in {:.1f} seconds'.format(batch_idx, batch_idx - processed, end))
                start = time.time()
                processed = batch_idx

            # Select sentences for the batch
            batch_sentences = self.sentences[batch_idx:batch_idx + batch_size]
            # Compute maximum length in the batch for efficiency
            batch_lengths = self.sent_lengths[batch_idx:batch_idx + batch_size]
            max_batch_length = np.max(batch_lengths)
            # Precompute negative samples for the batch
            batch_neg_contexts = np.random.choice(self.word_indices, (batch_lengths.shape[0], 2*self.window+max_batch_length), p=self.unigram)
            for target_idx in range(self.window, self.window + max_batch_length):
                # Valid sentences have length less than or equal to the target index
                valid_lengths_mask = batch_lengths >= target_idx - self.window
                # Given valid sentences, get targets and positive and negative contexts
                targets = batch_sentences[valid_lengths_mask, target_idx]
                pos_contexts = batch_sentences[valid_lengths_mask][:, context_idx + target_idx]
                neg_contexts = batch_neg_contexts[valid_lengths_mask][:, context_idx + target_idx]

                yield targets, pos_contexts, neg_contexts
