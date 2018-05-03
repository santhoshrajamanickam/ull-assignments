import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import get_ipython

class Skipgram(nn.Module):
    """ A skip-gram model for learning word embeddings.
    Args:
        - emb_dimensions (int): word embeddings dimensions
        - context_size: number of words used as context
    """
    def __init__(self, vocab_size, emb_dimensions):
        super(Skipgram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_dimensions)
        self.linear = nn.Linear(emb_dimensions, vocab_size, bias=False)

    def forward(self, words, contexts):
        """ Calculates the log-probabilities for all words
        given the inputs.
        Args:
            - inputs (tensor): (N, context_size), a tensor containing word indices
        Returns:
            - tensor: (N, vocab_size), the log-probabilities
        """
        embedding = self.embeddings(words)
        y = self.linear(embedding)
        log_probs = F.log_softmax(y, dim=1)

        losses = torch.sum(torch.gather(log_probs, 1, contexts))

        return -1 * torch.mean(losses)

class SkipgramNS(nn.Module):
    """ A skip-gram model for learning word embeddings.
    Args:
        - emb_dimensions (int): word embeddings dimensions
        - context_size: number of words used as context
    """
    def __init__(self, vocab_size, emb_dimensions):
        super(SkipgramNS, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_dimensions)
        self.out_embeddings = nn.Embedding(vocab_size, emb_dimensions)

    def forward(self, words, pos_contexts, neg_contexts):
        """ Calculates the log-probabilities for all words
        given the inputs.
        Args:
            - inputs (tensor): (N, context_size), a tensor containing word indices
        Returns:
            - tensor: (N, vocab_size), the log-probabilities
        """
        target_emb = self.embeddings(words)
        pos_emb = self.out_embeddings(pos_contexts)
        neg_emb = self.out_embeddings(neg_contexts)

        pos_similarity = torch.sum(torch.sum(target_emb * pos_emb.t(), dim=-1), dim=0)
        batch_pos_loss = F.logsigmoid(pos_similarity)

        neg_similarity = torch.sum(torch.sum(target_emb * neg_emb.t(), dim=-1), dim=0)
        batch_neg_loss = F.logsigmoid(-neg_similarity)

        return -torch.mean(batch_pos_loss + batch_neg_loss)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
V = 1000
d = 300
words = [i for i in range(V)]
pos_contexts = [[i for i in range(4)] for i in range(V)]
neg_contexts = [[i for i in range(4)] for i in range(V)]
words_t = torch.tensor(words, dtype=torch.long).to(device)
pos_contexts_t = torch.tensor(pos_contexts, dtype=torch.long).to(device)
neg_contexts_t = torch.tensor(neg_contexts, dtype=torch.long).to(device)
skipgram = Skipgram(V, 50)
skipgram_ns = SkipgramNS(V, 50)

ipython = get_ipython()
ipython.magic('timeit skipgram(words_t, pos_contexts_t)')
ipython.magic('timeit skipgram_ns(words_t, pos_contexts_t, neg_contexts_t)')
