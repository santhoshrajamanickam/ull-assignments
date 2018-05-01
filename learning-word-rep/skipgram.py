import torch
import torch.nn as nn
import torch.nn.functional as F


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
