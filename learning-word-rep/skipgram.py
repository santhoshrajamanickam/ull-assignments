import torch
import torch.nn as nn
import torch.nn.functional as F


class Skipgram(nn.Module):
    """ A skip-gram model for learning word embeddings.
    Args:
        - vocab_size (int): the size of the vocabulary
        - emb_dimensions (int): word embeddings dimensions
    """
    def __init__(self, vocab_size, emb_dimensions):
        super(Skipgram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_dimensions)
        self.out_embeddings = nn.Embedding(vocab_size, emb_dimensions)

    def forward(self, words, pos_contexts, neg_contexts):
        """ Calculates the skip-gram loss using the negative sampling objective.
        N is the batch size and c is the context size, usually equal to 2*w where
        w is the window size before and after the target word.
        Args:
            - words (tensor): (N), a tensor containing target word indices
            - pos_contexts (tensor): (N, c), a tensor containing positive contexts
            - neg_contexts (tensor): (N, c), a tensor containing negative contexts
        Returns:
            - tensor: (1), the skip-gram loss.
        """
        # Get word and context embeddings
        target_emb = self.embeddings(words)
        pos_emb = self.out_embeddings(pos_contexts)
        neg_emb = self.out_embeddings(neg_contexts)

        # Calculate loss from positive contexts
        pos_similarity = torch.bmm(pos_emb, target_emb.unsqueeze(2)).squeeze()
        pos_similarity = torch.sum(pos_similarity, dim=-1)
        pos_loss = F.logsigmoid(pos_similarity).squeeze()

        # Calculate loss from negative contexts
        neg_similarity = torch.bmm(neg_emb, target_emb.unsqueeze(2)).squeeze()
        neg_similarity = torch.sum(neg_similarity, dim=-1)
        neg_loss = F.logsigmoid(-neg_similarity).squeeze()

        return -torch.mean(pos_loss + neg_loss)
