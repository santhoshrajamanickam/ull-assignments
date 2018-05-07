import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.random import multivariate_normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EmbedAlign(nn.Module):
    """ A skip-gram model for learning word embeddings.
    Args:
        - vocab_size (int): the size of the vocabulary
        - emb_dimensions (int): word embeddings dimensions
    """
    def __init__(self, vocab_size1, vocab_size2, emb_dimensions):
        super(EmbedAlign, self).__init__()
        self.embeddings = nn.Embedding(vocab_size1, emb_dimensions)
        self.bilstm = nn.LSTM(emb_dimensions, emb_dimensions, bidirectional=False)
        self.affine1_mu = nn.Linear(emb_dimensions, emb_dimensions)
        self.affine2_mu = nn.Linear(emb_dimensions, emb_dimensions)
        self.affine1_sigma = nn.Linear(emb_dimensions, emb_dimensions)
        self.affine2_sigma = nn.Linear(emb_dimensions, emb_dimensions)
        self.affine1_L1 = nn.Linear(emb_dimensions, emb_dimensions)
        self.affine2_L1 = nn.Linear(emb_dimensions, vocab_size1)
        self.affine1_L2 = nn.Linear(emb_dimensions, emb_dimensions)
        self.affine2_L2 = nn.Linear(emb_dimensions, vocab_size2)

    def forward(self, sentence1, sentence2):
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
        sent_embeddings = self.embeddings(sentence1)
        m1 = sent_embeddings.shape[0]
        out, hidden = self.bilstm(sent_embeddings.view(m1, 1, -1))
        mu = self.affine2_mu(F.relu(self.affine1_mu(out.squeeze())))
        sigma = F.softplus(self.affine2_sigma(F.relu(self.affine1_sigma(out.squeeze()))))
        eps = torch.tensor(multivariate_normal(np.zeros(d), np.identity(d), m1), dtype=torch.float).to(device)
        z = mu + eps * sigma

        xk_log_probs = F.log_softmax(self.affine2_L1(F.relu(self.affine1_L1(z))))
        xk_sum = torch.sum(torch.gather(xk_log_probs, 1, sentence1.view(-1, 1)))

        yk_log_probs = F.log_softmax(self.affine2_L2(F.relu(self.affine1_L2(z))))
        yk_sum = torch.sum(torch.mean(yk_log_probs[:, sentence2], dim=0))

        kl_div = 0.5 * torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)

        return xk_sum + yk_sum + kl_div

V = 1000
d = 300
sentence1 = torch.tensor([i for i in range(9)], dtype=torch.long)
sentence2 = torch.tensor([i for i in range(11)], dtype=torch.long)
embalign = EmbedAlign(V, V, d)
embalign(sentence1, sentence2)
