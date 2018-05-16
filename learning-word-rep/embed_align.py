import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EmbedAlign(nn.Module):
    """ A skip-gram model for learning word embeddings.
    Args:
        - vocab_size (int): the size of the vocabulary
        - emb_dimensions (int): word embeddings dimensions
    """
    def __init__(self, vocab_size1, vocab_size2, emb_dimensions):
        super(EmbedAlign, self).__init__()
        # Encoder parameters
        self.embeddings = nn.Embedding(vocab_size1, emb_dimensions)
        self.bilstm = nn.LSTM(emb_dimensions, emb_dimensions, bidirectional=True)
        self.affine1_mu = nn.Linear(emb_dimensions, emb_dimensions)
        self.affine2_mu = nn.Linear(emb_dimensions, emb_dimensions)
        self.affine1_sigma = nn.Linear(emb_dimensions, emb_dimensions)
        self.affine2_sigma = nn.Linear(emb_dimensions, emb_dimensions)
        # Decoder parameters
        self.affine1_L1 = nn.Linear(emb_dimensions, emb_dimensions)
        self.affine2_L1 = nn.Linear(emb_dimensions, vocab_size1)
        self.affine1_L2 = nn.Linear(emb_dimensions, emb_dimensions)
        self.affine2_L2 = nn.Linear(emb_dimensions, vocab_size2)
        # A distribution to sample for the reparameterization trick
        self.normal_dist = MultivariateNormal(torch.zeros(emb_dimensions), torch.eye(emb_dimensions))
        self.emb_dimensions = emb_dimensions

    def forward(self, sentence1, sentence2):
        """ Calculates the ELBO for the Embed-align model.
        Args:
            - sentence1 (tensor): (N, m), a tensor containing sentences in L1
            - sentence2 (tensor): (N, n), a tensor containing sentences in L2
        Returns:
            - tensor: (1), the loss.
        """
        # Get embeddings for words in sentence1
        sent_embeddings = self.embeddings(sentence1)
        n_batch, m1, n_dim = sent_embeddings.shape

        # - Encoder -
        # Get output from bidirectional LSTM
        out, _ = self.bilstm(sent_embeddings.t())
        # Sum hidden states from both directions
        out = (out[:, :, :self.emb_dimensions] + out[:, :, :self.emb_dimensions]).t()
        # Calculate inference parameters
        mu = self.affine2_mu(F.relu(self.affine1_mu(out)))
        sigma = F.softplus(self.affine2_sigma(F.relu(self.affine1_sigma(out))))

        # - Evidence Lower Bound (ELBO) -
        # Reparameterization trick: get a sample
        eps = self.normal_dist.sample(torch.Size([n_batch, m1])).to(device)
        z = mu + eps * sigma
        # Log-likelihood for sentence1
        xk_log_probs = F.log_softmax(self.affine2_L1(F.relu(self.affine1_L1(z))), dim=-1)
        xk_sum = torch.sum(torch.gather(xk_log_probs, 2, sentence1.view(n_batch, m1, 1)), dim=1)
        batch_xk_loss = torch.mean(xk_sum)
        # Log-likelihood for sentence2
        yk_log_probs = F.log_softmax(self.affine2_L2(F.relu(self.affine1_L2(z))), dim=-1)
        yk_avg = torch.mean(torch.gather(yk_log_probs, 2, sentence2.expand(m1,-1,-1).t()), dim=1)
        yk_sum = torch.sum(yk_avg, dim=-1)
        batch_yk_loss = torch.mean(yk_sum)
        # KL-divergence
        kl_div = torch.sum(-0.5 * torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2, dim=-1), dim=-1)
        batch_kl = torch.mean(kl_div)

        return -batch_xk_loss -batch_yk_loss + batch_kl
