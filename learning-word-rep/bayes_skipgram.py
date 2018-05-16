import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BayesianSkipgram(nn.Module):
    """ A Bayesian skip-gram model for learning distributions of
    word embeddings.
        Args:
            - vocab_size (int): the size of the vocabulary
            - emb_dimensions (int): word embeddings dimensions
        """
    def __init__(self, vocab_size, emb_dimensions):
        super(BayesianSkipgram, self).__init__()
        # Encoder parameters
        self.embeddings = nn.Embedding(vocab_size, emb_dimensions)
        self.affine_mu = nn.Linear(2 * emb_dimensions, emb_dimensions)
        self.affine_sigma = nn.Linear(2 * emb_dimensions, emb_dimensions)

        # Decoder parameters
        self.affine_ck = nn.Linear(emb_dimensions, vocab_size)
        self.L = nn.Embedding(vocab_size, emb_dimensions)
        self.S = nn.Embedding(vocab_size, emb_dimensions)

        # A distribution to sample for the reparameterization trick
        self.normal_dist = MultivariateNormal(torch.zeros(emb_dimensions), torch.eye(emb_dimensions))

    def forward(self, word, context):
        """ Calculates the ELBO for the Bayesian skipgram model.
        N is the batch size and c is the context size, usually equal to 2*w where
        w is the window size before and after the target word.
        Args:
            - word (tensor): (N), a tensor containing target word indices
            - context (tensor): (N, c), a tensor containing positive contexts
        Returns:
            - tensor: (1), the loss.
        """
        # - Encoder -
        # Get word and context embeddings
        word_emb = self.embeddings(word)
        context_embs = self.embeddings(context)
        n_batch, n_context, n_dim = context_embs.shape
        # Concatenate word and context embeddings
        concat_emb = torch.cat((word_emb.expand(n_context, -1, -1).t(), context_embs), dim=-1)
        concat_sum = torch.sum(F.relu(concat_emb), dim=1)
        # Calculate inference parameters
        mu = self.affine_mu(concat_sum)
        sigma = F.softplus(self.affine_sigma(concat_sum))

        # - Evidence Lower Bound (ELBO) -
        # Reparameterization trick: get a sample
        eps = self.normal_dist.sample(torch.Size([n_batch])).to(device)
        z = mu + eps * sigma
        # Log-likelihood
        ck_logprobs = F.log_softmax(self.affine_ck(z), dim=1)
        ck_sum = ck_logprobs.gather(1, context).sum(dim=1)
        ck_batch_loss = ck_sum.mean()
        # KL-divergence
        mu_x = self.L(word)
        sigma_x = F.softplus(self.S(word))
        KL_div = torch.log(sigma_x) - torch.log(sigma) + 0.5 * (sigma**2 + (mu - mu_x)**2)/sigma_x**2 - 0.5
        KL_sum = KL_div.sum(dim=1)
        KL_batch_loss = KL_sum.mean()

        return -ck_batch_loss + KL_batch_loss
