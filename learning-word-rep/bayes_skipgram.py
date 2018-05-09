import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.random import multivariate_normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BayesianSkipgram(nn.Module):
    def __init__(self, vocab_size, emb_dimensions):
        super(BayesianSkipgram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_dimensions)
        self.affine_mu = nn.Linear(2 * emb_dimensions, emb_dimensions)
        self.affine_sigma = nn.Linear(2 * emb_dimensions, emb_dimensions)
        self.affine_ck = nn.Linear(emb_dimensions, vocab_size)
        self.L = nn.Embedding(vocab_size, emb_dimensions)
        self.linear_L = nn.Linear(emb_dimensions, emb_dimensions, bias=False)
        self.S = nn.Embedding(vocab_size, emb_dimensions)
        self.linear_S = nn.Linear(emb_dimensions, emb_dimensions, bias=False)

    def forward(self, word, context):
        word_emb = self.embeddings(word)
        context_embs = self.embeddings(context)
        n_batch, n_context, n_dim = context_embs.shape

        concat_emb = torch.cat((word_emb.expand(n_context, -1, -1).t(), context_embs), dim=-1)
        concat_sum = torch.sum(F.relu(concat_emb), dim=1)

        mu = self.affine_mu(concat_sum)
        log_sigma2 = self.affine_sigma(concat_sum)
        eps = torch.tensor(multivariate_normal(np.zeros(n_dim), np.identity(n_dim), n_batch), dtype=torch.float).to(device)
        z = mu + eps * torch.sqrt(torch.exp(log_sigma2))
        ck_logprobs = F.log_softmax(self.affine_ck(z), dim=1)
        ck_sum = ck_logprobs.gather(1, context).sum(dim=1)
        ck_batch_loss = ck_sum.mean()

        mu_x = self.linear_L(self.L(word))
        sigma_x = F.softplus(self.linear_S(self.S(word)))
        KL_div = torch.log(sigma_x) -0.5 * log_sigma2 + (torch.exp(log_sigma2) + (mu - mu_x)**2)/(2 * sigma_x**2) - 0.5
        KL_sum = KL_div.sum(dim=1)
        KL_batch_loss = KL_sum.mean()

        return -ck_batch_loss + KL_batch_loss


