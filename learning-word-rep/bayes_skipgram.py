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
        k = context_embs.shape[0]
        d = context_embs.shape[1]
        concat_emb = torch.zeros(2 * d, requires_grad=True).to(device)

        for i in range(context_embs.shape[0]):
            concat_emb = concat_emb + F.relu(torch.cat((word_emb, context_embs[i])))

        mu = self.affine_mu(concat_emb)
        log_sigma2 = self.affine_sigma(concat_emb)
        eps = torch.tensor(multivariate_normal(np.zeros(d), np.identity(d), k), dtype=torch.float).to(device)
        z = mu + eps * torch.sqrt(torch.exp(log_sigma2))
        ck_sum = torch.sum(torch.gather(F.log_softmax(self.affine_ck(z), dim=0), 1, context.view(-1, 1)), dim=0)
        mu_x = self.linear_L(self.L(word))
        sigma_x = F.softplus(self.linear_S(self.S(word)))
        KL_div = torch.sum(torch.log(sigma_x) -0.5 * log_sigma2 + (torch.exp(log_sigma2) + (mu - mu_x)**2)/(2 * sigma_x**2) - 0.5)

        return -ck_sum + KL_div


