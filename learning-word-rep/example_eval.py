import torch
from skipgram import Skipgram
import pickle

# Initialize model (add 2 words to vocab_size for UNK and EOS)
skipgram = Skipgram(vocab_size=10002, emb_dimensions=300)
# Load from training results
skipgram.load_state_dict(torch.load('10000V_300d_5w_Skipgram.pt', map_location='cpu'))

# Load word2idx used to train the model
word2idx = pickle.load(open('word2idx.p', 'rb'))

# Get the input embedding of a word
in_word = 'coal'
in_emb = skipgram.embeddings(torch.tensor(word2idx[in_word], dtype=torch.long))

# Get the output embedding for 2 context words
context1 = 'energy'
context2 = 'lady'
out_emb1 = skipgram.out_embeddings(torch.tensor(word2idx[context1], dtype=torch.long))
out_emb2 = skipgram.out_embeddings(torch.tensor(word2idx[context2], dtype=torch.long))

# Compute their dot product
print('Input word: {:s}'.format(in_word))
print('Context measure with {:s}: {:.3f}'.format(context1, torch.dot(in_emb, out_emb1).item()))
print('Context measure {:s}: {:.3f}'.format(context2, torch.dot(in_emb, out_emb2).item()))
