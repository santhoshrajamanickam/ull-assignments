import argparse
from corpus import SkipgramCorpus
from bayes_skipgram import BayesianSkipgram
import torch
from torch.nn import NLLLoss
from torch.optim import Adam

# Read arguments from command line
parser = argparse.ArgumentParser(description='Train word embeddings using a skip-gram model.')
parser.add_argument('-v', '--vocab_size', required=True, help='Size of the vocabulary', type=int)
parser.add_argument('-d', '--emb_dimensions', required=True, help='Dimensions of the embeddings', type=int)
parser.add_argument('-w', '--window', required=True, help='Window size before and after the word', type=int)
parser.add_argument('-c', '--corpus_path', required=True, help='Path of the training corpus', type=str)
args = vars(parser.parse_args())

vocab_size = args['vocab_size']
emb_dimensions = args['emb_dimensions']
window = args['window']
corpus_path = args['corpus_path']

# Load corpus
print('Loading {:s}'.format(corpus_path))
corpus = SkipgramCorpus(corpus_path, vocab_size, window)
print('Loaded corpus with {:d} words, {:d} sentences'.format(corpus.token_count, len(corpus.sentences)))

# Create model (add 2 words for EOS and UNK)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
skipgram = BayesianSkipgram(vocab_size + 2, emb_dimensions).to(device)

# Set up training
loss_function = NLLLoss()
optimizer = Adam(skipgram.parameters())
prev_valid_loss = float('inf')
embeddings = skipgram.embeddings.weight.detach()
batch_size = 1

# Start training
print('Training...')
epochs = 3
for ep in range(epochs):
    # Training corpus iteration
    training_loss = 0
    for words, pos_contexts in corpus.next_batch(batch_size):
        # Clear gradients
        skipgram.zero_grad()
        # Convert to tensors and calculate loss
        words_t = torch.tensor(words, dtype=torch.long).to(device).squeeze()
        pos_contexts_t = torch.tensor(pos_contexts, dtype=torch.long).to(device).squeeze()
        loss = skipgram(words_t, pos_contexts_t)

        # Update weights from loss
        loss.backward()
        optimizer.step()

        training_loss += loss.data.item()

    # Print stats
    avg_train_loss = training_loss / (len(corpus.sentences) / batch_size)
    print('{:2d}/{:2d}: avg_train_loss = {:11.1f}'.format(ep+1, epochs, avg_train_loss))

# Save embeddings
embeddings = skipgram.embeddings.weight.detach()
filename = '{:d}V_{:d}d_{:d}w.words'.format(vocab_size, emb_dimensions, window)
with open(filename, 'w') as file:
    # Omit the last 2 words, corresponding to UNK and EOS
    for i in range(0, len(embeddings) - 2):
        # Write actual word first
        file.write(corpus.idx2word[i])
        # Write values on the rest of the line
        for value in embeddings[i]:
            file.write(' {:.12f}'.format(value.item()))
        file.write('\n')
print('Saved embeddings to {:s}'.format(filename))
