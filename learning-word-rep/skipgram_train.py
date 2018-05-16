import argparse
from corpus import SkipgramCorpus
from skipgram import Skipgram
import torch
from torch.optim import Adam
import pickle

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
skipgram = Skipgram(vocab_size + 2, emb_dimensions).to(device)

# Set up training
optimizer = Adam(skipgram.parameters())
batch_size = 128

# Start training
print('Training...')
epochs = 3
for ep in range(epochs):
    # Training corpus iteration
    training_loss = 0
    for words, pos_contexts, neg_contexts in corpus.next_batch_neg_sampling(batch_size):
        # Clear gradients
        skipgram.zero_grad()
        # Convert to tensors and calculate loss
        words_t = torch.tensor(words, dtype=torch.long).to(device)
        pos_contexts_t = torch.tensor(pos_contexts, dtype=torch.long).to(device)
        neg_contexts_t = torch.tensor(neg_contexts, dtype=torch.long).to(device)
        loss = skipgram(words_t, pos_contexts_t, neg_contexts_t)

        # Update weights from loss
        loss.backward()
        optimizer.step()

        training_loss += loss.data.item()

    # Print stats
    avg_train_loss = training_loss / (len(corpus.sentences) / batch_size)
    print('{:2d}/{:2d}: avg_train_loss = {:11.1f}'.format(ep+1, epochs, avg_train_loss))

# Save model
filename = '{:d}V_{:d}d_{:d}w_Skipgram.pt'.format(vocab_size, emb_dimensions, window)
torch.save(skipgram.state_dict(), filename)
print('Saved model to {:s}'.format(filename))

# Save word2idx
pickle.dump(corpus.word2idx, open('word2idx.p', 'wb'))
