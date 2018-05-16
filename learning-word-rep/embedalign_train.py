import argparse
from corpus import EmbedAlignCorpus
from embed_align import EmbedAlign
import torch
from torch.optim import Adam
import pickle

torch.manual_seed(42)

# Read arguments from command line
parser = argparse.ArgumentParser(description='Train word embeddings using a skip-gram model.')
parser.add_argument('-v', '--vocab_size', required=True, help='Size of the vocabulary', type=int)
parser.add_argument('-d', '--emb_dimensions', required=True, help='Dimensions of the embeddings', type=int)
parser.add_argument('-c1', '--corpus1_path', required=True, help='Path of corpus 1', type=str)
parser.add_argument('-c2', '--corpus2_path', required=True, help='Path of corpus 2', type=str)
args = vars(parser.parse_args())

vocab_size = args['vocab_size']
emb_dimensions = args['emb_dimensions']
corpus1_path = args['corpus1_path']
corpus2_path = args['corpus2_path']

# Load corpus
print('Loading {:s} and {:s}'.format(corpus1_path, corpus2_path))
corpus = EmbedAlignCorpus(corpus1_path, corpus2_path, vocab_size)
print('Loaded corpus 1 with {:d} words, {:d} sentences'.format(corpus.corpus1.token_count, len(corpus.corpus1.sentences)))
print('Loaded corpus 2 with {:d} words, {:d} sentences'.format(corpus.corpus2.token_count, len(corpus.corpus2.sentences)))

# Create model (add 2 words for EOS and UNK)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embed_align = EmbedAlign(vocab_size + 2, vocab_size + 2, emb_dimensions).to(device)

# Set up training
optimizer = Adam(embed_align.parameters())
batch_size = 128
n_batches = len(corpus.corpus1.sentences) / batch_size

# Start training
print('Training...')
epochs = 3
for ep in range(epochs):
    # Training corpus iteration
    avg_train_loss = 0
    processed = 0
    for i, (sent1, sent2) in enumerate(corpus.next_batch(batch_size)):
        # Clear gradients
        embed_align.zero_grad()
        # Convert to tensors and calculate loss
        sent1_t = torch.tensor(sent1, dtype=torch.long).to(device)
        sent2_t = torch.tensor(sent2, dtype=torch.long).to(device)
        loss = embed_align(sent1_t, sent2_t)

        # Update weights from loss
        loss.backward()
        optimizer.step()

        # Monitor loss
        loss_value = loss.data.item()
        avg_train_loss += loss_value / n_batches
        if i - processed >= 79:
            print('batch {:d} loss: {:.1f}'.format(i, loss_value))
            processed = i

    # Print stats
    print('{:2d}/{:2d}: avg_train_loss = {:11.1f}'.format(ep+1, epochs, avg_train_loss))

# Save model
filename = '{:d}V_{:d}d_EmbedAlign.pt'.format(vocab_size, emb_dimensions)
torch.save(embed_align.state_dict(), filename)
print('Saved model to {:s}'.format(filename))

# Save word2idx
pickle.dump(corpus.corpus1.word2idx, open('word2idx1.p', 'wb'))
pickle.dump(corpus.corpus1.word2idx, open('word2idx2.p', 'wb'))
