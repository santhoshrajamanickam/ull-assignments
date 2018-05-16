from collections import defaultdict
import string
import torch
from skipgram import Skipgram
from bayes_skipgram import BayesianSkipgram
import pickle
import re

class Eval:
    def __init__(self, model, window_size=None):
        self.model = model
        self.window_size = window_size
        self.test_sentences = defaultdict(lambda : defaultdict(lambda : []))
        self.candidates = defaultdict(lambda : set())

    def load_test_sentences(self, test_sentences_path, candidates_path):
        # Read sentences with targets and contexts
        with open(test_sentences_path) as file:
            for line in file:
                # Split into tokens
                values = line.strip().split()
                target_word = values[0]
                id_sentence = values[1]
                init_target_position = int(values[2])

                # Sentence with punctuation
                sentence_all = values[3:]
                # Sentence without punctuation
                sentence = []
                # Keep track of the new position of the target
                target_position = init_target_position
                for i, word in enumerate(sentence_all):
                    if word not in string.punctuation:
                        # If current index is target, new target_position is
                        # length of sentence without punctuation
                        if i == init_target_position:
                            target_position = len(sentence)
                        sentence.append(word)

                # Extract context words
                start = int(target_position) - self.window_size
                end = int(target_position) + self.window_size
                # Calculate padding if necessary
                start_padding = 0
                end_padding = 0
                if start < 0:
                    start_padding = abs(start)
                    start = 0
                if end > len(sentence):
                    end_padding = end - len(sentence)
                    end = len(sentence)

                # Add start padding
                context_words = ['<s>'] * start_padding
                # Add context words
                context_words += sentence[start:target_position] + sentence[target_position+1:end]
                # Add end padding
                context_words += ['<s>'] * end_padding

                self.test_sentences[target_word][id_sentence] = context_words

        # Read candidates per word
        with open(candidates_path) as file:
            for line in file:
                # Remove all sorts of punctuation
                candidates = re.split('\W+',line)
                target_word = candidates[0]
                self.candidates[target_word] = candidates[2:]


    def score_context_words(self):
        missing_count = 0
        target_count = 0
        cos = torch.nn.CosineSimilarity(dim=0)

        # Load word2idx used to train the model
        word2idx = pickle.load(open('word2idx.p', 'rb'))

        if self.model == 'skipgram':
            # Initialize model (add 2 words to vocab_size for UNK and EOS)
            skipgram = Skipgram(vocab_size=10002, emb_dimensions=300)
            # Load from training results
            skipgram.load_state_dict(torch.load('10000V_300d_5w_Skipgram.pt', map_location='cpu'))

            # print(self.test_sentences.keys())
            file = open('./lst_skipgram.out', 'w')

            for target_word in self.test_sentences.keys():

                target_count += 1

                # Get the input embedding of a word
                words = target_word.split(".")
                target = words[0]
                if target in word2idx:
                    target_emb = skipgram.embeddings(torch.tensor(word2idx[target], dtype=torch.long))
                else:
                    # target_emb = torch.zeros(target_emb.shape)
                    target_emb = skipgram.embeddings(torch.tensor(word2idx['<unk>'], dtype=torch.long))
                    print('missing target: ' + str(target_word))
                    missing_count += 1
                    # continue

                candidate_embs =defaultdict()
                for candidate in self.candidates[target]:
                    if candidate in word2idx:
                        candidate_embs[candidate] = skipgram.embeddings(torch.tensor(word2idx[candidate], dtype=torch.long))
                    else:
                        candidate_embs[candidate] = torch.zeros(target_emb.shape)

                for sentence_id in self.test_sentences[target_word].keys():
                    file.write('RANKED\t')
                    file.write(str(target_word)+' ')
                    file.write(str(sentence_id))

                    for candidate in self.candidates[target]:
                        score = 0
                        score += cos(target_emb,candidate_embs[candidate])
                        num_context = 0
                        for context_word in self.test_sentences[target_word][sentence_id]:
                            if context_word in word2idx:
                                context_emb = skipgram.out_embeddings(
                                    torch.tensor(word2idx[context_word], dtype=torch.long))
                                score += cos(candidate_embs[candidate],context_emb)
                                num_context += 1

                        score /= num_context + 1
                        score = score.data.item()
                        file.write('\t' + str(candidate) + ' ' + str(score))
                    file.write('\n')

            file.close()
            print('Target count: '+str(target_count))
            print('Missing count: '+str(missing_count))

        elif self.model == 'bayesian':
            # Initialize model (add 2 words to vocab_size for UNK and EOS)
            bayesian = BayesianSkipgram(vocab_size=10002, emb_dimensions=300)
            # Load from training results
            bayesian.load_state_dict(torch.load('10000V_300d_5w_BayesianSkipgram.pt', map_location='cpu'))

            # print(self.test_sentences.keys())
            file = open('./lst_bayesian.out', 'w')

            for target_word in self.test_sentences.keys():



