from collections import defaultdict
import string
import torch
from skipgram import Skipgram
import pickle
import re

class Eval:

    def __init__(self, model, window_size=None):
        self.model = model
        self.window_size = window_size
        self.test_sentences = defaultdict(lambda : defaultdict(lambda : []))
        self.candidates = defaultdict(lambda : set())

    def load_test_sentences(self, test_sentences_path, candidates_path):

        with open(test_sentences_path) as file:
            for line in file:
                num_padding = 0
                # Split into tokens
                sentence = line.strip().split()
                target_word = sentence.pop(0)
                id_sentence = sentence.pop(0)
                target_position = sentence.pop(0)
                # Remove the target word
                sentence.pop(int(target_position))
                # Extract context words
                start = int(target_position) - self.window_size
                end = int(target_position) + 1 + self.window_size
                if start < 0:
                    num_padding += abs(start)
                    start = 0
                if end > len(sentence):
                    num_padding += end - len(sentence)
                    end = len(sentence)
                context_words = sentence[start:int(target_position)] + sentence[int(target_position):end]
                # Remove punctuation
                context_words = [s.translate(str.maketrans('', '', string.punctuation)) for s in context_words]
                context_words = [s for s in context_words if s]
                context_words += ['<s>' for padding_count in range(0,num_padding)]
                # print(context_words)
                self.test_sentences[target_word][id_sentence] = context_words

        with open(candidates_path) as file:
            for line in file:
                # print(line)
                candidates =re.split('\W+',line)
                target_word = candidates.pop(0)
                candidates.pop(0)
                candidates = [s for s in candidates if s]
                self.candidates[target_word] = candidates


    def score_context_words(self):

        missing_count = 0
        target_count = 0
        cos = torch.nn.CosineSimilarity(dim=0)

        if self.model == 'skipgram':
            # Initialize model (add 2 words to vocab_size for UNK and EOS)
            skipgram = Skipgram(vocab_size=10002, emb_dimensions=300)
            # Load from training results
            skipgram.load_state_dict(torch.load('10000V_300d_5w_Skipgram.pt', map_location='cpu'))

            # Load word2idx used to train the model
            word2idx = pickle.load(open('word2idx.p', 'rb'))

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
                    print('missing target: ' + str(target_word))
                    missing_count += 1
                    continue

                candidate_embs =defaultdict()
                for candidate in self.candidates[target]:
                    if candidate in word2idx:
                        candidate_embs[candidate] = skipgram.embeddings(torch.tensor(word2idx[candidate], dtype=torch.long))
                    else:
                        candidate_embs[candidate] = torch.zeros(target_emb.shape)

                for sentence_id in self.test_sentences[target_word].keys():
                    file.write('RANKED|\t')
                    file.write(str(target_word) + '\t')
                    file.write(str(sentence_id)+'|')

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
                        file.write('\t' + str(candidate) + '\t' + str(score)+'|')
                    file.write('\n')




                    # for context_word in self.test_sentences[target_word][sentence_id]:
                    #     if context_word in word2idx:
                    #         out_emb = skipgram.out_embeddings(torch.tensor(word2idx[context_word], dtype=torch.long))
                    #         score = torch.dot(in_emb, out_emb).item()
                    #         file.write('\t' + str(context_word)+'\t'+str(score))
                    #         # print(score)
                    #     else:
                    #         # print(context_word)
                    #         # file.write(str(context_word) + ' ' + str(0) + '|\t')
                    #         out_emb = skipgram.out_embeddings(torch.tensor(word2idx['<unk>'], dtype=torch.long))
                    #         score = torch.dot(in_emb, out_emb).item()
                    #         file.write('\t' + str(context_word) + '\t' + str(score))
                    # file.write('\n')

            file.close()
            print('Target count: '+str(target_count))
            print('Missing count: '+str(missing_count))

