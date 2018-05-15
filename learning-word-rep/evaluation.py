from collections import defaultdict
import string
import torch
from skipgram import Skipgram
import pickle

class Eval:

    def __init__(self, model):
        self.model=model
        self.test_sentences = defaultdict(lambda : defaultdict(lambda : set()))

    def load_test_sentences(self, test_sentences_path):

        with open(test_sentences_path) as file:
            for line in file:
                # Split into tokens
                sentence = line.strip().split()
                target_word = sentence.pop(0)
                id_sentence = sentence.pop(0)
                target_position = sentence.pop(0)
                # Remove the target word
                sentence.pop(int(target_position))
                # Remove punctuation
                # context_words = sentence.translate(str.maketrans('', '', string.punctuation))
                context_words = [s.translate(str.maketrans('', '', string.punctuation)) for s in sentence]
                context_words = set(s for s in context_words if s)
                # print(target_word, id_sentence, context_words)
                self.test_sentences[target_word][id_sentence] = context_words
                # break

    def score_context_words(self):

        missing_count = 0
        target_count = 0

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
                    in_emb = skipgram.embeddings(torch.tensor(word2idx[target], dtype=torch.long))
                else:
                    print('missing target: ' + str(target_word))
                    missing_count += 1
                    continue

                for sentence_id in self.test_sentences[target_word].keys():
                    file.write('RANKED|\t')
                    file.write(str(target_word) + '\t')
                    file.write(str(sentence_id))

                    for context_word in self.test_sentences[target_word][sentence_id]:
                        if context_word in word2idx:
                            out_emb = skipgram.out_embeddings(torch.tensor(word2idx[context_word], dtype=torch.long))
                            score = torch.dot(in_emb, out_emb).item()
                            file.write('|\t' + str(context_word)+'\t'+str(score))
                            # print(score)
                        else:
                            # print(context_word)
                            # file.write(str(context_word) + ' ' + str(0) + '|\t')
                            out_emb = skipgram.out_embeddings(torch.tensor(word2idx['<unk>'], dtype=torch.long))
                            score = torch.dot(in_emb, out_emb).item()
                            file.write('|\t' + str(context_word) + '\t' + str(score))
                    file.write('\n')

            file.close()
            print('Target count: '+str(target_count))
            print('Missing count: '+str(missing_count))

