from evaluation import Eval

test_sentences_path = './lst/lst_test.preprocessed'

eval = Eval(model='skipgram')
eval.load_test_sentences(test_sentences_path=test_sentences_path)
eval.score_context_words()