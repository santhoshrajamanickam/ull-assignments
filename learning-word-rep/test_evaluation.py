from evaluation import Eval

test_sentences_path = './lst/lst_test.preprocessed'
candidates_path = './lst/lst.gold.candidates'

eval = Eval(model='skipgram',window_size=5)
eval.load_test_sentences(test_sentences_path=test_sentences_path, candidates_path=candidates_path)
eval.score_context_words()