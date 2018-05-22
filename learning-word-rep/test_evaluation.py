from evaluation import Eval

test_sentences_path = './lst/lst_test.preprocessed'
candidates_path = './lst/lst.gold.candidates'

translation_path1 = 'data/wa/test.en'
translation_path2 = 'data/wa/test.fr'

#eval = Eval(model='skipgram', window_size=5)
#eval = Eval(model='bayesian', window_size=5)
eval = Eval(model='embedalign', window_size=0)
#eval.load_test_sentences(test_sentences_path=test_sentences_path, candidates_path=candidates_path)
#eval.score_context_words()
eval.load_translation_sentences(translation_path1, translation_path2)
eval.score_aer()
