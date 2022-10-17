from gensim.models import FastText
from gensim.utils import tokenize
from gensim import utils
from gensim.test.utils import get_tmpfile

class MyIter:
    def __iter__(self):
        path = 'wiki_texts.txt'
        with utils.open(path, 'r', encoding='utf-8') as fin:
            for line in fin:
                yield list(tokenize(line))

model = FastText(vector_size=100, window=3, min_count=1)
model.build_vocab(sentences=MyIter())
#print('hi')
total_examples = model.corpus_count
model.train(sentences=MyIter(), total_examples=total_examples, epochs=1, workers=4, min_n=3, max_n=6)

fname = get_tmpfile("wiki_en_ft_model02.model")
model.save(fname)
#model = FastText.load(fname)