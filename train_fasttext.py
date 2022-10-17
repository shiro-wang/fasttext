import fasttext
model = fasttext.train_unsupervised('wiki_texts.txt', model='cbow', minn=3, maxn=6, dim=100, epoch=1, lr=0.05, thread=4)
model.save_model("wiki_en_ft_model01.bin")