import os
from data import train_loader, train_corpus
from transformers import XLMRobertaForSequenceClassification, AutoTokenizer

xlmr = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base')
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')


def tokenize(x):
    return tokenizer.encode(x, return_tensors='pt',
                            max_length=128).view(-1)


loader = train_loader(batch_size=4,
                      max_seq_len=128,
                      pad_token=0,
                      labelize=lambda x: x,
                      tokenize=tokenize)

print(len(train_corpus().labels))
print(train_corpus().class_weights().shape)
print(train_corpus().labels[:10])
print('path is', os.path.join(os.getcwd(), 'checkpoints'))
    
for x, y in loader:
    print(x.size(), y.size())
    yhat = xlmr(x)[0]
    print(yhat.size())
    acc = yhat.argmax(1).eq(y).float().mean()
    print(acc)
    break
