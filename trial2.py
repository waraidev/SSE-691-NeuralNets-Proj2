import spacy
import torchtext
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext.datasets import Multi30k

en = spacy.load('en')
de = spacy.load('de')


def tokenize_en(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]


def tokenize_de(sentence):
    return [tok.text for tok in de.tokenizer(sentence)]


EN_TEXT = Field(tokenize=tokenize_en)
DE_TEXT = Field(tokenize=tokenize_de, init_token="<sos>", eos_token="<eos>")

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                    fields=(DE_TEXT, EN_TEXT))


