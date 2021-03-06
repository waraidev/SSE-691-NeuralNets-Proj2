import math
import time

# torch imports
from torchtext.datasets import Multi30k
from torchtext.data import Field
import torch
import torch.optim as optim
import torch.nn as nn

# local imports
from TranslateModels import Encoder, Decoder, Attention, Seq2Seq
from TrainModel import train, evaluate, epoch_time
from FastIterator import batch_size_fn, FastIterator

# plotting import
import matplotlib.pyplot as plt

SRC = Field(tokenize="spacy",
            tokenizer_language="de",
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

TRG = Field(tokenize="spacy",
            tokenizer_language="en",
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

Multi30k.download('data')

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                    fields=[('src', SRC), ('trg', TRG)])

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

# Define an iterator: BucketIterator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 512

train_iterator = FastIterator(train_data,
                              batch_size=BATCH_SIZE,
                              device=device,
                              repeat=False,
                              sort_key=lambda x: (len(x.src), len(x.trg)),
                              batch_size_fn=batch_size_fn,
                              train=True,
                              shuffle=True)

valid_iterator = FastIterator(valid_data,
                              batch_size=BATCH_SIZE,
                              device=device,
                              repeat=False,
                              sort_key=lambda x: (len(x.src), len(x.trg)),
                              batch_size_fn=batch_size_fn,
                              train=True,
                              shuffle=True)

test_iterator = FastIterator(test_data,
                             batch_size=BATCH_SIZE,
                             device=device,
                             repeat=False,
                             sort_key=lambda x: (len(x.src), len(x.trg)),
                             batch_size_fn=batch_size_fn,
                             train=True,
                             shuffle=True)

# Model Setup
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
# ENC_EMB_DIM = 256
# DEC_EMB_DIM = 256
# ENC_HID_DIM = 512
# DEC_HID_DIM = 512
# ATTN_DIM = 64
# ENC_DROPOUT = 0.5
# DEC_DROPOUT = 0.5

ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)

optimizer = optim.SGD(model.parameters(), lr=0.1)


def count_parameters(m: nn.Module):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

# ignore padding
PAD_INDICES = TRG.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDICES)

# train the model
N_EPOCHS = 15
CLIP = 1

best_valid_loss = float('inf')

total_mins = 0
all_train_loss = []
all_valid_loss = []

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    total_mins += epoch_mins
    all_train_loss.append(train_loss)
    all_valid_loss.append(valid_loss)

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
print(total_mins)

plt.plot(all_train_loss)
plt.plot(all_valid_loss)
plt.show()
