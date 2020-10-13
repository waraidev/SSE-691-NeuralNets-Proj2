# torch imports
import torch
import torch.nn as nn
from torch import optim
<<<<<<< HEAD

import random
=======
from torchtext.data import BucketIterator

>>>>>>> ea3a1b6e6d2fa12160c1af22efd59ae9571bd279
from FastIterator import FastIterator


def train(model: nn.Module,
          iterator: FastIterator,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):
    model.train()

    epoch_loss = 0
    index = 0

    for _, batch in enumerate(iterator):
<<<<<<< HEAD
=======
        index += 1

>>>>>>> ea3a1b6e6d2fa12160c1af22efd59ae9571bd279
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

<<<<<<< HEAD
        index += 1

        if index % 200 == 0:
            print("Run " + str(index) + " of iterator")
=======
        print('hi' + str(index))
>>>>>>> ea3a1b6e6d2fa12160c1af22efd59ae9571bd279

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: FastIterator,
             criterion: nn.Module):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for _, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)  # turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
