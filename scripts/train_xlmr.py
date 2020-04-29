#!/usr/bin/env python

import os
from torch import optim
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from sacred import Experiment
from tracking import expexp_observer

from model import ToxicXLMRModel
from data import train_corpus


ex = Experiment('toxicity-xlmr')
ex.observers.append(expexp_observer(os.environ["SACRED_USER"],
                                    os.environ["SACRED_PW"]))


@ex.config
def config():
    author = os.environ['USER']
    batch_size = 64
    max_seq_len = 128
    seed = 1
    optimizer = optim.AdamW
    loss = nn.CrossEntropyLoss(weight=train_corpus().class_weights())
    lr = 1e-5
    epochs = 1
    dropout_prob = 0.2


@ex.automain
def main(batch_size, optimizer, epochs,
         dropout_prob, max_seq_len, loss, lr, _run):

    model = ToxicXLMRModel(
        optimizer=optimizer,
        criterion=loss,
        lr=lr,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        dropout_prob=dropout_prob,
        log=_run.log_scalar
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), 'checkpoints'),
        save_top_k=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    trainer = Trainer(
        gpus=1,
        max_epochs=epochs,
        val_check_interval=10,
        # val_percent_check=.5,
        checkpoint_callback=checkpoint_callback
    )

    trainer.fit(model)
