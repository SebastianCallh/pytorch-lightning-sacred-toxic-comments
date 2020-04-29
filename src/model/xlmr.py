import os
import torch
import pytorch_lightning as pl
from transformers import XLMRobertaForSequenceClassification
from transformers import XLMRobertaConfig, AutoTokenizer
from data import train_loader, test_loader


class ToxicXLMRModel(pl.LightningModule):

    def __init__(self, optimizer, criterion, dropout_prob,
                 lr, batch_size, max_seq_len, log):
        super().__init__()
        self.optimier = optimizer
        self.criterion = criterion
        self.lr = lr
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.log = log
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

        config = XLMRobertaConfig()
        config.attention_probs_dropout_prob = dropout_prob
        config.hidden_dropout_prob = dropout_prob
        self.xlmr = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base')
        self.t = 0

    def forward(self, x):
        return self.xlmr(x)

    def training_step(self, batch, batch_idx):
        self.t += 1
        x, y = batch
        y_hat = self(x)[0]
        loss = self.criterion(y_hat, y)
        acc = y_hat.argmax(1).eq(y).float().mean()
        self.log('train.loss', loss.item(), self.t)
        self.log('train.accuracy', acc.item(), self.t)
        return {'loss': loss, 'accuracy': acc}

    def _tokenize(self, x):
        return self.tokenizer.encode(
            x, return_tensors='pt',
            max_length=self.max_seq_len).view(-1)

    def train_dataloader(self):
        return train_loader(
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            pad_token=0,
            tokenize=self._tokenize,
            labelize=lambda x: x)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)[0]
        loss = self.criterion(y_hat, y)
        acc = y_hat.argmax(1).eq(y).float().mean()
        return {'val_loss': loss, 'val_accuracy': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        log = {'avg_val_loss': avg_loss, 'avg_val_accuracy': avg_acc}
        self.log('val.loss', avg_loss.item(), self.t)
        self.log('val.accuracy', avg_acc.item(), self.t)

        checkpoint_path = os.path.join(os.getcwd(), 'checkpoints', f'{self.t}.pkl')
        torch.save(self.xlmr.state_dict(), checkpoint_path)

        return {'val_loss': avg_loss, 'log': log}

    def val_dataloader(self):
        return test_loader(
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            pad_token=0,
            shuffle=False,
            tokenize=self._tokenize,
            labelize=lambda x: x)

    def configure_optimizers(self):
        return self.optimier(self.parameters(), lr=self.lr)
