import torch
import gc
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from .model import review_loss
from .constants import REG_WEIGHT


class Trainer(object):

    def __init__(self, model):
        self.model = model

        self.loss_criterion = (
            lambda r, r_hat, c, c_hat, s, s_hat:
            nn.MSELoss()(r_hat, r) + review_loss(c_hat, c) + nn.NLLLoss()(s_hat, s)
        )
        self.optimizer = optim.Adadelta(model.parameters(),
                                        weight_decay=REG_WEIGHT)  # L2 regularisation is included here

    def train(self, train_iter, n_epochs):
        losses = []

        for epoch_i in range(n_epochs):
            for batch in tqdm(train_iter,
                              desc="Epochs: {} / {}, Loss: {}".format(epoch_i, n_epochs,
                                                                      losses[-1] if len(losses) > 0 else np.inf)):
                users_batch = batch.user
                items_batch = batch.item
                ratings_batch = batch.rating
                reviews_batch = batch.text
                tips_batch = torch.transpose(batch.tips, 0, 1)

                regression_result, review_softmax, tips_output = self.model.forward(users_batch, items_batch)
                self.optimizer.zero_grad()

                loss = self.loss_criterion(ratings_batch, regression_result,
                                           reviews_batch, review_softmax,
                                           tips_batch.contiguous().view(-1),
                                           tips_output.contiguous().view(-1, self.model.voc_size()))
                losses.append(loss.data.cpu().numpy())

                loss.backward()
                self.optimizer.step()

                gc.collect()
        return losses
