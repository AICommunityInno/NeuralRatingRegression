import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from itertools import chain

from .constants import USER_LF_NUM, ITEM_LF_NUM, HIDDEN_DIM, CONTEXT_SIZE
from .constants import RR_HIDDEN_LAYERS, TG_HIDDEN_LAYERS, MAX_LENGTH

# For GPU
if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class Merger(nn.Module):
    def __init__(self, hidden_size, user_latent_factors_count, item_latent_factors_count):
        super(Merger, self).__init__()
        self.user_model = nn.Linear(user_latent_factors_count, hidden_size, bias=False)
        self.item_model = nn.Linear(item_latent_factors_count, hidden_size, bias=True)

    def forward(self, user_embed, item_embed):
        return nn.Sigmoid()(self.user_model(user_embed) + self.item_model(item_embed))


class ContextMerger(nn.Module):
    def __init__(self, vocabulary_size, context_size,
                 user_latent_factors_count,
                 item_latent_factors_count):
        super(ContextMerger, self).__init__()
        self.user_model = nn.Linear(user_latent_factors_count, context_size, bias=False)
        self.item_model = nn.Linear(item_latent_factors_count, context_size, bias=False)
        self.rating_weight = nn.Parameter(device.FloatTensor(1))
        self.review_model = nn.Linear(vocabulary_size, context_size, bias=True)

    def forward(self, user_embed, item_embed, rating, review):
        return nn.Tanh()(
            self.user_model(user_embed) + self.item_model(item_embed) +
            self.rating_weight * rating + self.review_model(review)
        )


class EncoderModel(nn.Module):
    def __init__(self, users_count, items_count,
                 user_latent_factors_count,
                 item_latent_factors_count,
                 vocabulary_size,
                 context_size, hidden_size,
                 n_regression_layers,
                 n_review_layers):
        super(EncoderModel, self).__init__()

        self.user_embedding = nn.Embedding(users_count, user_latent_factors_count)
        self.item_embedding = nn.Embedding(items_count, item_latent_factors_count)

        self.merger = Merger(hidden_size,
                             user_latent_factors_count,
                             item_latent_factors_count)

        self.regression_model = nn.Sequential(
            *(list(chain.from_iterable([
                [nn.Linear(hidden_size, hidden_size), nn.Sigmoid()]
                for _ in range(n_regression_layers - 1)])) +
              [nn.Linear(hidden_size, hidden_size), nn.Linear(hidden_size, 1)])
        )
        self.review_model = nn.Sequential(
            *(list(chain.from_iterable([
                [nn.Linear(hidden_size, hidden_size), nn.Sigmoid()]
                for _ in range(n_review_layers - 1)])) +
              [nn.Linear(hidden_size, vocabulary_size)])
        )

        self.context_merger = ContextMerger(vocabulary_size, context_size,
                                            user_latent_factors_count,
                                            item_latent_factors_count)

    def forward(self, input_user, input_item):
        embedded_user = self.user_embedding(input_user)
        embedded_item = self.item_embedding(input_item)

        merged = self.merger(embedded_user, embedded_item)
        regression_result = self.regression_model(merged)
        review_result = self.review_model(merged)
        review_softmax = nn.LogSoftmax()(review_result)

        context = self.context_merger(embedded_user, embedded_item, regression_result, review_result)
        return regression_result, review_softmax, context


class DecoderModel(nn.Module):
    def __init__(self, context_size, vocabulary_size):
        super(DecoderModel, self).__init__()

        self.embedding = nn.Embedding(vocabulary_size, context_size)
        self.gru = nn.GRU(context_size, context_size)
        self.out = nn.Linear(context_size, vocabulary_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, output, hidden):
        output = torch.transpose(self.embedding(output), 0, 1)
        hidden = hidden.view(1, hidden.size()[0], -1)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(torch.transpose(self.out(output), 2, 0))
        output = torch.transpose(output, 0, 1)
        output = torch.transpose(output, 1, 2)
        return output, hidden


class Model(nn.Module):

    def __init__(self, users_count,  # defaults are for Electronics Dataset
                 items_count,
                 vocabulary_size,
                 user_latent_factors_count=USER_LF_NUM,
                 item_latent_factors_count=ITEM_LF_NUM,
                 context_size=CONTEXT_SIZE,
                 hidden_size=HIDDEN_DIM,
                 n_regression_layers=RR_HIDDEN_LAYERS,
                 n_review_layers=TG_HIDDEN_LAYERS,
                 max_tip_len=MAX_LENGTH + 2):
        super(Model, self).__init__()
        self.SEQ_START_ID = 2
        self.encoder = EncoderModel(users_count=users_count,
                                    items_count=items_count,
                                    user_latent_factors_count=user_latent_factors_count,
                                    item_latent_factors_count=item_latent_factors_count,
                                    vocabulary_size=vocabulary_size,
                                    context_size=context_size,
                                    hidden_size=hidden_size,
                                    n_regression_layers=n_regression_layers,
                                    n_review_layers=n_review_layers)

        self.decoder = DecoderModel(context_size=context_size,
                                    vocabulary_size=vocabulary_size)

        self.max_tip_len = max_tip_len
        self.empty_output = [[self.SEQ_START_ID] * self.max_tip_len]
        self.vocabulary_size = vocabulary_size

    def forward(self, input_user, input_item):
        regression_result, review_softmax, context = self.encoder.forward(input_user, input_item)
        output_tip_probs = Variable(device.LongTensor(self.empty_output * len(input_user)))
        output, hidden = self.decoder.forward(output_tip_probs, context)
        return regression_result, review_softmax, output

    def voc_size(self):
        return self.vocabulary_size


def review_loss(c_hat, c):
    assert c_hat.size() == c.size(), '{} != {}'.format(c_hat.size(), c.size())
    return torch.mul(c_hat, c.float()).sum()
