import torch
import warnings
import os

from src.model import Model
from src.trainer import Trainer
from src.data_reader import amazon_dataset_iters

warnings.filterwarnings('ignore')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')

# Parameters
epochs_count = 20
dataset_folder = './data/reviews_Electronics_5/'
model_weights_file = 'model_electronics.trch'


# Loading the dataset
text_vocab, tips_vocab, train_iter, val_iter, test_iter = (
    amazon_dataset_iters(dataset_folder, device=None)
)

items_count = int(max([i.item.max().cpu().data.numpy() for i in train_iter] +
                      [i.item.max().cpu().data.numpy() for i in test_iter])[0])
users_count = int(max([i.user.max().cpu().data.numpy() for i in train_iter] +
                      [i.user.max().cpu().data.numpy() for i in test_iter])[0])

# Creating the model
model = Model(vocabulary_size=len(text_vocab.itos),
              items_count=items_count+2,
              users_count=users_count+2).cuda()

# Training the model
trainer = Trainer(model)
trainer.train(train_iter, n_epochs=epochs_count)

# Saving the model state
torch.save(model.state_dict(), model_weights_file)
