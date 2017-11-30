# *** Model parameters *** #
USER_LF_NUM = 300  # Number of user latent factors
ITEM_LF_NUM = 300  # Number of item latent factors
WORD_LF_NUM = 300  # Number of word latent factors
HIDDEN_DIM = 400  # Dimension of hidden space
CONTEXT_SIZE = 400  # Number of dimensions for the context vector
RR_HIDDEN_LAYERS = 4  # Number of hidden layers in rating regression
TG_HIDDEN_LAYERS = 1  # Number of hidden layers in tips generation

# *** Algorithm parameters *** #
BEAM_SIZE = 4  # Beam size for beam algorithm
MAX_LENGTH = 20  # Max length of tips

# *** Loss parameters *** #
RR_LOSS_WEIGHT = 1  # Weight of rating regression loss in objective function
WG_LOSS_WEIGHT = 1  # Weight of word generation loss in objective function
TG_LOSS_WEIGHT = 1  # Weight of tips generation loss in objective function
REG_WEIGHT = 0.0001  # Regularisation term weight

# *** Data split parameters *** #
TRAIN_SPLIT = 0.8  # Proportion of train data
TEST_SPLIT = 0.2  # Proportion of test data
