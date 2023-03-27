# Note that these are not the optimized hyperparameters from the search
# These are hyperparameters that would commonly be found in literature

# Train from scratch
TFS_CONF = {
    "opt_fn": "adam",
    "T": 100,
    "train_batch_size": 16,
    "test_batch_size": 4,
    "lr": 0.001,
    "momentum": 0.9
}

# Fine Tuning
FT_CONF = {
    "opt_fn": "adam",
    "T": 100,
    "train_batch_size": 16,
    "test_batch_size": 4,
    "lr": 0.001,
    "momentum": 0.9
}

# Proto Fine Tuning
PROTO_FT_CONF = {
    "opt_fn": "adam",
    "T": 100,
    "train_batch_size": 16,
    "test_batch_size": 4,
    "lr": 0.001,
    "momentum": 0.9,
    "learn_lambda": False,
    "init_lambda": 50.0,
    "lambda_base": 1
}

# Model-agnostic meta-learning
MAML_CONF = {
    "opt_fn": "adam",
    "T": 5, 
    "lr": 0.001,
    "momentum": 0.9,
    "base_lr": 0.01,
    "meta_batch_size": 1
}

# Prototypical model-agnostic meta-learning
PROTO_MAML_CONF = {
    "opt_fn": "adam",
    "T": 5,
    "lr": 0.001,
    "momentum": 0.9,
    "base_lr": 0.01,
    "meta_batch_size": 1,
    "learn_lambda": False,
    "init_lambda": 50.0,
    "lambda_base": 1
}

# MetaCurvature
METACURVATURE_CONF = {
    "opt_fn": "adam",
    "T": 5,
    "lr": 0.001,
    "momentum": 0.9,
    "base_lr": 0.01,
    "meta_batch_size": 1
}

# Prototypical networks
PROTO_CONF = {
    "opt_fn": "adam",
    "lr": 0.001,
    "momentum": 0.9,
    "meta_batch_size":1,
    "T": 1,
    "dist_temperature": 0.5
}

# Matching networks
MATCHING_CONF = {
    "opt_fn": "adam",
    "lr": 0.001,
    "momentum": 0.9,
    "meta_batch_size":1,
    "T": 1
}

# DDRR
DDRR_CONF = {
    "opt_fn": "adam",
    "lr": 0.0003,
    "momentum": 0.9,
    "meta_batch_size":1,
    "T": 1,
    "learn_lambda": False,
    "init_lambda": 50.0,
    "lambda_base": 1,
    "init_adj_scale": 5.0,
    "adj_base": 1
}