import json

""" configuration json """


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, "r") as f:
            config = json.loads(f.read())
            return Config(config)


config = Config(
    {
        # optimization
        "batch_size": 2,  # 48 20 2
        "learning_rate": 1e-3,  # 3
        "weight_decay": 1e-4,  # 3
        "n_epoch": 30,
        # data
        "dataset": "WIN",
        # model
        "type": "NASSBLiF",
        "svPath": "results",
        # load & save checkpoint
        "model_name": "NASSBLiF",
        "type_name": "NASSBLiF_Win5_LID",
        "ckpt_path": "./output/models/",  # directory for saving checkpoint
        "log_path": "./output/log/",
        "log_file": ".log",
        "tensorboard_path": "./output/tensorboard/",
    }
)
