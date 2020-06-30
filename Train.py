import json

from core.Trainer import ModelTrainer
from cfgload.get_cfg import get_cfg


params, paths = get_cfg("GAN\\cfg\\config.json")

Trainer = ModelTrainer(paths["data_dir"], params["im_size"], params["batch_size"],
                       paths["disc_save_path"], paths["gen_save_path"])

Trainer.train(params["epochs"], params["save_every"])