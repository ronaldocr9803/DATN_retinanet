import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

from model import RetinaNetModel
from omegaconf import OmegaConf

#load in the hparams.ymal file using Omegaconf
hparams = OmegaConf.load("./hparams.yaml")

lr_logger  = LearningRateMonitor(logging_interval="step")
early_stop = EarlyStopping(mode="min", monitor="val_loss", patience=8, )

#instantiate LightningTrainer
trainer    = Trainer(precision=16, gpus=1, callbacks=[lr_logger, early_stop], max_epochs=50, weights_summary="full", )

# import ipdb;ipdb.set_trace()
litModel = RetinaNetModel(hparams)
trainer.fit(litModel)