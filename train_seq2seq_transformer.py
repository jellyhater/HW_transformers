import torch
import yaml

from src.data import datamodule
from src.models import seq2seq_transformer, trainer
from src.txt_logger import TXTLogger

if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    data_config = yaml.load(open("configs/data_config.yaml", "r"), Loader=yaml.Loader)
    dm = datamodule.DataManager(data_config, DEVICE)
    train_dataloader, dev_dataloader = dm.prepare_data()

    model_config = yaml.load(open("configs/model_config.yaml", "r"), Loader=yaml.Loader)

    # TODO: Инициализируйте модель Seq2SeqTransformer
    model = seq2seq_transformer.Seq2SeqTransformer(
        device=DEVICE,
        emb_size=model_config["emb_size"],
        vocab_size=len(dm.target_tokenizer),
        nhead=model_config["nhead"],
        num_layers=model_config["num_layers"],
        dim_feedforward=model_config["dim_feedforward"],
        target_tokenizer=dm.target_tokenizer,
        max_seq_length=data_config["max_length"] * 10,
    )

    logger = TXTLogger("training_logs")
    trainer_cls = trainer.Trainer(model=model, model_config=model_config, logger=logger)

    if model_config["try_one_batch"]:
        train_dataloader = [list(train_dataloader)[0]]
        dev_dataloader = [list(train_dataloader)[0]]

    trainer_cls.train(train_dataloader, dev_dataloader)
