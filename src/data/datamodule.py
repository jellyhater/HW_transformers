from torch.utils.data import DataLoader

from src.data.bpe_tokenizer import BPETokenizer
from src.data.mt_dataset import MTDataset
from src.data.t5_tikenizer import T5Tokenizer
from src.data.utils import TextUtils, short_text_filter_function


class DataManager:
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.input_lang_n_words = None
        self.output_lang_n_words = None
        self.device = device

    def prepare_data(self):
        pairs = TextUtils.read_langs_pairs_from_file(filename=self.config["filename"])
        prefix_filter = self.config["prefix_filter"]
        if prefix_filter:
            prefix_filter = tuple(prefix_filter)

        source_sentences, target_sentences = [], []
        # dataset is ambiguous -> i lied -> я солгал/я соврала
        unique_sources = set()
        for pair in pairs:
            source, target = pair[0], pair[1]
            if (
                short_text_filter_function(
                    pair, self.config["max_length"], prefix_filter
                )
                and source not in unique_sources
            ):
                source_sentences.append(source)
                target_sentences.append(target)
                unique_sources.add(source)

        train_size = int(len(source_sentences) * self.config["train_size"])
        source_train_sentences, source_val_sentences = (
            source_sentences[:train_size],
            source_sentences[train_size:],
        )
        target_train_sentences, target_val_sentences = (
            target_sentences[:train_size],
            target_sentences[train_size:],
        )

        # TODO: Замените на BPE токенизатор
        self.source_tokenizer = BPETokenizer(
            source_train_sentences + target_train_sentences,
            pad_flag=True,
            vocab_size=self.config["vocab_size"],
            min_freq=self.config["min_freq"],
            max_seq_len=self.config["max_length"],
        )
        self.global_tokenizer = T5Tokenizer(self.config)

        self.global_tokenizer.add_tokens(
            list(self.source_tokenizer.tokenizer.get_vocab().keys())
        )
        tokenized_source_train_sentences = [
            self.global_tokenizer(s) for s in source_train_sentences
        ]
        tokenized_source_val_sentences = [
            self.global_tokenizer(s) for s in source_val_sentences
        ]

        # TODO: Замените на BPE токенизатор
        # self.target_tokenizer = BPETokenizer(
        #     target_train_sentences,
        #     pad_flag=True,
        #     vocab_size=self.config["vocab_size"],
        #     min_freq=self.config["min_freq"],
        #     max_seq_len=self.config["max_length"],
        # )
        tokenized_target_train_sentences = [
            self.global_tokenizer(s) for s in target_train_sentences
        ]
        tokenized_target_val_sentences = [
            self.global_tokenizer(s) for s in target_val_sentences
        ]

        train_dataset = MTDataset(
            tokenized_source_list=tokenized_source_train_sentences,
            tokenized_target_list=tokenized_target_train_sentences,
            dev=self.device,
        )

        val_dataset = MTDataset(
            tokenized_source_list=tokenized_source_val_sentences,
            tokenized_target_list=tokenized_target_val_sentences,
            dev=self.device,
        )

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.config["batch_size"],
        )

        val_dataloader = DataLoader(
            val_dataset,
            shuffle=True,
            batch_size=self.config["batch_size"],
            drop_last=True,
        )
        return train_dataloader, val_dataloader
