from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing


class BPETokenizer:
    def __init__(self, sentence_list, pad_flag, vocab_size, min_freq, max_seq_len):
        """
        sentence_list - список предложений для обучения
        """
        self.special_tokens = ["[BOS]", "[EOS]", "[UNK]", "[PAD]"]
        self.pad_flag = pad_flag
        self.max_seq_len = max_seq_len

        self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tokenizer.decoder = decoders.BPEDecoder(" ")

        # Надо вынести в конфиг, либо в константы (когда-то)
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            special_tokens=self.special_tokens,
            show_progress=False,
        )
        self.tokenizer.train_from_iterator(sentence_list, trainer=trainer)

        self.tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[
                ("[BOS]", self.tokenizer.token_to_id("[BOS]")),
                ("[EOS]", self.tokenizer.token_to_id("[EOS]")),
            ],
        )

        self.special_id = [self.tokenizer.token_to_id(x) for x in self.special_tokens]

    def pad_sent(self, token_ids_list):
        if len(token_ids_list) < self.max_seq_len:
            padded_token_ids_list = token_ids_list + [
                self.tokenizer.token_to_id("[PAD]")
            ] * (self.max_seq_len - len(token_ids_list))
        else:
            padded_token_ids_list = token_ids_list[: self.max_seq_len - 1] + [
                self.tokenizer.token_to_id("[EOS]")
            ]
        return padded_token_ids_list

    def __len__(self):
        return len(self.tokenizer.get_vocab())

    def __call__(self, sentence):
        """
        sentence - входное предложение
        """
        tokenized_data = self.tokenizer.encode(sentence).ids
        if self.pad_flag and self.max_seq_len:
            tokenized_data = self.pad_sent(tokenized_data)

        return tokenized_data

    def decode(self, token_list):
        """
        token_list - предсказанные ID вашего токенизатора
        """
        decoded_sentence = "".join(self.tokenizer.decode(token_list))

        return decoded_sentence
