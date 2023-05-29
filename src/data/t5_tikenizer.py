from transformers import AutoTokenizer

# https://github.com/runnerup96/Transformers-Tuning/blob/main/t5_tokenizer.py


class T5Tokenizer:
    def __init__(self, data_config):
        self.data_config = data_config
        self.hf_tokenizer = AutoTokenizer.from_pretrained(data_config["tokenizer"])
        self.pad_token_id = self.hf_tokenizer.pad_token_id

    def __call__(self, text_list):
        return self.hf_tokenizer(
            text_list,
            padding="max_length",
            max_length=self.data_config["max_length"],
            truncation=True,
            return_token_type_ids=True,
        )

    def __len__(self):
        return len(self.hf_tokenizer)

    def add_tokens(self, tokens_list):
        self.hf_tokenizer.add_tokens(tokens_list)

    def decode(self, token_list):
        predicted_tokens = self.hf_tokenizer.decode(
            token_list,
            skip_special_tokens=True,
        )
        predicted_tokens = "".join(predicted_tokens)

        return predicted_tokens


if __name__ == "__main__":
    # test_data_ = "Токены рандомные на вход random tokens on input".split()
    test_data = ["Токены рандомные на вход", "random tokens on input"]
    data_config = {"tokenizer": "t5-small", "max_length": 10}
    tokenizer = T5Tokenizer(data_config)

    a = tokenizer(test_data)["input_ids"]
    b = tokenizer.decode(a)
    print(1)
