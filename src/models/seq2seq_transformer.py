import math

import torch
import torch.nn as nn

import src.metrics as metrics
from src.models.positional_encoding import PositionalEncoding


class Seq2SeqTransformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_size,
        nhead,
        num_layers,
        dim_feedforward,
        device,
        target_tokenizer,
        max_seq_length=300,
        lr_sched_step_every=1000,
        lr_sched_gamma=0.1,
    ):
        super(Seq2SeqTransformer, self).__init__()
        # TODO: Реализуйте конструктор seq2seq трансформера - матрица эмбеддингов, позиционные эмбеддинги, encoder/decoder трансформер, vocab projection head
        self.device = device
        self.emb_size = emb_size
        self.transformer = torch.nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
        )
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_encoder = PositionalEncoding(emb_size, max_seq_length)
        self.vocab_projection = nn.Linear(emb_size, vocab_size)
        self.src_mask = None
        self.trg_mask = None
        self.emb_size = emb_size
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=lr_sched_step_every,
            gamma=lr_sched_gamma,
        )
        self.target_tokenizer = target_tokenizer

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        ).to(self.device)
        return mask

    def forward(self, src, trg):
        # TODO: Реализуйте forward pass для модели, при необходимости реализуйте другие функции для обучения
        if self.src_mask is None or self.src_mask.size(0) != src.size(1):
            self.src_mask = self.generate_square_subsequent_mask(src.size(1)).to(
                self.device
            )

        if self.trg_mask is None or self.trg_mask.size(0) != trg.size(1):
            self.trg_mask = self.generate_square_subsequent_mask(trg.size(1)).to(
                self.device
            )

        src = self.embedding(src).long() * math.sqrt(self.emb_size)
        src = self.pos_encoder(src)
        trg = self.embedding(trg).long() * math.sqrt(self.emb_size)
        trg = self.pos_encoder(trg)
        # Permute dimensions from (Batch, SeqLen, EmbeddingDim) to (SeqLen, Batch, EmbeddingDim)
        src = src.permute(1, 0, 2)
        trg = trg.permute(1, 0, 2)

        output = self.transformer(
            src,
            trg,
            src_mask=self.src_mask,
            tgt_mask=self.trg_mask,
        )
        output = self.vocab_projection(output).permute(1, 0, 2)
        # target_vocab_distribution = nn.functional.softmax(output)
        topi = torch.argmax(output, dim=-1)

        return topi.clone(), output

    def training_step(self, batch):
        # TODO: Реализуйте обучение на 1 батче данных по примеру seq2seq_rnn.py
        self.optimizer.zero_grad()
        input_tensor, target_tensor = batch
        (_, output) = self.forward(input_tensor, target_tensor[:, :-1])
        target = target_tensor[:, 1:].reshape(-1)
        output = output.reshape(-1, output.shape[-1])
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()
        return loss.item()

    def predict(self, src):
        self.eval()

        with torch.no_grad():
            src = src.to(self.device)
            trg_input = torch.tensor(
                [[self.target_tokenizer.tokenizer.token_to_id("[BOS]")]],
                device=self.device,
            )
            output = []

            while (
                trg_input[:, -1].item()
                != self.target_tokenizer.tokenizer.token_to_id("[EOS]")
                and len(output) < 34
            ):
                trg_input = trg_input.to(self.device)
                _, output_step = self.forward(src, trg_input)
                pred_token = torch.argmax(output_step, dim=-1)[:, -1]
                output.append(pred_token.item())

                trg_input = torch.cat((trg_input, pred_token.unsqueeze(1)), dim=-1)

            predicted_ids = torch.tensor(output, device=self.device).unsqueeze(0)

        self.train()
        return predicted_ids

    def validation_step(self, batch):
        # TODO: Реализуйте оценку на 1 батче данных по примеру seq2seq_rnn.py
        input_tensor, target_tensor = batch
        (_, output) = self.forward(input_tensor, target_tensor[:, :-1])
        target = target_tensor[:, 1:].reshape(-1)
        output = output.reshape(-1, output.shape[-1])
        loss = self.criterion(output, target)
        return loss.item()

    def eval_bleu(self, predicted_ids_list, target_tensor):
        predicted = predicted_ids_list.clone()
        predicted = predicted.squeeze().detach().cpu().numpy()
        actuals = target_tensor.squeeze().detach().cpu().numpy()
        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=self.target_tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences
