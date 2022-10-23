from base64 import encode
import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, nchem_feature: int, dropout: float = 0.5, linear_dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.lnorm = nn.LayerNorm(d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model+nchem_feature, d_model),
            nn.LeakyReLU(),
            nn.Dropout(linear_dropout),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Dropout(linear_dropout),
            nn.Linear(d_model, 1),
        )
        self.nchem_feature = nchem_feature

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.encoder.weight)
        for m in self.transformer_encoder.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
        self.lnorm.bias.data.zero_()
        self.lnorm.weight.data.fill_(1.0)
        for m in self.decoder:
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
                nn.init.kaiming_uniform_(m.weight)

    # from min-GPT
    def configure_optimizer(self, lr: float, weight_decay: float) -> torch.optim.Optimizer:
        decay = set()
        no_decay = set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                # decay only weights
                if not 'norm' in fpn and 'weight' in pn:
                    decay.add(fpn)
                else:
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.98), eps=1e-9)
        return optimizer

    def forward(self, src: Tensor, src_pad_mask: Tensor, chem_value: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_pad_mask: Tensor, shape [batch_size, seq_len]

        Returns:
            output Tensor of shape [batch_size, 1]
        """
        src = src.transpose(0, 1)
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_pad_mask)[0]
        output = self.lnorm(output)
        output = torch.cat([output, chem_value], dim=-1)
        output = self.decoder(output)
        return output
