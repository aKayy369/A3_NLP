import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden


class AdditiveAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.W1 = nn.Linear(hid_dim, hid_dim)
        self.W2 = nn.Linear(hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # encoder_outputs: [src_len, batch, hid_dim]
        # hidden: [1, batch, hid_dim]

        src_len = encoder_outputs.shape[0]

        hidden = hidden.repeat(src_len, 1, 1)

        energy = torch.tanh(
            self.W1(encoder_outputs) + self.W2(hidden)
        )

        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=0)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, attention):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = attention
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim * 2, output_dim)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)

        attn = self.attention(hidden, encoder_outputs)
        attn = attn.unsqueeze(1).permute(1, 2, 0)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(attn, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)

        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc(torch.cat((output, weighted), dim=1))
        return prediction, hidden, attn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
