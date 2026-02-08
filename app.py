import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State

import torch
import pickle

from model import Encoder, Decoder, Seq2Seq, AdditiveAttention


# ---------------- Device ----------------
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ---------------- Load vocabs ----------------
with open("models/src_vocab.pkl", "rb") as f:
    src_vocab = pickle.load(f)

with open("models/tgt_vocab.pkl", "rb") as f:
    tgt_vocab = pickle.load(f)

src_token2idx = src_vocab["token2idx"]
src_idx2token = src_vocab["idx2token"]

tgt_token2idx = tgt_vocab["token2idx"]
tgt_idx2token = tgt_vocab["idx2token"]

# ---------------- Hyperparameters (MATCH TRAINING) ----------------
INPUT_DIM = len(src_token2idx)
OUTPUT_DIM = len(tgt_token2idx)
EMB_DIM = 128
HID_DIM = 256
MAX_LEN = 40

# ---------------- Load model ----------------
attention = AdditiveAttention(HID_DIM)
encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM)
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, attention)

model = Seq2Seq(encoder, decoder).to(device)
model.load_state_dict(
    torch.load("models/additive_attention_model.pt", map_location=device)
)
model.eval()

# ---------------- Translation logic ----------------
def translate(sentence):
    tokens = sentence.lower().strip().split()
    tokens = ["<SOS>"] + tokens + ["<EOS>"]

    src_ids = [
        src_token2idx.get(tok, src_token2idx["<UNK>"])
        for tok in tokens
    ]

    src_ids = src_ids[:MAX_LEN]
    src_ids += [src_token2idx["<PAD>"]] * (MAX_LEN - len(src_ids))

    src_tensor = torch.tensor(src_ids).unsqueeze(1).to(device)

    with torch.no_grad():
        enc_out, hidden = model.encoder(src_tensor)

    trg_ids = [tgt_token2idx["<SOS>"]]

    for _ in range(MAX_LEN):
        trg_tensor = torch.tensor([trg_ids[-1]]).to(device)
        with torch.no_grad():
            output, hidden, _ = model.decoder(trg_tensor, hidden, enc_out)

        pred = output.argmax(1).item()
        trg_ids.append(pred)

        if tgt_idx2token[pred] == "<EOS>":
            break

    words = [
        tgt_idx2token[i]
        for i in trg_ids[1:]
        if tgt_idx2token[i] not in ["<PAD>", "<EOS>"]
    ]

    return " ".join(words)


# ---------------- Dash App ----------------
app = dash.Dash(__name__)

app.layout = html.Div(
    style={
        "minHeight": "100vh",
        "background": "linear-gradient(135deg, #f5f7fa, #c3cfe2)",
        "paddingTop": "60px",
        "fontFamily": "Segoe UI, sans-serif"
    },
    children=[
        html.Div(
            style={
                "maxWidth": "720px",
                "margin": "auto",
                "background": "#ffffff",
                "padding": "40px",
                "borderRadius": "14px",
                "boxShadow": "0 12px 35px rgba(0,0,0,0.12)"
            },
            children=[
                html.H2(
                    "English → Nepali Machine Translation",
                    style={
                        "textAlign": "center",
                        "color": "#2c3e50",
                        "marginBottom": "8px"
                    }
                ),

                html.P(
                    "Seq2Seq Model with Additive Attention",
                    style={
                        "textAlign": "center",
                        "color": "#7f8c8d",
                        "marginBottom": "35px",
                        "fontSize": "15px"
                    }
                ),

                html.Label(
                    "Enter English sentence",
                    style={
                        "fontWeight": "600",
                        "color": "#34495e"
                    }
                ),

                dcc.Textarea(
                    id="input-text",
                    placeholder="Example: People should learn to live in harmony with nature.",
                    style={
                        "width": "100%",
                        "height": "120px",
                        "padding": "14px",
                        "fontSize": "16px",
                        "borderRadius": "8px",
                        "border": "1px solid #ccc",
                        "marginTop": "8px",
                        "resize": "none"
                    }
                ),

                html.Button(
                    "Translate",
                    id="translate-btn",
                    n_clicks=0,
                    style={
                        "width": "100%",
                        "marginTop": "22px",
                        "padding": "14px",
                        "fontSize": "16px",
                        "fontWeight": "bold",
                        "background": "#3498db",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "8px",
                        "cursor": "pointer"
                    }
                ),

                html.Div(
                    id="output-text",
                    style={
                        "marginTop": "30px",
                        "padding": "18px",
                        "background": "#f4f6f8",
                        "borderLeft": "5px solid #3498db",
                        "borderRadius": "6px",
                        "fontSize": "18px",
                        "color": "#2c3e50"
                    }
                ),

                html.Footer(
                    "AT82.05 – Natural Language Understanding | Assignment A3 | st125999",
                    style={
                        "marginTop": "35px",
                        "textAlign": "center",
                        "fontSize": "13px",
                        "color": "#95a5a6"
                    }
                )
            ]
        )
    ]
)


@app.callback(
    Output("output-text", "children"),
    Input("translate-btn", "n_clicks"),
    State("input-text", "value")
)
def translate_callback(n_clicks, text):
    if not n_clicks or not text:
        return ""
    return translate(text)


if __name__ == "__main__":
    app.run(debug=True)
