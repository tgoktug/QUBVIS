import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

# 📌 **Model Parametreleri**
MAX_TOKENS = 32
FEATURE_DIM = 512  # CLIP video embedding boyutu
NUM_FRAMES = 15
GPT2_EMBED_DIM = 768  # GPT-2'nin embedding boyutu
NUM_HEADS = 8  # Encoder için Multi-Head Attention
FF_DIM = 2048  # Feedforward Katmanı Boyutu
NUM_ENCODER_LAYERS = 6  # ⚡ Daha Derin Encoder
DROPOUT_RATE = 0.4  # 🔥 Dropout oranı (Overfitting önleme)

# 📌 **CLIP Özelliklerini Daha İyi Temsil Eden Feature Projection**
class FeatureProjection(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, output_dim=GPT2_EMBED_DIM):
        super(FeatureProjection, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim * 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim * 2, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.norm(x)

# 📌 **Positional Encoding**
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super(PositionalEncoding, self).__init__()
        angles = torch.arange(seq_len).unsqueeze(1) / torch.pow(10000, (2 * torch.arange(d_model).unsqueeze(0) // 2) / d_model)
        pos_enc = torch.zeros(seq_len, d_model)
        pos_enc[:, 0::2] = torch.sin(angles[:, 0::2])
        pos_enc[:, 1::2] = torch.cos(angles[:, 1::2])
        self.register_buffer('pos_encoding', pos_enc.unsqueeze(0))

    def forward(self, x):
        return x + self.pos_encoding.to(x.device)

# 📌 **Geliştirilmiş Transformer Encoder (Self-Attention Eklendi!)**
class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        self.pos_encoding = PositionalEncoding(NUM_FRAMES, GPT2_EMBED_DIM)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=GPT2_EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=FF_DIM * 2,
            dropout=DROPOUT_RATE,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_ENCODER_LAYERS)

        # 🔹 **Ek Self-Attention Katmanı**
        self.self_attn = nn.MultiheadAttention(embed_dim=GPT2_EMBED_DIM, num_heads=NUM_HEADS, batch_first=True)
        self.norm = nn.LayerNorm(GPT2_EMBED_DIM)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        x = self.pos_encoding(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)

        # 🔹 **Self-Attention Ekle**
        self_attn_output, _ = self.self_attn(x, x, x)
        x = self.norm(x + self.dropout(self_attn_output))

        return x

# 📌 **Attention Adapter (Self-Attention ve Cross-Attention Eklendi!)**
class AttentionAdapter(nn.Module):
    def __init__(self):
        super(AttentionAdapter, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=GPT2_EMBED_DIM, num_heads=8, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=GPT2_EMBED_DIM, num_heads=8, batch_first=True)

        # 🔹 **Ek Self-Attention Katmanı**
        self.self_attn = nn.MultiheadAttention(embed_dim=GPT2_EMBED_DIM, num_heads=8, batch_first=True)

        self.norm1 = nn.LayerNorm(GPT2_EMBED_DIM)
        self.norm2 = nn.LayerNorm(GPT2_EMBED_DIM)
        self.norm3 = nn.LayerNorm(GPT2_EMBED_DIM)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, video_features):
        attn_output, _ = self.multihead_attn(video_features, video_features, video_features)
        x = self.norm1(video_features + self.dropout(attn_output))

        # 🔹 **Ek Self-Attention**
        self_attn_output, _ = self.self_attn(x, x, x)
        x = self.norm2(x + self.dropout(self_attn_output))

        # 🔹 **Cross-Attention**
        cross_attn_output, _ = self.cross_attn(x, x, x)
        return self.norm3(x + self.dropout(cross_attn_output))

# 📌 **Video Captioning Model (Özelleştirilmiş, Self-Attention Eklenmiş)**
class VideoCaptioningModel(nn.Module):
    def __init__(self):
        super(VideoCaptioningModel, self).__init__()
        self.feature_projection = FeatureProjection()
        self.encoder = TransformerEncoder()
        self.adapter = AttentionAdapter()

        config = GPT2Config.from_pretrained("gpt2", add_cross_attention=True)
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

        for param in self.gpt2.parameters():
            param.requires_grad = True

    def forward(self, video_features, captions):
        projected_features = self.feature_projection(video_features)
        encoder_output = self.encoder(projected_features)
        adapted_encoding = self.adapter(encoder_output)
        gpt2_output = self.gpt2(input_ids=captions, encoder_hidden_states=adapted_encoding, return_dict=True)
        return gpt2_output.logits

# 📌 **Modeli Oluştur**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VideoCaptioningModel().to(device)

# 📌 **Modeli Yükle**
def load_video_captioning_model(model_path):
    """
    Modeli belirtilen model_path'tan yükler.
    """
    model.load_state_dict(torch.load(model_path))
    return model

# Modeli yüklemek
model_path = "./bml-selfattn-gpt_torch_2.pth"  # Modelin dosya yolu
captioning_model = load_video_captioning_model(model_path)
print("Model başarıyla yüklendi.")
