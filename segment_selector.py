# segment_selector.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "bert-base-uncased"
MODEL_PATH = "./dual_encoder_model.pt"
MAX_LEN_QUERY = 32
MAX_LEN_SEGMENT = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model tanımı
class DualEncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token

# Tokenizer ve model yükleme
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = DualEncoderModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Ana metod: sadece zaman damgalarını döndürür
def get_relevant_segments(query, segments, threshold=0.5):
    """
    query: str
    segments: list of dicts, each with 'start', 'end', 'text'
    returns: list of dicts with 'start', 'end', 'similarity'
    """
    relevant = []

    # Query encode
    query_inputs = tokenizer(query, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LEN_QUERY).to(DEVICE)
    with torch.no_grad():
        query_emb = model(query_inputs["input_ids"], query_inputs["attention_mask"])

    for seg in segments:
        seg_text = seg.get("text", "")
        seg_inputs = tokenizer(seg_text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LEN_SEGMENT).to(DEVICE)

        with torch.no_grad():
            seg_emb = model(seg_inputs["input_ids"], seg_inputs["attention_mask"])
            sim = F.cosine_similarity(query_emb, seg_emb).item()

        if sim >= threshold:
            relevant.append({
                "start": seg["start"],
                "end": seg["end"],
                "similarity": round(sim, 4)
            })

    return relevant
