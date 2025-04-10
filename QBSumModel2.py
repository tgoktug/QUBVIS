import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, BatchNormalization, Dropout
import numpy as np


# 📌 Transformer Encoder Layer
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.05):
        super(TransformerEncoder, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm1 = BatchNormalization()
        self.norm2 = BatchNormalization()
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim)])
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output, training=training)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output, training=training)


# 📌 Cross-Attention Layer
class CrossAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.05):
        super(CrossAttentionLayer, self).__init__()
        self.cross_attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm = BatchNormalization()
        self.dropout = Dropout(dropout_rate)

    def call(self, video_features, query_features, training=False):
        query_expanded = tf.expand_dims(query_features, axis=1)
        query_tiled = tf.tile(query_expanded, [1, 512, 1])  # Broadcast query across all video frames
        attn_output = self.cross_attention(query_tiled, video_features)
        attn_output = self.dropout(attn_output, training=training)
        return self.norm(video_features + attn_output, training=training)


# 📌 Model Definition
def create_video_summary_model(embed_dim=512, num_heads=8, ff_dim=1024):
    # Input layers
    video_input = Input(shape=(512, 512), name="video_features")  # (512, 512) video CLIP features
    query_input = Input(shape=(512,), name="query_embedding")  # (512,) query CLIP feature

    # 📌 Transformer Encoder for Video Features
    video_encoded = TransformerEncoder(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)(video_input)

    # 📌 Query Encoder (Weighted Query)
    query_encoded = query_input

    # 📌 Video & Query Fusion with Cross-Attention
    cross_attention_output = CrossAttentionLayer(embed_dim=embed_dim, num_heads=num_heads)(video_encoded, query_encoded)

    # 📌 Decoder using Transformer Encoder
    decoder_output = TransformerEncoder(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)(cross_attention_output)

    # 📌 Output layer (Binary Classification for frame selection)
    output = Dense(1, activation="sigmoid")(decoder_output)

    # Create the model
    model = tf.keras.Model(inputs=[video_input, query_input], outputs=output, name="QueryBasedVideoSummarization")
    return model


# 📌 Model Loading Function
def load_video_summary_model(weights_path=None):
    model = create_video_summary_model()
    if weights_path:
        model.load_weights(weights_path)  # Load pre-trained weights if provided
    return model