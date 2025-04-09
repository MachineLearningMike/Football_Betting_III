
import tensorflow as tf
import numpy as np

class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()   # So the default -1 axix is normalized across. No inter-token operatoin.
    self.add = tf.keras.layers.Add()

class CrossAttention(BaseAttention):
  def call(self, x, context, mask):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        attention_mask=mask,
        return_attention_scores=True)
   
    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
  
class GlobalSelfAttention(BaseAttention): 
  def call(self, x, mask):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        attention_mask=mask)    # intentional inter-token operation
    x = self.add([x, attn_output])  # token-wise
    x = self.layernorm(x)         # normalize across the default -1 axis. No inter-token operatoin.
    return x
  
class CausalSelfAttention(BaseAttention): # mask-agnostic
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True)     # look-over mask is generagted and used, in decoder layers
    x = self.add([x, attn_output])  # mask-agnostic
    x = self.layernorm(x)  # normalize across the default -1 axis. No inter-token operatoin.
    return x
  
class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),    # across -1 axis
      tf.keras.layers.Dense(d_model),    # across -1 axis
      tf.keras.layers.Dropout(dropout_rate)    # mask-agnostic
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])  # mask-agnostic
    x = self.layer_norm(x)  # normalize across the default -1 axis. No inter-token operatoin.
    return x
  
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x, mask):
    x = self.self_attention(x, mask)
    x = self.ffn(x)
    return x
  
class Encoder(tf.keras.layers.Layer):
  def __init__(self, hyperparams, dropout_rate=0.1):
    super().__init__()

    self.d_model = hyperparams.d_model
    self.num_layers = hyperparams.num_layers

    self.pos_embedding = PositionalEmbedding(hyperparams)

    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.enc_layers = [
        EncoderLayer(d_model=hyperparams.d_model,
                     num_heads=hyperparams.num_heads,
                     dff=hyperparams.d_model * 4,
                     dropout_rate=dropout_rate)
        for _ in range(hyperparams.num_layers)]

def call(self, x):
  x, mask = self.pos_embedding(x)  # output Shape `(batch_size, seq_len, d_model)`.

  x = self.dropout(x)
  for encoder_layer in self.enc_layers:
    x = encoder_layer(x, mask)
  return x  # Shape `(batch_size, seq_len, d_model)`.
  
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)
    
    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

def call(self, x, context, cross_attention_mask):
  x = self.causal_self_attention(x=x)
  x = self.cross_attention(x, context, cross_attention_mask)

  # Cache the last attention scores for plotting later
  self.last_attn_scores = self.cross_attention.last_attn_scores

  x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
  return x
  
class Decoder(tf.keras.layers.Layer):
  def __init__(self, hyperparams, dropout_rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = hyperparams.d_model
    self.num_layers = hyperparams.num_layers

    self.pos_embedding = PositionalEmbedding(hyperparams, isEncoder=False)

    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(d_model=hyperparams.d_model, num_heads=hyperparams.num_heads,
                     dff=hyperparams.d_model * 4, dropout_rate=dropout_rate)
        for _ in range(hyperparams.num_layers)]

    self.last_attn_scores = None

def call(self, x, context):
  # `x` is token-IDs shape (batch, target_seq_len)
  x, cross_attention_mask = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)
  # print('decoder input x, cross_attention_mask', x.shape, cross_attention_mask.shape)
  
  x = self.dropout(x)
  for decoder_layer in self.dec_layers:
    x  = decoder_layer(x, context, cross_attention_mask)
  self.last_attn_scores = self.dec_layers[-1].last_attn_scores

  # The shape of x is (batch_size, target_seq_len, d_model).
  return x
  
class Transformer(tf.keras.Model):
  def __init__(self, hyperparams, dropout_rate=0.1):
    super().__init__()
    self.encoder = Encoder(hyperparams, dropout_rate=dropout_rate)

    self.decoder = Decoder(hyperparams, dropout_rate=dropout_rate)

    self.final_layer = tf.keras.layers.Dense(hyperparams.d_model) #-------------- to modify

  def call(self, inputs):
    x = self.encoder(inputs)  # (batch_size, context_len, d_model)
    x = self.decoder(inputs, x)  # (batch_size, target_len, d_model)
    logits = self.final_layer(x)  # (batch_size, target_len=1, hyperparams.d_model)
    logits = tf.squeeze(logits, axis=-2)  # (batch, hyperparams.d_model)

    return logits
  
