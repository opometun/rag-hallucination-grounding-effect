"""Binary hallucination classifier: a ModernBERT encoder with a single linear head.

The stock ModernBertForSequenceClassification head is dense, GELU, LayerNorm, dropout,
then linear. This one is just the linear layer. Mean pooling and zero classifier dropout
are already ModernBERT-base defaults, so the head is the only real difference. It is a
simpler controlled architecture, not an expected gain.
"""

from torch import nn
from transformers import AutoModel

DEFAULT_MODEL = "answerdotai/ModernBERT-base"
NUM_LABELS = 2


def masked_mean_pool(last_hidden_state, attention_mask):
    # attention_mask marks the special tokens as real, so [CLS] and [SEP] are part of the
    # mean. Deliberate, and the rule is the same in both conditions, so it does not tilt
    # the A/B comparison either way.
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    counts = mask.sum(dim=1)
    # The clamp stops an all-pad row from dividing by zero. The assert says such a row
    # should not reach here at all, so we fail instead of pooling to something meaningless
    # and carrying on. The pipeline guarantees it, this just writes the invariant down.
    assert bool((counts > 0).all()), "every row needs at least one unmasked token"
    summed = (last_hidden_state * mask).sum(dim=1)
    return summed / counts.clamp(min=1.0)


class HallucinationClassifier(nn.Module):
    def __init__(self, model_name: str = DEFAULT_MODEL):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, NUM_LABELS)

    def forward(self, input_ids, attention_mask):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = masked_mean_pool(encoded.last_hidden_state, attention_mask)
        return self.classifier(pooled)
