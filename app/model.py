
import torch.nn as nn
from transformers import AutoModel


class MultiTaskParsBERT(nn.Module):

    def __init__(
        self,
        model_name,
        num_recommendation_classes,
        num_rating_classes,
        dropout_rate=0.3
    ):
        super().__init__()

        # Load the pretrained ParsBERT encoder
        self.encoder = AutoModel.from_pretrained(model_name)

        # Get the encoder hidden size
        hidden_size = self.encoder.config.hidden_size

        # Shared dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Recommendation classification head
        self.recommendation_classifier = nn.Linear(
            hidden_size,
            num_recommendation_classes
        )

        # Rating classification head
        self.rating_classifier = nn.Linear(
            hidden_size,
            num_rating_classes
        )

    def forward(self, input_ids, attention_mask):

        # Extract contextual representations from ParsBERT
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use the representation of the [CLS] token
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Generate predictions for both tasks
        recommendation_logits = self.recommendation_classifier(
            pooled_output
        )

        rating_logits = self.rating_classifier(
            pooled_output
        )

        return {
            "recommendation_logits": recommendation_logits,
            "rating_logits": rating_logits
        }
