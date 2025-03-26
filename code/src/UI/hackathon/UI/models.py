from django.db import models
import uuid  # To generate unique IDs

class Transaction(models.Model):
    transaction_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)  # Auto-generated UUID
    sender = models.CharField(max_length=255, blank=True, null=True)  # Some data may be unstructured
    receiver = models.CharField(max_length=255, blank=True, null=True)
    entity_type = models.JSONField(default=list)  # Store as a JSON array
    risk_score = models.FloatField(default=0.0)
    supporting_evidence = models.JSONField(default=list)  # Store as a JSON array
    confidence_score = models.FloatField(default=0.0)
    reason = models.TextField(blank=True, null=True)
    full_text = models.TextField()  # Store the raw transaction data

    def __str__(self):
        return f"{self.transaction_id} | {self.sender} → {self.receiver}"
