import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class Metrics:
    def __init__(self):
        self.y_true = []
        self.y_pred = []
        self.loss = []

    def update(self, batch_y_pred, batch_y_true, batch_loss):
        """Update the state with predictions and true labels."""
        self.y_pred.extend(batch_y_pred)
        self.y_true.extend(batch_y_true)
        self.loss.append(batch_loss)

    def compute(self):
        """Compute metrics based on the current state."""
        y_pred = np.array(self.y_pred).flatten()
        y_true = np.array(self.y_true).flatten()

        loss = np.mean(self.loss)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        return {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    def reset(self):
        """Reset the internal state."""
        self.y_true = []
        self.y_pred = []
        self.loss = []
