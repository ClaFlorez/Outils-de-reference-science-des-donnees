# Etape 08 Fonction Evaluation

```python
# Fonction d'évaluation
def evaluate_model(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            total_loss += loss.item()
    return total_loss / len(dataloader)
