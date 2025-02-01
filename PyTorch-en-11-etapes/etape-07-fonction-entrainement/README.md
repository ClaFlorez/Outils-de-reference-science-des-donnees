# Etape 07 Fonction Entrainement

```python
# Fonction d'entraînement
def train_model(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
