# Etape 09 Entrainement Modele

```python
# Entraînement du modèle
num_epochs = 50
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    train_loss = train_model(train_loader, model, loss_fn, optimizer)
    test_loss = evaluate_model(test_loader, model, loss_fn)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
