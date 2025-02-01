# Etape 10 Visualisation Performances

```python
# Visualisation des pertes
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), train_losses, label="Train Loss")
plt.plot(range(num_epochs), test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training and Testing")
plt.legend()
plt.show()
