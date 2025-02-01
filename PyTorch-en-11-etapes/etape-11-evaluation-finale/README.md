# Etape 11 Evaluation Finale

```python
# Évaluation finale
model.eval()
X_sample = torch.tensor(X_test[:5], dtype=torch.float32)
y_sample = y_test[:5]

predictions = model(X_sample).detach().numpy()
for i, (real, pred) in enumerate(zip(y_sample, predictions)):
    print(f"Sample {i + 1}: Real Value = {real:.2f}, Predicted Value = {pred[0]:.2f}")
```
