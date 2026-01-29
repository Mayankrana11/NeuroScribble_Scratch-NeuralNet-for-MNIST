from data import load_mnist, batches
from model import NeuralNet
from loss import CrossEntropyLoss
from optimizers import RMSProp
from utils import accuracy, save

X_train, y_train, X_test, y_test = load_mnist()

model = NeuralNet()
loss_fn = CrossEntropyLoss()

optimizer = RMSProp(model.params(), lr=0.001)

EPOCHS = 15
BATCH = 64

for epoch in range(EPOCHS):
    for Xb, yb in batches(X_train, y_train, BATCH):
        preds = model.forward(Xb)
        loss = loss_fn.forward(preds, yb)

        grad = loss_fn.backward()
        model.backward(grad)

        optimizer.step(model.params(), model.grads())

    test_preds = model.forward(X_test)
    acc = accuracy(test_preds, y_test)
    print(f"Epoch {epoch+1} | Loss {loss:.4f} | Accuracy {acc*100:.2f}%")

save(model)
print("Model saved to weights.npz")
