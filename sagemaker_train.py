import numpy as np
import pandas as pd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def compute_cost(X, y, w, b, lam=0.0):
    m = len(y)
    z = X.dot(w) + b
    p = sigmoid(z)
    eps = 1e-9
    cost = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    if lam > 0:
        cost += lam * np.sum(w ** 2) / (2 * m)
    return cost


def compute_gradient(X, y, w, b, lam=0.0):
    m = len(y)
    z = X.dot(w) + b
    p = sigmoid(z)
    dz = p - y
    dw = X.T.dot(dz) / m
    db = np.mean(dz)
    if lam > 0:
        dw += (lam / m) * w
    return dw, db


def train_logistic_regression(X, y, lr=0.01, epochs=1000, lam=0.0):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    history = []
    for _ in range(epochs):
        dw, db = compute_gradient(X, y, w, b, lam)
        w -= lr * dw
        b -= lr * db
        history.append(compute_cost(X, y, w, b, lam))
    return w, b, history


def main():
    df = pd.read_csv('heart.csv')
    df['target'] = (df['target'] > 0).astype(int)
    selected = ['age', 'chol', 'trestbps', 'thalach', 'oldpeak', 'ca']
    X = df[selected].astype(float)
    y = df['target'].astype(int).to_numpy()
    X = (X - X.mean()) / X.std(ddof=0)
    X = X.to_numpy()
    w, b, history = train_logistic_regression(X, y, lr=0.01, epochs=1000, lam=0.01)
    np.save('sagemaker_weights.npy', w)
    np.save('sagemaker_bias.npy', np.array([b]))
    print('Training completed. Final cost:', history[-1])


if __name__ == '__main__':
    main()
