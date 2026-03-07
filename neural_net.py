"""
neural_net.py
=============
Neural network built from scratch in pure numpy.
No PyTorch. No sklearn. No magic.

Architecture:
    Input(3) -> H1·ReLU(8) -> H2·ReLU(4) -> Output·Linear(1)

Features (X):
    - momentum   (normalised)
    - volatility (normalised)
    - pe_ratio   (normalised)

Target (y):
    - next month return (regression)

Usage:
    from neural_net import NeuralNet
    model = NeuralNet(layer_sizes=[3, 8, 4, 1], lr=0.01)
    model.train(X_train, y_train, epochs=500)
    preds = model.predict(X_test)
"""

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# ACTIVATIONS
# ══════════════════════════════════════════════════════════════════════════════

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    """Derivative of ReLU — 1 where z>0, 0 elsewhere."""
    return (z > 0).astype(float)

def linear(z):
    """No activation — used for regression output."""
    return z

def linear_deriv(z):
    return np.ones_like(z)


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def init_weights(layer_sizes, seed=42):
    """
    He initialisation — recommended for ReLU networks.

    W ~ N(0, sqrt(2 / n_in))

    Why not random small numbers like our toy example?
    With many layers, weights that are too small cause vanishing gradients.
    He init keeps variance stable across layers.

    Returns:
        params: dict with W1, b1, W2, b2, ... for each layer
    """
    rng = np.random.default_rng(seed)
    params = {}

    for i in range(len(layer_sizes) - 1):
        n_in  = layer_sizes[i]
        n_out = layer_sizes[i + 1]

        # He initialisation
        params[f"W{i+1}"] = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out))
        params[f"b{i+1}"] = np.zeros((1, n_out))

    return params


# ══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK CLASS
# ══════════════════════════════════════════════════════════════════════════════

class NeuralNet:
    """
    Fully connected neural network for regression.

    Architecture:
        Input -> [Hidden layers with ReLU] -> Output (linear)

    Parameters learned via mini-batch gradient descent + backpropagation.
    """

    def __init__(self, layer_sizes=[3, 8, 4, 1], lr=0.01, seed=42):
        """
        Args:
            layer_sizes : list of ints — neuron count per layer including input/output
                          e.g. [3, 8, 4, 1] means:
                          input=3, hidden1=8, hidden2=4, output=1
            lr          : learning rate alpha
            seed        : random seed for reproducibility
        """
        self.layer_sizes = layer_sizes
        self.lr          = lr
        self.n_layers    = len(layer_sizes) - 1   # number of weight layers
        self.params      = init_weights(layer_sizes, seed)
        self.loss_history = []

        print(f"Network initialised:")
        print(f"  Architecture : {' -> '.join(map(str, layer_sizes))}")
        print(f"  Total params : {self._count_params()}")
        print(f"  Learning rate: {lr}")
        print(f"  Activation   : ReLU (hidden), Linear (output)\n")

    def _count_params(self):
        return sum(v.size for v in self.params.values())


    # ── FORWARD PASS ─────────────────────────────────────────────────────────

    def forward(self, X):
        """
        Full forward pass through all layers.

        For each layer i:
            Z_i = A_{i-1} @ W_i + b_i     (linear step)
            A_i = activation(Z_i)          (ReLU for hidden, linear for output)

        Returns:
            cache : dict storing all Z and A values needed for backprop
        """
        cache = {"A0": X}   # A0 = input X

        for i in range(1, self.n_layers + 1):
            W = self.params[f"W{i}"]
            b = self.params[f"b{i}"]
            A_prev = cache[f"A{i-1}"]

            # ── Linear step ────────────────────────────────────────────────
            Z = A_prev @ W + b                        # (n × n_out)
            cache[f"Z{i}"] = Z

            # ── Activation ─────────────────────────────────────────────────
            if i == self.n_layers:
                A = linear(Z)                         # output layer — no activation
            else:
                A = relu(Z)                           # hidden layers — ReLU

            cache[f"A{i}"] = A

        return cache


    # ── LOSS ─────────────────────────────────────────────────────────────────

    def compute_loss(self, y_hat, y):
        """
        Mean Squared Error:
            L = (1/n) * sum((y_hat - y)^2)

        Matrix form (identical to OLS residual sum of squares / n):
            L = (1/n) * (y_hat - y)^T (y_hat - y)
        """
        n = y.shape[0]
        return np.mean((y_hat - y) ** 2)


    # ── BACKPROPAGATION ───────────────────────────────────────────────────────

    def backward(self, cache, y):
        """
        Backpropagation — chain rule applied in reverse through all layers.

        For output layer:
            dL/dZ_out = dL/dA_out * activation'(Z_out)
                      = (2/n)(y_hat - y) * 1          [linear activation]

        For each hidden layer i (going backwards):
            dL/dZ_i = (dL/dA_i) * ReLU'(Z_i)         [elementwise]
            dL/dW_i = A_{i-1}^T @ dL/dZ_i
            dL/db_i = sum(dL/dZ_i, axis=0)
            dL/dA_{i-1} = dL/dZ_i @ W_i^T             [pass gradient back]

        Returns:
            grads : dict with dW1, db1, dW2, db2, ...
        """
        n     = y.shape[0]
        grads = {}

        y_hat = cache[f"A{self.n_layers}"]

        # ── Gradient of loss w.r.t. output ─────────────────────────────────
        # dL/dA_out = (2/n)(y_hat - y)
        # For linear output: dL/dZ_out = dL/dA_out * 1
        dA = (2 / n) * (y_hat - y)                   # (n × 1)

        # ── Backprop through layers in reverse ─────────────────────────────
        for i in reversed(range(1, self.n_layers + 1)):
            Z      = cache[f"Z{i}"]
            A_prev = cache[f"A{i-1}"]
            W      = self.params[f"W{i}"]

            # Chain rule through activation
            if i == self.n_layers:
                dZ = dA * linear_deriv(Z)             # output: linear deriv = 1
            else:
                dZ = dA * relu_deriv(Z)               # hidden: ReLU mask

            # Gradient w.r.t. weights and biases
            grads[f"dW{i}"] = A_prev.T @ dZ           # (n_in × n_out)
            grads[f"db{i}"] = np.sum(dZ, axis=0, keepdims=True)  # (1 × n_out)

            # Pass gradient to previous layer
            dA = dZ @ W.T                             # (n × n_in)

        return grads


    # ── WEIGHT UPDATE ─────────────────────────────────────────────────────────

    def update_params(self, grads):
        """
        Gradient descent update:
            W_i <- W_i - lr * dL/dW_i
            b_i <- b_i - lr * dL/db_i
        """
        for i in range(1, self.n_layers + 1):
            self.params[f"W{i}"] -= self.lr * grads[f"dW{i}"]
            self.params[f"b{i}"] -= self.lr * grads[f"db{i}"]


    # ── TRAINING LOOP ─────────────────────────────────────────────────────────

    def train(self, X, y, epochs=500, batch_size=32, verbose=True):
        """
        Mini-batch gradient descent training loop.

        Each epoch:
            1. Shuffle data
            2. Split into mini-batches
            3. For each batch: forward -> loss -> backward -> update

        Args:
            X          : (n × k) feature matrix
            y          : (n × 1) target vector
            epochs     : number of full passes through data
            batch_size : observations per mini-batch
            verbose    : print loss every 50 epochs
        """
        n = X.shape[0]
        rng = np.random.default_rng(42)

        print(f"Training for {epochs} epochs, batch_size={batch_size}...")
        print(f"{'Epoch':>8}  {'Train Loss':>12}")
        print("-" * 24)

        for epoch in range(1, epochs + 1):

            # ── Shuffle data each epoch ────────────────────────────────────
            idx = rng.permutation(n)
            X_shuf, y_shuf = X[idx], y[idx]

            epoch_loss = 0.0
            n_batches  = 0

            # ── Mini-batch loop ────────────────────────────────────────────
            for start in range(0, n, batch_size):
                end   = min(start + batch_size, n)
                X_bat = X_shuf[start:end]
                y_bat = y_shuf[start:end]

                # Forward
                cache = self.forward(X_bat)
                y_hat = cache[f"A{self.n_layers}"]

                # Loss
                loss = self.compute_loss(y_hat, y_bat)
                epoch_loss += loss
                n_batches  += 1

                # Backward
                grads = self.backward(cache, y_bat)

                # Update
                self.update_params(grads)

            # ── Record average epoch loss ──────────────────────────────────
            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)

            if verbose and (epoch % 50 == 0 or epoch == 1):
                print(f"{epoch:>8}  {avg_loss:>12.6f}")

        print(f"\nTraining complete. Final loss: {self.loss_history[-1]:.6f}")


    # ── PREDICT ───────────────────────────────────────────────────────────────

    def predict(self, X):
        """Forward pass only — returns predictions."""
        cache = self.forward(X)
        return cache[f"A{self.n_layers}"]


    # ── EVALUATE ──────────────────────────────────────────────────────────────

    def evaluate(self, X, y, label=""):
        """
        Computes MSE and R² on given data.

        R² = 1 - SS_res / SS_tot
           = fraction of variance in y explained by the model
           (0 = no better than mean, 1 = perfect, <0 = worse than mean)
        """
        y_hat  = self.predict(X)
        mse    = np.mean((y_hat - y) ** 2)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2     = 1 - ss_res / ss_tot

        print(f"{label}")
        print(f"  MSE : {mse:.6f}")
        print(f"  R²  : {r2:.4f}  {'(good)' if r2 > 0.1 else '(low — expected for financial data)'}")
        return {"mse": mse, "r2": r2}


    # ── GRADIENT INSPECTOR ────────────────────────────────────────────────────

    def inspect_gradients(self, X_batch, y_batch):
        """
        Shows gradient norms per layer — useful for diagnosing
        vanishing or exploding gradients.
        """
        cache = self.forward(X_batch)
        grads = self.backward(cache, y_batch)

        print("\nGradient norms per layer:")
        print(f"  {'Layer':>8}  {'||dW||':>10}  {'||db||':>10}")
        print("  " + "-" * 34)
        for i in range(1, self.n_layers + 1):
            dw_norm = np.linalg.norm(grads[f"dW{i}"])
            db_norm = np.linalg.norm(grads[f"db{i}"])
            print(f"  {'Layer '+str(i):>8}  {dw_norm:>10.6f}  {db_norm:>10.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / TEST SPLIT UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def train_test_split(df, test_frac=0.2):
    """
    Chronological split — NOT random.

    IMPORTANT: Random splits leak future data into training in time series.
    We always split by time: first 80% = train, last 20% = test.
    """
    n      = len(df)
    cutoff = int(n * (1 - test_frac))

    X = df[["momentum", "volatility", "pe_ratio"]].values   # (n × 3)
    y = df[["target"]].values                                # (n × 1)

    X_train, X_test = X[:cutoff], X[cutoff:]
    y_train, y_test = y[:cutoff], y[cutoff:]

    print(f"Train size : {X_train.shape[0]} observations")
    print(f"Test size  : {X_test.shape[0]} observations")
    print(f"Note: chronological split used (not random) to prevent data leakage\n")

    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — run standalone to verify everything works
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    import pandas as pd
    from fetch_data import generate_synthetic_data, clean_and_save

    # ── Load or generate data ──────────────────────────────────────────────
    if os.path.exists("data/features.csv"):
        print("Loading data/features.csv...")
        df = pd.read_csv("data/features.csv")
    else:
        print("No data found — generating synthetic data...")
        df = generate_synthetic_data(n_samples=500)
        df = clean_and_save(df)

    # ── Split ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(df)

    # ── Build and train model ──────────────────────────────────────────────
    # Architecture: 3 features -> 8 -> 4 -> 1 output
    model = NeuralNet(layer_sizes=[3, 8, 4, 1], lr=0.01)

    model.train(X_train, y_train, epochs=500, batch_size=32)

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\n" + "="*40)
    model.evaluate(X_train, y_train, label="Train set:")
    model.evaluate(X_test,  y_test,  label="Test set:")

    # ── Gradient inspection ────────────────────────────────────────────────
    model.inspect_gradients(X_train[:32], y_train[:32])

    # ── Honest note ────────────────────────────────────────────────────────
    print("""
Note on R² in finance:
  Low R² is expected. Monthly stock returns are mostly noise.
  A positive R² on test data — even 0.02 — suggests real signal.
  Our goal here is correct implementation, not beating the market.
    """)
