# Neural Network from Scratch on Financial Data

Predicting monthly stock returns using **momentum**, **volatility**, and **P/E ratio** as features.

Built entirely in **pure numpy**. I used no PyTorch, no sklearn for the core model. Every forward pass, backpropagation step, and weight update is implemented from first principles.

---

## Motivation

Most neural network tutorials use MNIST or toy datasets. This project applies the same mathematical machinery to a real problem in quantitative finance. Cross-sectional return prediction, where the data is noisy, the signal is weak, and honest evaluation matters as much as implementation.

---

## Architecture

```
Input(3)  →  H1·ReLU(8)  →  H2·ReLU(4)  →  Output·Linear(1)
```

| Layer | Weight matrix | Shape | Bias | Parameters |
|---|---|---|---|---|
| Input → H1 | W₁ | 3 × 8 | b₁ (1×8) | 32 |
| H1 → H2 | W₂ | 8 × 4 | b₂ (1×4) | 36 |
| H2 → Output | W₃ | 4 × 1 | b₃ (1×1) | 5 |
| **Total** | | | | **73** |

---

## Mathematical Derivations

### 1. Forward Pass

For each layer $i$, the computation is:

$$Z_i = A_{i-1} W_i + b_i$$

$$A_i = f(Z_i)$$

where $A_0 = X$ (the input matrix), $f$ is ReLU for hidden layers and identity (linear) for the output.

In full for our architecture:

$$Z_1 = X W_1 + b_1 \quad A_1 = \text{ReLU}(Z_1)$$

$$Z_2 = A_1 W_2 + b_2 \quad A_2 = \text{ReLU}(Z_2)$$

$$Z_3 = A_2 W_3 + b_3 \quad \hat{y} = Z_3$$

**Dimension check** (n = batch size):

$$X: (n \times 3) \quad W_1: (3 \times 8) \quad \Rightarrow Z_1: (n \times 8) \checkmark$$

$$A_1: (n \times 8) \quad W_2: (8 \times 4) \quad \Rightarrow Z_2: (n \times 4) \checkmark$$

$$A_2: (n \times 4) \quad W_3: (4 \times 1) \quad \Rightarrow \hat{y}: (n \times 1) \checkmark$$

---

### 2. Loss Function

Mean Squared Error:

$$L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \frac{1}{n}(y - \hat{y})^\top(y - \hat{y})$$

This is identical to the OLS residual sum of squares scaled by $\frac{1}{n}$. Under Gaussian error assumptions, minimising MSE is equivalent to Maximum Likelihood Estimation.

$$\frac{\partial L}{\partial \hat{y}} = \frac{2}{n}(\hat{y} - y)$$

---

### 3. Backpropagation — Chain Rule

Backprop applies the chain rule in reverse through every layer. The goal is to compute $\frac{\partial L}{\partial W_i}$ and $\frac{\partial L}{\partial b_i}$ for every layer.

#### Output layer gradient

$$\frac{\partial L}{\partial Z_3} = \frac{2}{n}(\hat{y} - y) \quad \text{(linear activation, derivative = 1)}$$

$$\frac{\partial L}{\partial W_3} = A_2^\top \cdot \frac{\partial L}{\partial Z_3} \qquad \text{shape: } (4 \times n)(n \times 1) = (4 \times 1) \checkmark$$

$$\frac{\partial L}{\partial b_3} = \sum_{\text{rows}} \frac{\partial L}{\partial Z_3} \qquad \text{shape: } (1 \times 1) \checkmark$$

#### Passing gradient to previous layer

$$\frac{\partial L}{\partial A_2} = \frac{\partial L}{\partial Z_3} \cdot W_3^\top \qquad \text{shape: } (n \times 1)(1 \times 4) = (n \times 4) \checkmark$$

#### Hidden layer gradient (chain rule through ReLU)

ReLU derivative:

$$\text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}$$

$$\frac{\partial L}{\partial Z_2} = \frac{\partial L}{\partial A_2} \odot \text{ReLU}'(Z_2) \qquad \text{(elementwise — the gradient mask)}$$

$$\frac{\partial L}{\partial W_2} = A_1^\top \cdot \frac{\partial L}{\partial Z_2} \qquad \text{shape: } (8 \times n)(n \times 4) = (8 \times 4) \checkmark$$

#### Full chain to W₁

$$\frac{\partial L}{\partial A_1} = \frac{\partial L}{\partial Z_2} \cdot W_2^\top$$

$$\frac{\partial L}{\partial Z_1} = \frac{\partial L}{\partial A_1} \odot \text{ReLU}'(Z_1)$$

$$\frac{\partial L}{\partial W_1} = X^\top \cdot \frac{\partial L}{\partial Z_1} \qquad \text{shape: } (3 \times n)(n \times 8) = (3 \times 8) \checkmark$$

**Key property:** every gradient $\frac{\partial L}{\partial W_i}$ has exactly the same shape as $W_i$. This is required for the update step.

---

### 4. Weight Update — Gradient Descent

$$W_i \leftarrow W_i - \alpha \cdot \frac{\partial L}{\partial W_i}$$

$$b_i \leftarrow b_i - \alpha \cdot \frac{\partial L}{\partial b_i}$$

where $\alpha$ is the learning rate. We use mini-batch gradient descent with batch size 32.

---

### 5. Weight Initialisation — He Initialisation

We use **He initialisation** rather than small random values:

$$W_i \sim \mathcal{N}\left(0,\ \sqrt{\frac{2}{n_{\text{in}}}}\right)$$

**Why:** With multiple ReLU layers, naive small random initialisation causes the variance of activations to shrink layer by layer, gradients vanish before reaching early layers. He initialisation keeps variance stable by accounting for the fact that ReLU kills roughly half the neurons.

---

### 6. Why Chronological Train/Test Split

Financial data is a time series. A random split would allow the model to train on observations from 2022 and test on 2018 — effectively leaking future information backwards. We always split by time:

```
|─────── train (80%) ───────|── test (20%) ──|
t=0                       t=T*0.8           t=T
```

---

## Features & Target

| Feature | Construction | Academic source |
|---|---|---|
| Momentum | $\frac{P_{t-1}}{P_{t-13}} - 1$ (12-1 month return) | Jegadeesh & Titman (1993) |
| Volatility | 12-month rolling std of monthly returns | Ang et al. (2006) |
| P/E Ratio | Trailing price / earnings | Basu (1977) |
| **Target y** | Forward 1-month return | — |

All features normalised to zero mean and unit variance before training. Target $y$ is **not** normalised. We want actual return predictions.

---

## Results

Low R² is expected and is not a failure of implementation. Monthly stock returns are dominated by idiosyncratic noise. The academic literature treats R² of 0.5–2% as meaningful signal in cross-sectional return prediction.

What this project validates:
- Correct forward pass dimensions throughout
- Gradients match weight shapes at every layer
- Loss decreases monotonically across epochs
- Gradient norms remain healthy (no vanishing/exploding)

Dataset covers 10 large-cap US equities from 2015–2024 (~1080 monthly observations). A production model would require broader cross-sectional coverage across the full market universe to avoid survivorship bias.

---

## Project Structure

```
quant-nn-backprop/
│
├── data/
│   └── features.csv          # generated by fetch_data.py
│
├── fetch_data.py             # yfinance data pipeline + synthetic fallback
├── neural_net.py             # full NN: forward, backprop, training loop
├── main.ipynb                # end-to-end walkthrough with plots
└── README.md
```

---

## Quickstart

```bash
# Clone and install dependencies
git clone https://github.com/YOUR_USERNAME/quant-nn-backprop
cd quant-nn-backprop
pip install numpy pandas matplotlib yfinance jupyter

# Fetch real data (requires internet)
python fetch_data.py

# Run notebook
jupyter notebook main.ipynb
```

If `yfinance` is unavailable, `fetch_data.py` automatically generates synthetic data with realistic feature distributions so the full pipeline still runs.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | All matrix operations — forward pass, backprop, weight updates |
| `pandas` | Data loading, feature engineering, rolling calculations |
| `matplotlib` | Loss curves, gradient norm plots, prediction visualisations |
| `yfinance` | Real stock data (optional — synthetic fallback included) |
| `jupyter` | Running the notebook |

---

## References

- Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers.* Journal of Finance.
- Ang, A. et al. (2006). *The Cross-Section of Volatility and Expected Returns.* Journal of Finance.
- Basu, S. (1977). *Investment Performance of Common Stocks in Relation to their Price-Earnings Ratios.* Journal of Finance.
- He, K. et al. (2015). *Delving Deep into Rectifiers.* ICCV. *(He initialisation)*
- Rumelhart, D., Hinton, G. & Williams, R. (1986). *Learning representations by back-propagating errors.* Nature.
