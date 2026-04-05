---
title: Autoencoder
description: Autoencoder — Detailed Step-by-Step Notes
topic: deep-learning
---
# Autoencoder — Detailed Step-by-Step Notes

## 1. Core Idea and Why It Matters

An **autoencoder** is a neural network that tries to **reconstruct its own input**.

It does this in two stages:

1. **Encoder**: compresses the input into a smaller internal representation
2. **Decoder**: expands that compressed representation back into the original form

So the network learns:

$$
x \rightarrow z \rightarrow \hat{x}
$$

Where:

* $x$ = original input
* $z$ = compressed representation, often called **latent code**, **bottleneck representation**, or **embedding**
* $\hat{x}$ = reconstructed input

### Intuition

Imagine you must describe a large image using only a few important numbers.
You cannot store every pixel, so you are forced to keep only the most important information.

That is exactly what an autoencoder does.

It learns:

* what information is essential
* what can be compressed
* how to rebuild the input from compressed information

### Why is this useful?

Autoencoders are useful for:

* **dimensionality reduction**
* **feature learning**
* **data compression**
* **denoising**
* **anomaly detection**
* **representation learning before downstream tasks**

---

# 2. Big Picture Architecture

A simple autoencoder looks like this:

$$
x \xrightarrow{\text{Encoder}} z \xrightarrow{\text{Decoder}} \hat{x}
$$

If the encoder and decoder are neural networks, then:

$$
z = f_{\text{enc}}(x)
$$

$$
\hat{x} = f_{\text{dec}}(z)
$$

The whole objective is to make $\hat{x}$ as close as possible to $x$.

That means we minimize a **reconstruction loss**:

$$
\mathcal{L}(x, \hat{x})
$$

Usually:

* Mean Squared Error (MSE) for real-valued data
* Binary Cross Entropy (BCE) for binary or normalized pixel data in $[0,1]$

---

# 3. Everyday Analogy

Suppose you read a long article and want to write a short summary.

* The **article** is the input $x$
* The **summary** is the latent code $z$
* Rewriting the article from the summary is the decoded output $\hat{x}$

If the summary keeps the important ideas, the rewritten article will be close to the original.

If the summary is poor, reconstruction will be poor.

So the bottleneck forces the model to learn meaningful structure.

---

# 4. Mathematical Definition

## 4.1 Input

Let the input be a vector:

$$
x \in \mathbb{R}^n
$$

This means $x$ has $n$ features.

Example:

$$
x = \begin{bmatrix}
0.8 \
0.2 \
0.1 \
0.9
\end{bmatrix}
\in \mathbb{R}^4
$$

---

## 4.2 Encoder

The encoder maps input $x$ to latent representation $z$:

$$
z = \phi(W_e x + b_e)
$$

Where:

* $W_e \in \mathbb{R}^{m \times n}$ = encoder weight matrix
* $b_e \in \mathbb{R}^{m}$ = encoder bias vector
* $\phi(\cdot)$ = activation function
* $z \in \mathbb{R}^{m}$ = latent code

Usually $m < n$ in a bottleneck autoencoder.

That means compression happens.

---

## 4.3 Decoder

The decoder reconstructs the input from $z$:

$$
\hat{x} = \psi(W_d z + b_d)
$$

Where:

* $W_d \in \mathbb{R}^{n \times m}$ = decoder weight matrix
* $b_d \in \mathbb{R}^{n}$ = decoder bias vector
* $\psi(\cdot)$ = decoder activation
* $\hat{x} \in \mathbb{R}^{n}$ = reconstructed input

---

## 4.4 Full Mapping

So the full autoencoder is:

$$
\hat{x} = \psi\left(W_d , \phi(W_e x + b_e) + b_d\right)
$$

---

# 5. Reconstruction Loss

The model learns by minimizing the difference between input $x$ and reconstruction $\hat{x}$.

## 5.1 Mean Squared Error (MSE)

For continuous data:

$$
\mathcal{L}*{\text{MSE}} = \frac{1}{n}\sum*{i=1}^{n}(x_i - \hat{x}_i)^2
$$

Where:

* $x_i$ = $i$-th component of the original input
* $\hat{x}_i$ = $i$-th component of the reconstruction

### Intuition

If reconstruction is perfect, every term is zero.

So:

$$
x = \hat{x} \Rightarrow \mathcal{L}=0
$$

The larger the difference, the larger the loss.

---

## 5.2 Binary Cross Entropy (BCE)

For binary-valued or pixel values scaled to $[0,1]$:

$$
\mathcal{L}*{\text{BCE}} = - \frac{1}{n}\sum*{i=1}^{n}\left[x_i \log(\hat{x}_i) + (1-x_i)\log(1-\hat{x}_i)\right]
$$

This is often used when decoder output uses a sigmoid activation.

---

# 6. Why the Bottleneck Matters

If the network had unlimited capacity, it could just memorize the identity function:

$$
\hat{x} = x
$$

That would not be very useful.

The point is to force the network to learn a **compressed internal representation**.

This happens when:

* latent dimension is smaller than input dimension
* or the model is regularized
* or noise/corruption is added

This forces the model to discover the hidden structure of the data.

---

# 7. Step-by-Step Numerical Example

Let us build the smallest useful autoencoder manually.

## 7.1 Input

Assume input dimension is 2:

$$
x =
\begin{bmatrix}
1 \
0
\end{bmatrix}
$$

We will compress it to 1 number.

So:

* input dimension = 2
* latent dimension = 1
* output dimension = 2

---

## 7.2 Encoder Parameters

Let:

$$
W_e =
\begin{bmatrix}
0.6 & 0.4
\end{bmatrix}
$$

$$
b_e =
\begin{bmatrix}
0.1
\end{bmatrix}
$$

Use identity activation for simplicity first:

$$
z = W_e x + b_e
$$

Substitute:

$$
z =
\begin{bmatrix}
0.6 & 0.4
\end{bmatrix}
\begin{bmatrix}
1 \
0
\end{bmatrix}
+
\begin{bmatrix}
0.1
\end{bmatrix}
$$

Multiply:

$$
z = 0.6(1) + 0.4(0) + 0.1 = 0.7
$$

So the compressed code is:

$$
z = 0.7
$$

---

## 7.3 Decoder Parameters

Let:

$$
W_d =
\begin{bmatrix}
0.9 \
0.2
\end{bmatrix}
$$

$$
b_d =
\begin{bmatrix}
0.05 \
0.10
\end{bmatrix}
$$

Then:

$$
\hat{x} = W_d z + b_d
$$

Substitute:

$$
\hat{x} =
\begin{bmatrix}
0.9 \
0.2
\end{bmatrix}(0.7)
+
\begin{bmatrix}
0.05 \
0.10
\end{bmatrix}
$$

Compute:

$$
\hat{x} =
\begin{bmatrix}
0.63 \
0.14
\end{bmatrix}
+
\begin{bmatrix}
0.05 \
0.10
\end{bmatrix}
=============

\begin{bmatrix}
0.68 \
0.24
\end{bmatrix}
$$

So reconstruction is:

$$
\hat{x} =
\begin{bmatrix}
0.68 \
0.24
\end{bmatrix}
$$

Original was:

$$
x =
\begin{bmatrix}
1 \
0
\end{bmatrix}
$$

Not perfect, but somewhat close.

---

## 7.4 Compute Loss

Using MSE:

$$
\mathcal{L} = \frac{1}{2}\left[(1-0.68)^2 + (0-0.24)^2\right]
$$

$$
= \frac{1}{2}\left[(0.32)^2 + (-0.24)^2\right]
$$

$$
= \frac{1}{2}(0.1024 + 0.0576)
$$

$$
= \frac{0.1600}{2} = 0.08
$$

So the loss is:

$$
\mathcal{L} = 0.08
$$

Training adjusts weights and biases to reduce this.

---

# 8. First-Principles Reasoning

## 8.1 Why reconstruction can teach useful features

Suppose data is not random.

Example:

* images of handwritten digits
* patient lab patterns
* documents about similar topics

These data points usually lie on a lower-dimensional structure.

Even if an image has 784 pixels, the true factors may be much fewer:

* stroke thickness
* angle
* shape
* location

An autoencoder tries to discover these hidden factors.

So it learns a representation $z$ that retains the important information needed to rebuild the input.

---

## 8.2 Why compression leads to abstraction

If you must compress 100 numbers into 5 numbers, you cannot keep noise and unimportant details equally well.

So the network tends to keep:

* dominant structure
* repeated patterns
* important correlations

This is why autoencoders can learn meaningful internal features.

---

# 9. Training an Autoencoder from Scratch

Training uses the same neural network principles as other feedforward models.

The steps are:

1. Initialize weights and biases
2. For each training example:

   * forward pass
   * compute loss
   * backward pass
   * update parameters
3. Repeat for many epochs

---

# 10. Forward Pass Step by Step

Consider a one-hidden-layer autoencoder.

## 10.1 Equations

### Encoder

$$
a^{(1)} = W_e x + b_e
$$

$$
z = \phi(a^{(1)})
$$

### Decoder

$$
a^{(2)} = W_d z + b_d
$$

$$
\hat{x} = \psi(a^{(2)})
$$

Then compute loss:

$$
\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}(x_i-\hat{x}_i)^2
$$

---

# 11. Backpropagation Intuition

The idea is simple:

* If reconstruction is wrong, measure the error
* Compute how much each weight contributed to that error
* Move each weight slightly in the direction that reduces error

This uses derivatives.

---

# 12. Backpropagation Derivation for MSE Autoencoder

We derive for one sample.

Let:

$$
\mathcal{L} = \frac{1}{2}\sum_{i=1}^{n}(x_i - \hat{x}_i)^2
$$

We use factor $\frac{1}{2}$ to simplify derivatives.

---

## 12.1 Output Error

Derivative of loss w.r.t. reconstruction:

$$
\frac{\partial \mathcal{L}}{\partial \hat{x}_i} = \hat{x}_i - x_i
$$

Vector form:

$$
\frac{\partial \mathcal{L}}{\partial \hat{x}} = \hat{x} - x
$$

If decoder activation is identity, then:

$$
\delta^{(2)} = \hat{x} - x
$$

If decoder activation is not identity, multiply elementwise by derivative:

$$
\delta^{(2)} = (\hat{x}-x) \odot \psi'(a^{(2)})
$$

Where $\odot$ means elementwise multiplication.

---

## 12.2 Gradients for Decoder

Since:

$$
a^{(2)} = W_d z + b_d
$$

Then:

$$
\frac{\partial \mathcal{L}}{\partial W_d} = \delta^{(2)} z^T
$$

$$
\frac{\partial \mathcal{L}}{\partial b_d} = \delta^{(2)}
$$

---

## 12.3 Backpropagate to Latent Code

The decoder error affects the latent representation:

$$
\frac{\partial \mathcal{L}}{\partial z} = W_d^T \delta^{(2)}
$$

---

## 12.4 Encoder Error

Since:

$$
z = \phi(a^{(1)})
$$

Then:

$$
\delta^{(1)} = (W_d^T \delta^{(2)}) \odot \phi'(a^{(1)})
$$

---

## 12.5 Gradients for Encoder

Since:

$$
a^{(1)} = W_e x + b_e
$$

Then:

$$
\frac{\partial \mathcal{L}}{\partial W_e} = \delta^{(1)} x^T
$$

$$
\frac{\partial \mathcal{L}}{\partial b_e} = \delta^{(1)}
$$

---

## 12.6 Gradient Descent Update

For learning rate $\eta$:

$$
W_e \leftarrow W_e - \eta \frac{\partial \mathcal{L}}{\partial W_e}
$$

$$
b_e \leftarrow b_e - \eta \frac{\partial \mathcal{L}}{\partial b_e}
$$

$$
W_d \leftarrow W_d - \eta \frac{\partial \mathcal{L}}{\partial W_d}
$$

$$
b_d \leftarrow b_d - \eta \frac{\partial \mathcal{L}}{\partial b_d}
$$

---

# 13. Activation Functions

Common activation functions:

## 13.1 Sigmoid

$$
\sigma(x) = \frac{1}{1+e^{-x}}
$$

Derivative:

$$
\sigma'(x) = \sigma(x)(1-\sigma(x))
$$

Use when output must be in $[0,1]$.

---

## 13.2 Tanh

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

Derivative:

$$
\frac{d}{dx}\tanh(x)=1-\tanh^2(x)
$$

Output lies in $[-1,1]$.

---

## 13.3 ReLU

$$
\text{ReLU}(x)=\max(0,x)
$$

Derivative:

$$
\text{ReLU}'(x)=
\begin{cases}
1 & x>0 \
0 & x\le 0
\end{cases}
$$

Popular in deeper models.

---

# 14. Practical Example with Real Interpretation

## 14.1 Healthcare Example

Suppose each patient has 20 measurements:

* blood pressure
* glucose
* creatinine
* heart rate
* age
* medication counts
* diagnosis indicators

So:

$$
x \in \mathbb{R}^{20}
$$

You build an autoencoder with latent size 5:

$$
20 \rightarrow 5 \rightarrow 20
$$

The latent code may learn compressed patient state factors like:

* metabolic health
* kidney function
* cardiovascular stability
* disease burden
* medication intensity

These 5 learned factors can then be used for:

* clustering patients
* anomaly detection
* RL state encoding
* visualization

---

## 14.2 Computer Vision Example

MNIST image has:

$$
28 \times 28 = 784
$$

Flattened input:

$$
x \in \mathbb{R}^{784}
$$

Autoencoder:

$$
784 \rightarrow 128 \rightarrow 32 \rightarrow 128 \rightarrow 784
$$

The bottleneck 32-dimensional code learns key structure like:

* digit identity
* stroke thickness
* slant
* shape style

This compressed representation can be used for:

* denoising
* compression
* learned features for classification

---

## 14.3 NLP Example

Suppose a document is represented by a bag-of-words vector with 5000 dimensions.

Autoencoder:

$$
5000 \rightarrow 256 \rightarrow 64 \rightarrow 256 \rightarrow 5000
$$

The bottleneck representation may learn:

* topic
* sentiment
* style
* domain keywords

This can help in:

* topic clustering
* anomaly detection in text
* semantic retrieval

---

# 15. Undercomplete vs Overcomplete Autoencoders

## 15.1 Undercomplete Autoencoder

Latent dimension is smaller than input dimension:

$$
m < n
$$

Example:

$$
100 \rightarrow 20 \rightarrow 100
$$

This forces compression.

Usually the most intuitive type.

---

## 15.2 Overcomplete Autoencoder

Latent dimension is larger than input dimension:

$$
m > n
$$

Example:

$$
20 \rightarrow 100 \rightarrow 20
$$

This can still learn useful representations if regularized properly, otherwise it may just copy input.

---

# 16. Major Variants

## 16.1 Basic Autoencoder

Learns to reconstruct input as accurately as possible.

### Use case

General representation learning and compression.

---

## 16.2 Sparse Autoencoder

Adds a sparsity constraint on hidden units.

This means only a few neurons should be active for any input.

### Why?

It encourages specialized feature detectors.

### Typical extra penalty

For hidden unit average activation $\hat{\rho}_j$ and target sparsity $\rho$:

$$
\Omega_{\text{sparse}} = \sum_j \text{KL}(\rho ,|, \hat{\rho}_j)
$$

Total loss:

$$
\mathcal{L}*{\text{total}} = \mathcal{L}*{\text{recon}} + \beta \Omega_{\text{sparse}}
$$

---

## 16.3 Denoising Autoencoder

Instead of feeding clean input $x$, feed a corrupted version $\tilde{x}$:

$$
\tilde{x} \rightarrow z \rightarrow \hat{x}
$$

But compare reconstruction with the original clean input:

$$
\mathcal{L}(x, \hat{x})
$$

### Why?

The network learns robust features rather than simple copying.

---

## 16.4 Contractive Autoencoder

Adds a penalty so the encoder does not change too much with tiny input changes.

Idea:

$$
\Omega_{\text{contractive}} = \left|\frac{\partial z}{\partial x}\right|_F^2
$$

This makes representation locally stable.

---

## 16.5 Variational Autoencoder (VAE)

Instead of learning one fixed code $z$, it learns a probability distribution over latent variables.

Encoder outputs:

* mean $\mu$
* standard deviation $\sigma$

Then sample:

$$
z \sim \mathcal{N}(\mu, \sigma^2)
$$

Loss has two parts:

1. reconstruction loss
2. KL divergence regularization

VAE is more generative than the basic autoencoder.

---

# 17. Comparison Table

| Variant                 | Main Idea                                | Strength                      | Weakness                   | Best Use                              |
| ----------------------- | ---------------------------------------- | ----------------------------- | -------------------------- | ------------------------------------- |
| Basic Autoencoder       | Reconstruct input                        | Simple, intuitive             | Can learn identity mapping | Compression, feature learning         |
| Sparse Autoencoder      | Few hidden activations                   | Learns interpretable features | Extra tuning needed        | Feature discovery                     |
| Denoising Autoencoder   | Reconstruct clean input from noisy input | Robust features               | Need corruption strategy   | Noise removal, robustness             |
| Contractive Autoencoder | Stable encoding near input               | Good local invariance         | Harder to implement        | Manifold learning                     |
| Variational Autoencoder | Probabilistic latent space               | Generative capability         | More complex math          | Data generation, smooth latent spaces |

---

# 18. Storyboard: Step-by-Step Flow

## Example: 6-dimensional patient vector compressed to 2 dimensions

Input:

$$
x = [\text{glucose}, \text{bp}, \text{hr}, \text{creatinine}, \text{age}, \text{med_count}]
$$

Suppose:

$$
x = [160, 140, 88, 1.3, 65, 4]
$$

### Step 1: Normalize the data

Raw scales differ too much. Convert to a comparable range.

Maybe after normalization:

$$
x = [0.78, 0.70, 0.44, 0.30, 0.65, 0.40]
$$

### Step 2: Encoder compresses to latent code

$$
z = [1.2, -0.4]
$$

Interpretation:

* first latent unit may capture disease severity
* second may capture physiological imbalance

### Step 3: Decoder reconstructs from latent code

$$
\hat{x} = [0.75, 0.68, 0.47, 0.29, 0.63, 0.42]
$$

### Step 4: Compute reconstruction error

Difference is small, so encoding is useful.

### Step 5: Update weights

Reduce the differences next time.

---

# 19. How to Implement Without Any Third-Party Library

You only need:

* arrays
* loops
* random number generation
* basic math functions like `exp`
* matrix multiplication written manually

That means you can implement it in:

* C
* C++
* Java
* C#
* Go
* Rust
* Python without NumPy
* JavaScript
* any language with arrays and loops

---

# 20. Required Building Blocks

To implement from scratch, you need to write functions for:

1. matrix multiplication
2. vector addition
3. activation functions
4. activation derivatives
5. loss computation
6. outer product
7. transpose
8. parameter update

---

# 21. Data Structures

For a single-layer autoencoder:

* `W_enc[latent_dim][input_dim]`
* `b_enc[latent_dim]`
* `W_dec[input_dim][latent_dim]`
* `b_dec[input_dim]`

For each sample:

* `x[input_dim]`
* `a1[latent_dim]`
* `z[latent_dim]`
* `a2[input_dim]`
* `x_hat[input_dim]`

---

# 22. From-Scratch Pseudocode

## 22.1 Helper Functions

```text
function sigmoid(x):
    return 1 / (1 + exp(-x))

function sigmoid_derivative(output):
    return output * (1 - output)
```

Note: if you already computed `sigmoid(x)`, derivative is easier from the output value.

---

## 22.2 Matrix-Vector Multiply

```text
function matvec(W, x):
    rows = number of rows in W
    cols = number of cols in W
    y = array of size rows filled with 0

    for i from 0 to rows-1:
        sum = 0
        for j from 0 to cols-1:
            sum = sum + W[i][j] * x[j]
        y[i] = sum

    return y
```

---

## 22.3 Vector Addition

```text
function add_vectors(a, b):
    n = length(a)
    c = array of size n
    for i from 0 to n-1:
        c[i] = a[i] + b[i]
    return c
```

---

## 22.4 Forward Pass

```text
function forward(x):
    a1 = add_vectors(matvec(W_enc, x), b_enc)
    z = apply_activation(a1)

    a2 = add_vectors(matvec(W_dec, z), b_dec)
    x_hat = apply_output_activation(a2)

    return a1, z, a2, x_hat
```

---

## 22.5 MSE Loss

```text
function mse_loss(x, x_hat):
    sum = 0
    n = length(x)
    for i from 0 to n-1:
        diff = x_hat[i] - x[i]
        sum = sum + diff * diff
    return sum / n
```

---

## 22.6 Backward Pass

For simplicity assume decoder activation is identity and encoder uses sigmoid.

```text
function backward(x, a1, z, a2, x_hat):
    input_dim = length(x)
    latent_dim = length(z)

    # output delta
    delta2 = array of size input_dim
    for i from 0 to input_dim-1:
        delta2[i] = x_hat[i] - x[i]

    # gradients for decoder
    grad_W_dec = matrix of size [input_dim][latent_dim]
    grad_b_dec = array of size input_dim

    for i from 0 to input_dim-1:
        grad_b_dec[i] = delta2[i]
        for j from 0 to latent_dim-1:
            grad_W_dec[i][j] = delta2[i] * z[j]

    # backprop to encoder hidden layer
    delta1 = array of size latent_dim filled with 0
    for j from 0 to latent_dim-1:
        sum = 0
        for i from 0 to input_dim-1:
            sum = sum + W_dec[i][j] * delta2[i]
        delta1[j] = sum * sigmoid_derivative(z[j])

    # gradients for encoder
    grad_W_enc = matrix of size [latent_dim][input_dim]
    grad_b_enc = array of size latent_dim

    for j from 0 to latent_dim-1:
        grad_b_enc[j] = delta1[j]
        for k from 0 to input_dim-1:
            grad_W_enc[j][k] = delta1[j] * x[k]

    return grad_W_enc, grad_b_enc, grad_W_dec, grad_b_dec
```

---

## 22.7 Parameter Update

```text
function update_parameters(lr):
    for i over all rows of W_enc:
        for j over all cols of W_enc:
            W_enc[i][j] = W_enc[i][j] - lr * grad_W_enc[i][j]

    for i over b_enc:
        b_enc[i] = b_enc[i] - lr * grad_b_enc[i]

    for i over all rows of W_dec:
        for j over all cols of W_dec:
            W_dec[i][j] = W_dec[i][j] - lr * grad_W_dec[i][j]

    for i over b_dec:
        b_dec[i] = b_dec[i] - lr * grad_b_dec[i]
```

---

## 22.8 Full Training Loop

```text
initialize W_enc, b_enc, W_dec, b_dec with small random values

for epoch from 1 to num_epochs:
    total_loss = 0

    for each training sample x:
        a1, z, a2, x_hat = forward(x)
        loss = mse_loss(x, x_hat)
        total_loss = total_loss + loss

        grad_W_enc, grad_b_enc, grad_W_dec, grad_b_dec = backward(x, a1, z, a2, x_hat)

        update_parameters(learning_rate)

    print(epoch, total_loss / number_of_samples)
```

---

# 23. Tiny Worked Example of One Backprop Step

Let:

* input dimension = 2
* latent dimension = 1
* decoder output activation = identity
* encoder activation = sigmoid

Input:

$$
x =
\begin{bmatrix}
1 \
0
\end{bmatrix}
$$

Encoder:

$$
W_e = [0.5 ;; -0.4], \quad b_e = [0.1]
$$

Compute pre-activation:

$$
a^{(1)} = 0.5(1) + (-0.4)(0) + 0.1 = 0.6
$$

Sigmoid:

$$
z = \sigma(0.6) \approx 0.6457
$$

Decoder:

$$
W_d =
\begin{bmatrix}
0.3 \
0.8
\end{bmatrix}, \quad
b_d =
\begin{bmatrix}
0.0 \
0.0
\end{bmatrix}
$$

Reconstruction:

$$
\hat{x} =
\begin{bmatrix}
0.3 \cdot 0.6457 \
0.8 \cdot 0.6457
\end{bmatrix}
=

\begin{bmatrix}
0.1937 \
0.5166
\end{bmatrix}
$$

Loss derivative at output:

$$
\delta^{(2)} = \hat{x} - x =
\begin{bmatrix}
0.1937 - 1 \
0.5166 - 0
\end{bmatrix}
=

\begin{bmatrix}
-0.8063 \
0.5166
\end{bmatrix}
$$

Gradient for decoder weights:

$$
\frac{\partial \mathcal{L}}{\partial W_d} =
\delta^{(2)} z^T
$$

Since $z$ is scalar:

$$
\frac{\partial \mathcal{L}}{\partial W_d}
=

\begin{bmatrix}
-0.8063 \cdot 0.6457 \
0.5166 \cdot 0.6457
\end{bmatrix}
=

\begin{bmatrix}
-0.5206 \
0.3336
\end{bmatrix}
$$

Backprop to hidden layer:

$$
W_d^T \delta^{(2)} = 0.3(-0.8063) + 0.8(0.5166)
$$

$$
= -0.2419 + 0.4133 = 0.1714
$$

Sigmoid derivative:

$$
\sigma'(0.6)=z(1-z)=0.6457(1-0.6457)\approx 0.2288
$$

So hidden delta:

$$
\delta^{(1)} = 0.1714 \cdot 0.2288 \approx 0.0392
$$

Gradient for encoder weights:

$$
\frac{\partial \mathcal{L}}{\partial W_e} = \delta^{(1)} x^T
$$

$$

0.0392
\begin{bmatrix}
1 & 0
\end{bmatrix}
=

\begin{bmatrix}
0.0392 & 0
\end{bmatrix}
$$

So the update knows:

* decrease first encoder weight slightly
* decoder first weight should increase because gradient is negative
* decoder second weight should decrease because gradient is positive

This is how learning happens numerically.

---

# 24. Practical Training Advice

## 24.1 Normalize inputs

Very important.

If one feature is in range $[0,1]$ and another is in $[0,10000]$, training becomes unstable.

Common approaches:

* min-max scaling to $[0,1]$
* standardization:
  $$
  x' = \frac{x-\mu}{\sigma}
  $$

---

## 24.2 Use small random initialization

Initialize weights with small random values.

Bad initialization can:

* saturate sigmoid
* slow training
* produce unstable gradients

---

## 24.3 Start with shallow architecture

Before deep autoencoders, first make sure this works:

$$
n \rightarrow m \rightarrow n
$$

Then extend.

---

## 24.4 Choose latent dimension carefully

If latent size is too small:

* reconstruction becomes too poor

If too large:

* model may just memorize

---

## 24.5 Watch loss curve

Loss should decrease over epochs.

If loss explodes or does not move:

* learning rate may be too high
* normalization may be poor
* gradients may be wrong

---

# 25. Common Pitfalls

## Pitfall 1: Autoencoder simply copies input

This often happens with too much capacity.

### Fix

* smaller bottleneck
* regularization
* denoising objective

---

## Pitfall 2: Poor reconstruction due to wrong activation

Example:

* using sigmoid output for values not in $[0,1]$

### Fix

Match output activation to data type.

---

## Pitfall 3: Noisy gradients from unnormalized input

### Fix

Normalize data before training.

---

## Pitfall 4: Vanishing gradients with deep sigmoid networks

### Fix

Use ReLU or tanh in hidden layers, careful initialization, maybe batch training.

---

## Pitfall 5: Wrong matrix dimensions

This is the most common implementation bug from scratch.

### Rule

If:

$$
x \in \mathbb{R}^n,\quad z \in \mathbb{R}^m
$$

Then:

* $W_e$ must be $m \times n$
* $W_d$ must be $n \times m$

---

# 26. Visual Shape Summary

## Example architecture: 4 → 2 → 4

Input:

$$
x \in \mathbb{R}^4
$$

Encoder weights:

$$
W_e \in \mathbb{R}^{2 \times 4}
$$

Latent:

$$
z \in \mathbb{R}^2
$$

Decoder weights:

$$
W_d \in \mathbb{R}^{4 \times 2}
$$

Output:

$$
\hat{x} \in \mathbb{R}^4
$$

---

# 27. Linear Autoencoder and Connection to PCA

A very important result:

If the autoencoder is:

* linear
* uses MSE loss
* has one hidden layer
* no nonlinear activation
* and hidden size $m < n$

then it learns a subspace related to **PCA**.

### Meaning

A linear autoencoder behaves similarly to principal component analysis.

### Why this matters

It shows that autoencoders are a nonlinear generalization of classical dimensionality reduction.

---

# 28. When to Use Which Loss

| Data Type                       | Output Activation | Loss       |
| ------------------------------- | ----------------- | ---------- |
| Real-valued continuous features | Identity          | MSE        |
| Values in [0,1]                 | Sigmoid           | BCE or MSE |
| Binary features                 | Sigmoid           | BCE        |
| Standardized continuous values  | Identity          | MSE        |

---

# 29. Minimal Language-Agnostic Implementation Plan

If you want to build it from scratch in any language, follow this order:

## Phase 1: Math utilities

Implement:

* vector add
* vector subtract
* scalar multiply
* dot product
* matrix-vector multiply
* transpose

## Phase 2: Neural utilities

Implement:

* sigmoid / tanh / ReLU
* derivatives
* MSE loss

## Phase 3: Autoencoder forward

Implement:

* encoder
* decoder
* loss

## Phase 4: Backward

Implement:

* output delta
* decoder gradients
* hidden delta
* encoder gradients

## Phase 5: Training loop

Implement:

* epoch loop
* sample loop
* update step
* loss logging

## Phase 6: Validation

Check:

* loss decreases
* reconstruction improves
* latent vectors make sense

---

# 30. Minimal Example Dataset You Can Create Yourself

You do not need fancy data to test.

Create simple 4-dimensional vectors like:

```text
[1, 0, 1, 0]
[1, 1, 1, 0]
[0, 0, 1, 1]
[0, 1, 0, 1]
```

Train a small autoencoder:

$$
4 \rightarrow 2 \rightarrow 4
$$

If implemented correctly:

* loss should decrease
* reconstructions should become closer to inputs

---

# 31. Simple End-to-End Pseudocode

```text
input_dim = 4
latent_dim = 2
learning_rate = 0.1
epochs = 1000

initialize W_enc[2][4], b_enc[2]
initialize W_dec[4][2], b_dec[4]

training_data = [
    [1,0,1,0],
    [1,1,1,0],
    [0,0,1,1],
    [0,1,0,1]
]

for epoch in 1 to epochs:
    total_loss = 0

    for x in training_data:

        # forward
        a1 = W_enc * x + b_enc
        z = sigmoid(a1)

        a2 = W_dec * z + b_dec
        x_hat = sigmoid(a2)

        # loss
        loss = mse(x, x_hat)
        total_loss += loss

        # backward
        delta2 = (x_hat - x) elementwise_mul sigmoid_derivative(x_hat)

        grad_W_dec = outer_product(delta2, z)
        grad_b_dec = delta2

        temp = transpose(W_dec) * delta2
        delta1 = temp elementwise_mul sigmoid_derivative(z)

        grad_W_enc = outer_product(delta1, x)
        grad_b_enc = delta1

        # update
        W_enc = W_enc - lr * grad_W_enc
        b_enc = b_enc - lr * grad_b_enc
        W_dec = W_dec - lr * grad_W_dec
        b_dec = b_dec - lr * grad_b_dec

    print(epoch, total_loss / len(training_data))
```

---

# 32. How to Verify Your Implementation

A correct implementation usually shows:

1. loss decreases steadily
2. reconstructions get closer to inputs
3. weights change over time
4. no dimension mismatch
5. no exploding values or NaNs

### Debug checklist

If training fails, check:

* matrix dimensions
* derivative formulas
* whether loss uses $\hat{x}$ - $x$ or $x$ - $\hat{x}$
* learning rate too large
* activation derivative computed correctly
* normalization of input data

---

# 33. Historical Context

Autoencoders became important because people wanted neural networks to learn useful representations **without labels**.

That is a major idea in machine learning:

* supervised learning needs labels
* unsupervised/self-supervised learning uses structure already present in the data

Autoencoders were among the classic early tools for representation learning.

They later influenced:

* deep belief network pretraining
* denoising feature learning
* variational autoencoders
* modern self-supervised learning ideas

---

# 34. Why Autoencoders Still Matter Today

Even though newer self-supervised methods are very powerful, autoencoders are still valuable because they teach core ideas clearly:

* compression
* reconstruction
* representation learning
* latent spaces
* encoder-decoder structure

If you understand autoencoders deeply, many modern models become easier to understand.

Examples:

* VAE
* sequence-to-sequence models
* transformers with encoder-decoder structure
* masked reconstruction models
* diffusion latent encoders

---

# 35. Summary in One Paragraph

An autoencoder is a neural network that learns to reconstruct its input after first compressing it into a latent representation. The encoder maps input $x$ to a lower-dimensional code $z$, and the decoder maps $z$ back to reconstruction $\hat{x}$. Training minimizes reconstruction loss such as MSE or BCE. The bottleneck forces the model to keep only the most important information, making autoencoders useful for compression, feature learning, denoising, anomaly detection, and latent representation learning.

---

# 36. Quick Revision Sheet

## Core equations

$$
z = \phi(W_e x + b_e)
$$

$$
\hat{x} = \psi(W_d z + b_d)
$$

$$
\mathcal{L}*{\text{MSE}} = \frac{1}{n}\sum*{i=1}^n (x_i - \hat{x}_i)^2
$$

## Backprop

$$
\delta^{(2)} = (\hat{x} - x)\odot \psi'(a^{(2)})
$$

$$
\frac{\partial \mathcal{L}}{\partial W_d} = \delta^{(2)} z^T
$$

$$
\delta^{(1)} = (W_d^T\delta^{(2)})\odot \phi'(a^{(1)})
$$

$$
\frac{\partial \mathcal{L}}{\partial W_e} = \delta^{(1)} x^T
$$

---

# 37. Exercises

## Exercise 1

Implement a linear autoencoder for 2D points compressed to 1D.

## Exercise 2

Implement a sigmoid autoencoder for binary vectors of length 4.

## Exercise 3

Change latent dimension from 2 to 1 and observe reconstruction quality.

## Exercise 4

Add Gaussian noise to inputs and train a denoising autoencoder.

## Exercise 5

Store all latent codes and visualize whether similar inputs get similar codes.

---

# 38. Challenge Questions

1. Why does a bottleneck help the network learn meaningful structure?
2. What happens if latent dimension is larger than input dimension?
3. Why must output activation match the data type?
4. How is a linear autoencoder related to PCA?
5. Why does denoising make features more robust?
6. What is the difference between representation learning and memorization?

---

# 39. Best Mental Model

Think of an autoencoder as:

* a **compressor** plus **reconstructor**
* trained not by labels, but by self-recovery
* forced to discover what matters in the data

So the central question it answers is:

> “What is the smallest useful internal description of this input that still lets me rebuild it?”

That is the heart of the autoencoder.


