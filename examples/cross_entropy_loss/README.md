# Cross Entropy Loss and LogSumExp

## Introduction

Cross entropy loss is a widely used loss function in machine learning, especially in classification tasks. It quantifies the disparity between two probability distributions, typically the predicted probabilities and the actual label distribution.

This README provides an overview of the mathematical formulation of cross entropy loss and how the LogSumExp trick enhances computational stability when dealing with exponentials.

## Cross Entropy Loss Forward

The cross entropy loss $L$ for a single example is given by:

$L = - \sum_{i} y_i \log(p_i)$

Where:
- $y_i$ represents the actual label (0 or 1).
- $p_i$ denotes the predicted probability of class $i$.

For binary classification, when $y = 0$, the loss simplifies to 0, and when $y = 1$, the loss simplifies to $-\log(p)$.

## LogSumExp Trick

The LogSumExp trick is a numerical stability technique used when computing sums of exponentials, mitigating potential overflow or underflow errors.

### Mathematical Explanation

Consider the expression for LogSumExp:

$\log(\sum_{i} \exp(x_i)) = c + \log(\sum_{i} \exp(x_i - c))$

Where:
- $x_i$ are the input values.
- $c$ is a constant chosen to stabilize the computation, typically the maximum of $x_i$.

By subtracting the maximum value from each $x_i$ before exponentiating, we ensure that the largest exponent is 1, preventing overflow issues.

### Example

Suppose we have input values $x = \{3, 1, 2\}$. To compute the sum of exponentials, we first subtract the maximum value, which is 3:

$c = \max(x) = 3$

$x' = \{0, -2, -1\}$

Then, we compute the sum of exponentials:

$\exp(x'_1) + \exp(x'_2) + \exp(x'_3)$

$= \exp(0) + \exp(-2) + \exp(-1)$

$= 1 + \frac{1}{e^2} + \frac{1}{e}$

$\approx 1 + 0.135 + 0.368$

$\approx 1.503$

Finally, we take the logarithm:

$\log(1.503)$

$\approx 0.407$

Thus, using the LogSumExp trick, we've computed the sum of exponentials in a stable manner.

## Chunked Cross Entropy Forward

Chunked cross entropy forward is a technique used to efficiently compute cross entropy loss with large vocabularies by dividing them into smaller chunks, thereby reducing computational complexity.

### Mathematical Explanation

- $V$ : Vocabulary size.
- $C$ : Number of chunks.
- $\text{logits}_{ij}$ : Logits for word $i$ in chunk $j$.
- $\text{logsumexp}_j$ : Logsumexp for chunk $j$.
- $\text{chunk\_sum}$ : Sum of logsumexp values across all chunks.

1. **Logsumexp for Each Chunk**:

   For each chunk $j$:
   $\text{logsumexp}_j = \log \left( \sum_{i=1}^{V/C} e^{\text{logits}_{ij}} \right)$

2. **Chunk Sum**:

   Compute the sum of logsumexp values across all chunks:
   $\text{chunk\_sum} = \log \left( \sum_{j=1}^{C} e^{\text{logsumexp}_j} \right)$

3. **Final Computation**:

   The final cross entropy loss for each word $i$ in chunk $j$ is computed as:
   $CE_{ij} = \text{chunk\_sum} - \text{logits}_{ij}$


### Mathematical Calculation Example

Suppose we have a vocabulary of 12 words divided into 3 equal chunks, each containing 4 words. We'll denote the logits for each word in each chunk as follows:

Chunk 1:
- Word 1: $\text{logits}_{11} = 2.0$
- Word 2: $\text{logits}_{12} = 1.5$
- Word 3: $\text{logits}_{13} = 1.8$
- Word 4: $\text{logits}_{14} = 2.2$

Chunk 2:
- Word 5: $\text{logits}_{21} = 1.7$
- Word 6: $\text{logits}_{22} = 2.8$
- Word 7: $\text{logits}_{23} = 2.0$
- Word 8: $\text{logits}_{24} = 3.3$

Chunk 3:
- Word 9: $\text{logits}_{31} = 3.2$
- Word 10: $\text{logits}_{32} = 2.4$
- Word 11: $\text{logits}_{33} = 1.9$
- Word 12: $\text{logits}_{34} = 2.1$

Let's follow the steps outlined for Chunked Cross Entropy Forward:

### Step 1: Logsumexp for Each Chunk

For each chunk $j$, we compute the logsumexp:

Chunk 1:
$\text{logsumexp}_1 = \log \left( \sum_{i=1}^{4} e^{\text{logits}_{i1}} \right)$

$\text{logsumexp}_1 = \log \left( e^{2.0} + e^{1.5} + e^{1.8} + e^{2.2} \right)$

$\text{logsumexp}_1 = \log \left( 7.389 + 4.481 + 6.049 + 9.025 \right)$

$\text{logsumexp}_1 \approx \log(27.944) \approx 3.328$

Similarly, we compute logsumexp for Chunk 2 and Chunk 3.

Chunk 2:
$\text{logsumexp}_2 = \log \left( \sum_{i=5}^{8} e^{\text{logits}_{i2}} \right)$

Chunk 3:
$\text{logsumexp}_3 = \log \left( \sum_{i=9}^{12} e^{\text{logits}_{i3}} \right)$

### Step 2: Chunk Sum

We compute the sum of logsumexp values across all chunks:

$\text{chunk\_sum} = \log \left( e^{\text{logsumexp}_1} + e^{\text{logsumexp}_2} + e^{\text{logsumexp}_3} \right)$

$\text{chunk\_sum} = \log \left( e^{3.328} + e^{... [truncated]

## Cross Entropy Backward

Cross Entropy Backward is a technique used to compute the gradient of the cross entropy loss with respect to the logits in neural network training. It's a crucial step in backpropagation for updating model parameters during optimization.

### Mathematical Explanation


The gradient of the cross entropy loss \(CE_i\) with respect to the logits \(x\) is denoted as $\frac{dC}{dx}$, where \(C\) is the cross entropy loss and \(i\) represents a particular example in the dataset.

The cross entropy loss \(CE_i\) is given by:
$CE_i = -y \log(P) = y \cdot \left( \log \left[ \sum \left( \exp(x) \right) \right] - x \right)$

To compute the gradient $\frac{dC}{dx}$, we differentiate \(CE_i\) with respect to \(x\):
$\frac{dC}{dx} = \frac{d}{dx} \left( y \cdot \log \left[ \sum \left( \exp(x) \right) \right] - x \cdot y \right)$

Using the derivative of the logsumexp function, which is given by:
$\frac{d}{dx} \text{logsumexp} = \frac{\exp(x)}{\sum \left( \exp(x) \right)} = \text{softmax}(x)$

We rewrite $\frac{dC}{dx}$ as:
$\frac{dC}{dx} = y \cdot \frac{\exp(x)}{\sum \left( \exp(x) \right)} - \frac{d}{dx} (x \cdot y)$

Using the derivative of \(x \cdot y\), which is \(y\) if \(y\) is a constant with respect to \(x\), and 0 if \(y\) is a variable, we further simplify $\frac{dC}{dx}$:
$\frac{dC}{dx} = y \cdot \exp \left[ \log \left( \frac{\exp(x)}{\sum \left( \exp(x) \right)} \right) \right] - \frac{d}{dx} (x \cdot y)$

$\frac{dC}{dx} = y \cdot \exp \left[ x - \text{logsumexp} \right] - \frac{d}{dx} (x \cdot y)$

The gradient $\frac{dC}{dx}$ has different expressions depending on the value of \(y\):
- If \(y = 0\), then $\frac{dC}{dx} = 0$.
- If \(y = 1\) and \(x\) is the label, then $\frac{dC}{dx} = \exp \left[ x - \text{logsumexp} \right] - 1$.
- If \(y = 1\) and \(x\) is not the label, then $\frac{dC}{dx} = \exp \left[ x - \text{logsumexp} \right]$.

