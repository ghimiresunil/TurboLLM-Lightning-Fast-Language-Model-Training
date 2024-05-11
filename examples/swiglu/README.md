# Sigmoid-Weighted Linear Unit (SwiLU) 

Large language models like Google's PaLM 2 and Meta's LLaMA leverage SwiGLU to enhance the effectiveness of their Transformer architecture. SwiGLU, a variant of Gated Linear Units (GLU) activation functions, emerged from research by Noam Shazeer (formerly of Google Brain and co-author of the foundational Transformer paper).

Shazeer's exploration, detailed in the paper "GLU Variants Improve Transformer," demonstrated that specific GLU variants surpassed the performance of prior state-of-the-art activations like ReLU and GELU in language tasks. This article delves into the GLU variants experimented with in the paper, including ReGLU, GEGLU, and the impactful SwiGLU.

## 1. FFN with ReLU Activation

The core building block of Transformer models is a series of identical layers stacked together. These layers alternate between two key components:

- Multi-head attention: This mechanism allows the model to focus on specific parts of the input sequence when processing information.
- Position-wise feed-forward network (FFN): This network adds non-linearity and helps the model learn more complex relationships within the sequence data.

Both the encoder and decoder parts of the Transformer architecture utilize these alternating layers to achieve their respective goals.

FFN, or Feed-Forward Network layers, are a fundamental building block within the Transformer architecture. They act like miniature neural networks themselves, performing a specific task. Each FFN layer consists of two key parts:

- Linear Transformations: Imagine multiplying the input data by a weight matrix and adding a bias term. This linear transformation essentially creates a new, refined version of the data.
- Non-Linear Activation Function: Data in the real world rarely behaves in perfectly straight lines. The activation function, often ReLU (Rectified Linear Unit) in the original Transformer, injects some non-linearity into the data. This allows the network to capture more complex relationships within the data.

By combining these linear transformations with a non-linear activation, FFN layers enable the Transformer to learn intricate patterns and relationships within the input data. This processing power is crucial for various tasks the Transformer excels at, such as machine translation and text summarization.

$$FNN(x, W_1, W_2, b_1, b_2 = ReLU(xW_i + b_1)W_2+b_2$$

where, $ReLU(x) = max(0,x)$

In their experiments, the authors remove the bias term from the FFN layer and refer to it as $FFN_{ReLU}$. This simplification aligns with the approach taken in the T5 paper, co-authored by Noam Shazeer. 

Consequently, the equations for the FFN layer are modified.

$$FNN(x, W_1, W_2 = ReLU(xW_i)W_2$$

> Note: Building upon the original Transformer paper, researchers have proposed alternative activation functions to replace ReLU.

## 2. FFN with GELU Activation

While the original Transformer architecture relied on the ReLU (Rectified Linear Unit) activation function, advancements have been proposed. The paper "Gaussian Error Linear Units (GELUs)" introduces GELU as a smoother alternative to ReLU. 

As the chart illustrates, GELU closely approximates ReLU but with a more continuous and differentiable nature. This characteristic, crucial for training deep neural networks, allows for smoother gradients and potentially faster convergence. 

Consequently, the FFN layer can be reformulated to incorporate the GELU activation function.

$$FNN(x, W_1, W_2 = GELU(xW_i)W_2$$

where, $GELU(x) = x\phi(x)$ and $\phi(x) is the cumulative distribution function of the standard normal distribution.

> Note: Further enhance computational efficiency, the GELU paper employs approximations of the standard normal distribution's cumulative distribution function (cdf).

## 3. FFN with Swish Activation

The quest for improved activation functions in Transformer architectures extends beyond GELU. The paper "Swish: a Self-Gated Activation Function" proposes another contender: the Swish function. 

Similar to GELU, Swish offers a smoother approximation of ReLU compared to the original "hard" threshold. 

As the chart depicts, Swish achieves this smoothness while maintaining a crucial advantage - a non-zero gradient for negative values. This property can improve the flow of information through the network during training, potentially leading to better performance.

> Note: As with GELU, it is a non-linear function that is differentiable everywhere. Swish has the beta parameter, which controls the shape of the function.

So, the FFN with Swish activation becomes:

$$FNN_{Swish}(x, W_i, W_2 = Swish(xW_1)W_2$$

where, $Swish_{\beta} = x{\sigma}(\beta{x})$

> Note: They use $\beta = 1$ in their experiments.

## 4. GLU and Variants


While often categorized as an activation function, GLU (Gated Linear Unit) acts differently. It's a neural network layer comprised of two parts: a linear transformation and a gating mechanism. 

This gating mechanism, typically a sigmoid function, controls how much information from the linear transformation gets passed on. Essentially, it acts like a valve, regulating the flow of information within the network.

$$GLU(x,W,V,b,c) = \sigma(xW+b)\otimes(xV+c)$$

where $\sigma$ is the sigmoid function and $\otimes$ is the element-wise product. The sigmoid function is the gating mechanism, similar to the gates in LSTM.

We can deﬁne GLU variants using other activation functions than the sigmoid function.

### 4.1. Bilinear activation
The bilinear layer, a variant of GLU, excludes the sigmoid function, instead comprising a bilinear transformation succeeded by an element-wise product.

$$Bilinear(x,W,V,b,c) = (xW+b)\otimes(xV+c)$$


### 4.2. ReGLU activation

ReGLU is a GLU variant that uses ReLU as the activation function.

$$ReGLU(x,W,V,b,c) = ReLU(xW+b)\otimes(xV+c)$$

### 4.3 GEGLU activation

GEGLU is a GLU variant that uses GELU as the activation function.

$$GEGLU(x,W,V,b,c) = GELU(xW+b)\otimes(xV+c)$$

### 4.4 SwiGLU activation

SwiGLU is a GLU variant that uses Swish as the activation function.

$$SwiGLU(x,W,V,b,c) = Swish_1(xW+b)\otimes(xV+c)$$

### 5 FFN and GLU variants

In the paperm they use the GLU variants without bias

$GLU(x,W,V) = \sigma(xW)\otimes xV$

$Bilinear(x,W,V) = xW\otimes xV$

$ReGLU(x,W,V) = ReLU(xW)\otimes xV$

$GEGLU(x,W,V) = GELU(xW) \otimes xV$

$SwiGLU(x,W,V) = Swish_1(xW) \otimes xV$

And they use the GLU Variants in the FNN Layers,

$ FNN_{Bilinear}(x, W, V, W_2) = Bilinear(x,W,V)W_2$

$ FNN_{ReGLU}(x, W, V, W_2) = ReGLU(x,W,V)W_2$

$ FNN_{GEGLU}(x, W, V, W_2) = GEGLU(x,W,V)W_2$

$ FNN_{SwiGLU}(x, W, V, W_2) = SwiGLU(x,W,V)W_2$

All these FFN layers have three weight matrices, $W$ , $V$ , and $W_2$, whereas the original FFN layer has two weight matrices, $W_1$ and $W_2$. 

To keep the number of parameters and the amount of the computation the same, they reduce the size of hidden units $d_{ff}$ (the second dimension of $W$ and $V$ and the ﬁrst dimension of $W_2$) by a factor of $\frac{2}{3}$. So, it makes the number of parameters in the three weight matrices comparable to the two-weight matrix version.















