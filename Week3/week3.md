### Lecture 25: Neural Networks Representation

3 main components: Input Layer, Hidden Layer, Output Layer
New notation called activations, a
for input layer: a<sup>[0]</sup> = X
for hidden layer: a<sup>[1]</sup>
for output later: a<sup>[2]</sup>

Hidden and output layers has parameters associated with them, w<sup>[1]</sup>, b<sup>[1]</sup> and w<sup>[2]</sup>, b<sup>[2]</sup>, respectively


### Lecture 26: Computing Neural Network Output

Each node has two computations:
	1. $ z\sub_{1}^{[1]} = w\sub_{1}^{[1]T}x + b\sub_{1}^{[1]} $
	2. $ a\sub_{1}^{[1]} = \sigma(z\sub_{1}^{[1]}) $

$ w\sub_{1}^{[1]} $ is a column vector
So if we take it's transpose, it becomes a row vector. We can then stack all the row vectors from $ w\sub_{1}^{[1]} $, $ w\sub_{2}^{[1]} $, $ w\sub_{3}^{[1]} $ and so on to make a $W^{[1]}$ matrix. The input vector x is a column vector with all the input features $[x\sub_{1}, x\sub_{2}, x\sub_{3}]$ and finally b is a column vector of $[b\sub_{1}^{[1]}, b\sub_{2}^{[1]}, b\sub_{3}^{[1]}, b\sub_{4}^{[1]}]$.

So with this we get $z^{[1]} = W^{[1]}x + b^{[1]}$ for the first step of the computation and then we take a sigmoid or other (ReLU) activation of $z^{[1]}$ to get $a^{[1]}$ as a column vector of all the activations of layer 1 (hidden layer). In other words: $a^{[1]} = \sigma(z^{[1]})$

Similarly for the next layer,
$z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}$
$a^{[2]} = \sigma(z^{[2]})$


### Lecture 27: Vectorizing NN Output Computation

m training examples
loop version =>
for i = 1 to m:
	z[1](i) = w[1]x(i) + b[1]
	a[1](i) = sigma(z[1](i))
	z[2](i) = w[2]a[1](i) + b[2]
	a[2](i) = sigma(z[2](i))

X = (nx, m) matrix
Stacking all a[1](i) column vectors for all m training examples givs us A
Similarly, stacking all z[1](i) in columns for m examples gives us Z

vectorized version =>
	Z[1] = W[1]X + b[1]
	A[1] = sigma(Z[1])
	Z[2] = W[2]A[1] + b[2]
	A[2] = sigma(Z[2])


### Lecture 29: Activation Functions

tanh is another example of an activation function that works better than sigmoid. tanh is just a shifted version of sigmoid so that it's y ranges from (-1, 1)
$ tanh = \frac{e^z - e^{-z}}{e^z + e^{-z}} $

Exception to use sigmoid activation is in the output layer where the output should be 0 or 1 so it makes sense, but everywhere else tanh is superior.

Caveat of tanh and sigmoid functions is that with large value of z, the slope of the functions becomes very small so it greatly slows down gradient descent. To help with this, we have ReLU activation.
a = max(0, z)

Leaky ReLU => a = max(0.01z, z)


### Lecture 31: Derivatives of Activation Functions

a = sigmoid(z)
a' = a (1-a)

a = tanh(z)
a' = 1 - tanh(z)^2

a = max(0, z)  #ReLU
a' = 0 if z <= 0 and a' = 1 if z > 0

a = max(0.01z, z)  #Leaky ReLU
a1 = 0.01 if z <= 0 and a' = 1 if z > 0


### Lecture 32: Gradient Descent for Neural Networks

number of nodes in each layer denoted by $ n\sub_{x} = n^{[0]} (input layer), n^{[1]} (hidden layer), n^{[2]} = 1 (output layer)$

Parameters: $ w^{[1]}, b^{[1]}, w^{[2]}, b^{[2]} $
Dimensions: $ (n^{[1]}, n^{[0]}), (n^{[1]}, 1), (n^{[2]}, n^{[1]}), (n^{[2]}, 1) $

For binary classification we can use the same loss function as we used for logistic regression {-(ylogy^ + (1-y)log(1-y^))}


### Lecture 34: Random Initialization

If initial weights are identical or all zero, due to the inherent symmetry of the neural network, the gradients will always come out to be the same for all rows of w[1], w[2] etc. Due to this, no matter how long we run gradient descent, there won't be any convergence and all the w[1], w[2] parameters would be symmetrical so there would be no point to having multiple hidden units cause they'll be computing exactly the same function.

This effect does not happen with b[1], b[2] because there is no problem with this symmetry breaking for b parameters

w[1] = np.random.randn((n[1], n[0])) * 0.01

0.01 constant is added to make sure the random numbers are relative small so that in the activation functions we don't start with very large values for z[1] which slows down gradient descent


