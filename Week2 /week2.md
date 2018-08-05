## Week 2
----------

### Lecture 7: Binary Classification

Images are rgb pixels so its a 64x64x3 matrix
we unroll the matrix to a vector or dimension 12288

Notations:
	- each training example is (x,y)
	- x is the input and y is the target label of a single training example
	- m = # training examples
	- n<sub>x</sub> = # input features

It is better to stack m training examples in columns rather than in rows since it makes the neural net computation much easier

X.shape = (n<sub>x</sub>, m); Y.shape = (1,m)


### Lecture 8: Logistic Regression

P(y | x) = sigmoid(wx + b)
sigmoid(z) = 1 / (1 + e^-z)

Basic steps for Logistic Regression:
  $$ z^{(1)} = w^{T}x^{(1)} + b $$
  $$ a^{(1)} = \sigma(z^{(1)}) $$
  $$
### Lecture 9: Cost Function

Gradient descent doesn't work very well if we use a mean square error function
mean square error is : 1/2 (y^ - y) ^ 2

{% raw %}
  $$J(w, b) = 1/m \sum_{i=1}^{m}L(\hat{y}^{(i)}, y^{(i)})$$
  $$= 1/m \sum_{i=1}^{m} y^{(i)}\log \hat{y}^{(i)} + (1-y^{(i)})\log (1 - \hat{y}^{(i)})$$
{% endraw %}


### Lecture 11: Gradient Descent


### Lecture 12: Computation Graph


### Lecture 17: Python Vectorization

Always avoid explicit for loops for performance concerns. As illustrated by the code below:

```python
import numpy as np

a = np.array([1,2,3,4])
print(a)

```

    [1 2 3 4]



```python
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()

print("Vectorized version: " + str(1000 * (toc - tic)))

c = 0
tic = time.time()
for i in range(len(a)):
    c += a[i] * b[i]
toc = time.time()

print("Loop version: " + str(1000 * (toc - tic)))
```

    Vectorized version: 0.8749961853027344
    Loop version: 434.6497058868408



```python

```

### Lecture 19: Vectorizing Logistic Regression

X = (n<sub>x</sub>, m) matrix
w = (n<sub>x</sub>, 1) vector
Z = w<sup>T</sup>X + b
 or Z = [1, n<sub>x</sub>] * [n<sub>x</sub>, m] = [1, m] vector

In python
```python
Z = np.dot(w.T, X) + b
```
'b' is a real number but in python it is automatically extrapolated to a [1, m] vector before the addition operation. This is called **broadcasting**.


### Lecture 20: Vectorizing Logistic Regression's Gradient Computation
{% raw %}
$$ dz^{(1)} = a^{(1)} - y^{(1)} $$
$$ dz^{(2)} = a^{(2)} - y^{(2)} $$
$$ dZ = [dz^{(1)} dz^{(2)} . . . dz^{(m)} ] $$
$$ A = [a^{(1)} a^{(2)} . . . a^{(m)} ] $$
$$ Y = [y^{(1)} y^{(2)} . . . y^{(m)} ] $$
$$ dZ = A - Y $$
{% endraw %}


Loop version:

{% raw %}
$$ J = 0, dw\sub{1} = 0, dw\sub{2}, db = 0 $$
for i = 0 to m:
	$$ dz^{(1)} = a^{(1)} - y^{(1)} $$
	$$ dz^{(2)} = a^{(2)} - y^{(2)} $$
	$$ dZ = [dz^{(1)} dz^{(2)} . . . dz^{(m)} ] $$
	$$ A = [a^{(1)} a^{(2)} . . . a^{(m)} ] $$
	$$ Y = [y^{(1)} y^{(2)} . . . y^{(m)} ] $$

{% endraw %}

Vectorized Version:
```python
Z = np.dot(w.T, X) + b
A = sigmoid(Z)
dZ = A - Y
dW = 1/m np.dot(X, dZ.T)
db = 1/m np.sum(dZ)
w = w - alpha.dW
b = b - alpha.db
```

### Lecture 21: Broadcasting in Python



for 1 training example: cost = y * log A + (1-y) log (1-A)
for m examples: cost = - (np.dot(Y, np.log(A).T) + np.dot((1-Y), np.log(1-A).T)) / m
