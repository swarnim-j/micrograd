import math
import random

class Value:

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._prev = set(_children)
        self._backward = lambda: None # stores function that's why lambda needed
        self._op = _op
        self.grad = 0


    def __add__(self, other):

        other = other if isinstance(other, Value) else Value(other) # handles case where you add a non-Value object
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1
            other.grad += 1

        out._backward = _backward # storing the function '_backward()'

        return out

    def __mul__(self, other):

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting integers and floating point numbers"

        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad = other * (self.data**(other - 1))

        out._backward = _backward

        return out

    def __neg__(self):
        return -1 * self.data

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**-1

    def __rmul__(self, other):
        return other * self

    def __radd__(self, other):
        return other + self

    def __rsub__(self, other):
        return other + (-self)

    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self):
        return f'Value(data={self.data}, grad={self.grad})'

    def tanh(self):
        out = Value((math.exp(self.data) - 1) / (math.exp(self.data) + 1), (self,), 'tanh')

        def _backward():
            self.grad += 1 - (out.data)**2

        out._backward = _backward

        return out      

    def backward(self):

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child) # recursively adds children to the visited set
                topo.append(v)
        build_topo(self)
        
        self.grad = 1
        for v in reversed(topo):
            v._backward()  

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True): # nin weights for each neuron corresponding to weight of connection with each neuron of previous layer
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = 0
        self.nonlin = nonlin

    def __call__(self, x): # x is an input vector
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'tanh' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out # len(out) == 1 checks if neuron is in output layer

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"       

class MLP(Module):

    def __init__(self, nin, nouts):
        size = [nin] + nouts
        self.layers = [Layer(l1, l2) for l1, l2 in zip(size, size[1:])]

    def __call__(self, x):
        for layer in self.layers: # x changes to become the output of the next layer until it reaches the final layer and becomes the final output
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"