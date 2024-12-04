Optimizer
class Optimizer(schedulers=None)
The base class for all optimizers. It allows us to implement an optimizer on a per-parameter basis and apply it to a parameter tree.

Attributes

Optimizer.state

The optimizer's state dictionary.

Methods

Optimizer.apply_gradients(gradients, parameters)

Apply the gradients to the parameters and return the updated parameters.

Optimizer.init(parameters)

Initialize the optimizer's state

Optimizer.update(model, gradients)

Apply the gradients to the parameters of the model and update the model with the new parameters.

=========================================================

## mlx.utils.tree_map
tree_map(fn: Callable, tree: Any, *rest: Any, is_leaf: Callable | None = None) → Any
Applies fn to the leaves of the Python tree tree and returns a new collection with the results.

If rest is provided, every item is assumed to be a superset of tree and the corresponding leaves are provided as extra positional arguments to fn. In that respect, tree_map() is closer to itertools.starmap() than to map().

The keyword argument is_leaf decides what constitutes a leaf from tree similar to tree_flatten().

import mlx.nn as nn
from mlx.utils import tree_map

model = nn.Linear(10, 10)
print(model.parameters().keys())
# dict_keys(['weight', 'bias'])

# square the parameters
model.update(tree_map(lambda x: x*x, model.parameters()))
Parameters
:
fn (callable) – The function that processes the leaves of the tree.

tree (Any) – The main Python tree that will be iterated upon.

rest (tuple[Any]) – Extra trees to be iterated together with tree.

is_leaf (callable, optional) – An optional callable that returns True if the passed object is considered a leaf or False otherwise.

Returns
:
A Python tree with the new values returned by fn.

=========================================================

## Pure Functions
Compiled functions are intended to be pure; that is they should not have side effects. For example:

state = []

@mx.compile
def fun(x, y):
    z = x + y
    state.append(z)
    return mx.exp(z)

fun(mx.array(1.0), mx.array(2.0))
# Crash!
print(state)
After the first call of fun, the state list will hold a placeholder array. The placeholder does not have any data; it is only used to build the computation graph. Printing such an array results in a crash.

You have two options to deal with this. The first option is to simply return state as an output:

state = []

@mx.compile
def fun(x, y):
   z = x + y
   state.append(z)
   return mx.exp(z), state

 _, state = fun(mx.array(1.0), mx.array(2.0))
 
 print(state)
# Prints [array(3, dtype=float32)]

In some cases returning updated state can be pretty inconvenient. Hence, compile() has a parameter to capture implicit outputs:

from functools import partial

state = []

# Tell compile to capture state as an output
@partial(mx.compile, outputs=state)
def fun(x, y):
    z = x + y
    state.append(z)
    return mx.exp(z), state

fun(mx.array(1.0), mx.array(2.0))

print(state)
# Prints [array(3, dtype=float32)]

This is particularly useful for compiling a function which includes an update to a container of arrays, as is commonly done when training the parameters of a mlx.nn.Module.

Compiled functions will also treat any inputs not in the parameter list as constants. For example:

state = [mx.array(1.0)]

@mx.compile
def fun(x):
    return x + state[0]

print(fun(mx.array(1.0)))
# Prints array(2, dtype=float32)

# Update state
state[0] = mx.array(5.0)


print(fun(mx.array(1.0)))
# Still prints array(2, dtype=float32)

In order to have the change of state reflected in the outputs of fun you again have two options. The first option is to simply pass state as input to the function. In some cases this can be pretty inconvenient. Hence, compile() also has a parameter to capture implicit inputs:

from functools import partial
state = [mx.array(1.0)]

# Tell compile to capture state as an input
@partial(mx.compile, inputs=state)
def fun(x):
    return x + state[0]

# Prints array(2, dtype=float32)
print(fun(mx.array(1.0)))

# Update state
state[0] = mx.array(5.0)

# Prints array(6, dtype=float32)
print(fun(mx.array(1.0)))
Compiling Training Graphs
This section will step through how to use compile() with a simple example of a common setup: training a model with mlx.nn.Module using an mlx.optimizers.Optimizer with state. We will show how to compile the full forward, backward, and update with compile().

To start, here is the simple example without any compilation:

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# 4 examples with 10 features each
x = mx.random.uniform(shape=(4, 10))

# 0, 1 targets
y = mx.array([0, 1, 0, 1])

# Simple linear model
model = nn.Linear(10, 1)

# SGD with momentum
optimizer = optim.SGD(learning_rate=0.1, momentum=0.8)

def loss_fn(model, x, y):
    logits = model(x).squeeze()
    return nn.losses.binary_cross_entropy(logits, y)

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# Perform 10 steps of gradient descent
for it in range(10):
    loss, grads = loss_and_grad_fn(model, x, y)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
To compile the update we can put it all in a function and compile it with the appropriate input and output captures. Here’s the same example but compiled:

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from functools import partial

# 4 examples with 10 features each
x = mx.random.uniform(shape=(4, 10))

# 0, 1 targets
y = mx.array([0, 1, 0, 1])

# Simple linear model
model = nn.Linear(10, 1)

# SGD with momentum
optimizer = optim.SGD(learning_rate=0.1, momentum=0.8)

def loss_fn(model, x, y):
    logits = model(x).squeeze()
    return nn.losses.binary_cross_entropy(logits, y)

# The state that will be captured as input and output
state = [model.state, optimizer.state]

@partial(mx.compile, inputs=state, outputs=state)
def step(x, y):
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model, x, y)
    optimizer.update(model, grads)
    return loss

# Perform 10 steps of gradient descent
for it in range(10):
    loss = step(x, y)
    # Evaluate the model and optimizer state
    mx.eval(state)
    print(loss)
Note

If you are using a module which performs random sampling such as mlx.nn.Dropout(), make sure you also include mx.random.state in the state captured by compile(), i.e. state = [model.state, optimizer.state, mx.random.state].

Note

For more examples of compiling full training graphs checkout the MLX Examples GitHub repo.

Transformations with Compile
In MLX function transformations are composable. You can apply any function transformation to the output of any other function transformation. For more on this, see the documentation on function transforms.

Compiling transformed functions works just as expected:

grad_fn = mx.grad(mx.exp)

compiled_grad_fn = mx.compile(grad_fn)

# Prints: array(2.71828, dtype=float32)
print(grad_fn(mx.array(1.0)))

# Also prints: array(2.71828, dtype=float32)
print(compiled_grad_fn(mx.array(1.0)))
Note

In order to compile as much as possible, a transformation of a compiled function will not by default be compiled. To compile the transformed function simply pass it through compile().

You can also compile functions which themselves call compiled functions. A good practice is to compile the outer most function to give compile() the most opportunity to optimize the computation graph:

@mx.compile
def inner(x):
    return mx.exp(-mx.abs(x))

def outer(x):
    inner(inner(x))

# Compiling the outer function is good to do as it will likely
# be faster even though the inner functions are compiled
fun = mx.compile(outer)


=========================================================

mlx.core.value_and_grad
value_and_grad(fun: Callable, argnums: int | list[int] | None = None, argnames: str | list[str] = []) → Callable
Returns a function which computes the value and gradient of fun.

The function passed to value_and_grad() should return either a scalar loss or a tuple in which the first element is a scalar loss and the remaining elements can be anything.

import mlx.core as mx

def mse(params, inputs, targets):
    outputs = forward(params, inputs)
    lvalue = (outputs - targets).square().mean()
    return lvalue

# Returns lvalue, dlvalue/dparams
lvalue, grads = mx.value_and_grad(mse)(params, inputs, targets)

def lasso(params, inputs, targets, a=1.0, b=1.0):
    outputs = forward(params, inputs)
    mse = (outputs - targets).square().mean()
    l1 = mx.abs(outputs - targets).mean()

    loss = a*mse + b*l1

    return loss, mse, l1

(loss, mse, l1), grads = mx.value_and_grad(lasso)(params, inputs, targets)
Parameters
:
fun (Callable) – A function which takes a variable number of array or trees of array and returns a scalar output array or a tuple the first element of which should be a scalar array.

argnums (int or list(int), optional) – Specify the index (or indices) of the positional arguments of fun to compute the gradient with respect to. If neither argnums nor argnames are provided argnums defaults to 0 indicating fun’s first argument.

argnames (str or list(str), optional) – Specify keyword arguments of fun to compute gradients with respect to. It defaults to [] so no gradients for keyword arguments by default.

Returns
:
A function which returns a tuple where the first element is the output of fun and the second element is the gradients w.r.t. the loss.

Return type
:
Callable
=========================================================

mlx.nn.value_and_grad
value_and_grad(model: Module, fn: Callable)
Transform the passed function fn to a function that computes the gradients of fn wrt the model’s trainable parameters and also its value.

Parameters
:
model (Module) – The model whose trainable parameters to compute gradients for

fn (Callable) – The scalar function to compute gradients for

Returns
:
A callable that returns the value of fn and the gradients wrt the trainable parameters of model