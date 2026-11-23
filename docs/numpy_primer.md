# NumPy primer for reading PyAstroDyn-edu

Every function of the library works for **one state or many states in one
call**. That is done with six NumPy idioms, and only six. This page explains
each with a small example. Comments in the code refer to them as "primer 1"
to "primer 6".

Shapes used throughout:

| Quantity | One state | n states |
|---|---|---|
| Scalar (distance, eccentricity, an angle) | a number | shape `(n,)` |
| Vector (position, velocity) | shape `(3,)` | shape `(n, 3)`, one **row** per state |
| Rotation matrix | shape `(3, 3)` | shape `(n, 3, 3)`, a stack of n matrices |

If you prefer to start without any of this, read the module
`pyastrodyn_edu/single_state.py` first: it contains the three hardest
algorithms written for a single state, with ordinary `if` statements.

```python
import numpy as np
```

## 1. Broadcasting: one value combined with many

Arithmetic between arrays works element by element. When one operand is a
single number, NumPy uses it for every element.

```python
e = 0.2
f = np.array([0.0, 1.0, 2.0])      # three true anomalies
1.0 + e * np.cos(f)                # three results, one per true anomaly
```

`np.broadcast_arrays(a, e, f)` does this explicitly: it returns the inputs
stretched to a common shape. The library calls it at the top of a function
so that the rest of the code can assume "all inputs have the same shape".

## 2. The last axis: `axis=-1`

The three components of a vector are always on the **last** axis. For one
vector that is the only axis; for n vectors it is the second. Writing
`axis=-1` ("the last one") makes one line of code serve both.

```python
r_N = np.array([[7000.0, 0.0, 0.0],
                [0.0, 8000.0, 0.0]])      # two positions, shape (2, 3)
np.linalg.norm(r_N, axis=-1)              # array([7000., 8000.]): one length per row
np.linalg.norm(r_N[0], axis=-1)           # 7000.0: the same call for one vector
```

The helpers `norm(a)` and `dot(a, b)` in `_util.py` are exactly this.

## 3. A scalar times a vector: the trailing axis

For n states, a scalar has shape `(n,)` and a vector `(n, 3)`. NumPy cannot
multiply them directly: it would try to pair the n scalars with the three
components. Adding a trailing axis of length 1 turns the scalars into a
column of shape `(n, 1)`, and then each row is multiplied by its own scalar.

```python
r = np.array([7000.0, 8000.0])            # shape (2,)
r_N / r[..., np.newaxis]                  # shape (2, 3): each row divided by its length
```

The helper `col(s)` does `s[..., np.newaxis]`. Whenever you see `col(...)`
in the code, read it as "this scalar multiplies (or divides) a vector".

## 4. The ellipsis `...` and stacking

`...` in an index means "all the leading axes, whatever they are".

```python
r_N[..., 2]        # third component: a number for one vector, shape (n,) for n vectors
x_N[..., :3]       # first three numbers of every state: the position
C[..., 0, 1]       # entry (0, 1) of one matrix, or of every matrix in a stack
```

`np.stack` builds a new axis out of several arrays:

```python
np.stack((x, y, z), axis=-1)       # three scalars per state -> one vector per state
np.stack((u_N, v_N, w_N), axis=-2) # three vectors per state -> the ROWS of one matrix per state
```

## 5. `einsum`: a matrix times a vector, for every state

For one matrix and one vector, `C_BA @ v_A` is the product. For n matrices
and n vectors we want "matrix k times vector k". `np.einsum` writes the
product out by its indices:

```python
np.einsum("...ij,...j->...i", C_BA, v_A)
```

Read it as: *result component i = sum over j of C[i, j] · v[j], and the same
for every state (the `...`)*. This single line is the function
`change_basis(C_BA, v_A)`.

## 6. Masks and `np.where`: an `if` for every state

A comparison on an array gives one `True`/`False` per state, called a mask.
`np.where(mask, x, y)` takes `x` where the mask is `True` and `y` elsewhere.
It is the many-states form of `if mask: x else: y`.

```python
e = np.array([0.3, 0.0, 0.1])
circular = e < 1e-10                       # array([False,  True, False])
argp = np.where(circular, 0.0, argp)       # 0 for the circular orbit, unchanged for the others
```

Two things to know:

- **Both branches are computed for every state**, and then one is kept. That
  is why the code sometimes guards a division that will be discarded anyway:
  `np.where(equatorial, 1.0, node)` replaces a zero divisor by 1 in the states
  whose result is not used.
- When the mask selects between **vectors**, it needs the trailing axis of
  primer 3: `np.where(col(circular), node_hat_N, e_N)`.

## One more thing: `out(x)`

NumPy sometimes returns a single value wrapped as a "zero-dimensional
array". The helper `out(x)` unwraps it, so that one state in gives plain
numbers out.
