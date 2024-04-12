


## Tensor arithmetic

Tensors are a generic abstraction of n-dimensional arrays of numbers with fixed dimensions. The number of dimensions
together with the tensor's size in each dimension is called the tensor's *geometry*.

The term *scalar* is used for a 0-dimensional tensor, *vector* for one dimension, and *matrix* for a two-dimensional 
tensor. These are colloquial terms for tensors of a given dimension, but they are not reflected in the type system.
They are just regular tensors at that level.

Regular arithmetic operations are defined per element, and for binary operations, identical geometry is required. 

NB: 'Regular' matrix multiplication. 'applying' a matrix to a vector or calculating the scalar product between two
vectors are primitive operation in their own right, and they are different from 'multiplication' 

### Lifting

TODO implicit lifting? Or applying the operation on the lower dimensions while iterating over the higher ones?

Sometimes a lower-dimension tensor is used where a higher-dimension tensor is required, e.g. multiplying a matrix
with a scalar. The semantics for this is to add dimensions to the lower-dimension tensor to match the other's
geometry. 

This is called *lifting*, and it is done implicitly in arithmetic operations (i.e. the operation is performed 'as if'
one tensor was lifted, without requiring the lifted tensor to be materialized). There is also an API for lifting
tensors explicitly.

Lifting is only possible if the lower-dimension tensor's geometry is identical to the inner dimensions of the target
geometry, e.g.

* `[]` can be lifted to `[1]`, `[2]` or `[35]`
* `[]` can be lifted to `[1][2]` or `[10][5][25]`
* `[2]` can be lifted to `[5][2]`, `[10][2]` or `[4][9][2]` but not to `[2][5]`, `[5][4]` or `[10][1]`
* `[3][4]` can be lifted to `[2][3][4]` but not to `[3][4][2]`

NB: Implicit lifting is done regardless of the tensor's position in a binary operation. In the sum `a+b`, both a and b
can be potentially lifted to match the other tensor's geometry.


## notes for documentation
* column major
* t.r()
* multi threading: RwLock, 
* env
* global id
* tracker
* op, expr
* t.mult() / t._mult()
* cyclic, version
* reshaping

### optimization
* `profile.release` -> 5-10% speedup (for pytorch initial example)
* `RUSTFLAGS="-C target-cpu=native" cargo ...` -> another 15-20% speedup (for pytorch initial example) 

## Links
* Gradient Descent: https://www.youtube.com/watch?v=tIeHLnjs5U8&t=13s
* GPUs vs CPUs for training: https://gcore.com/blog/deep-learning-gpu/
* library for lin alg on accelerated hardware  https://github.com/LaurentMazare/xla-rs
