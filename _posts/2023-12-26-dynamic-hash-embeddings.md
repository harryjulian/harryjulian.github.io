---
layout: post
title: Dynamic Hash Embeddings for Transformers
---

Model compression methods for Transformers have been gaining traction for a while. The open-source community is pushing consumer hardware to it’s limits, and you can now just about [run LLMs on your laptop](https://github.com/ggerganov/ggml). The question remains as to what extent we can push large models to run on increasingly smaller edge devices; have you ever wanted to ask your garmin to to write a rap about post-modernism in the style of Snoop Dogg? Probably not, but it might be possible soon.

I came across [this](https://arxiv.org/abs/2310.20144) paper from Apple MLR recently, which details a model compression technique I hadn't come across - dynamically computed embeddings.

## Why consider using Dynamic Embeddings?

The main observation of the authors, citing previous work, is as follows:

- All transformer based language models contain an embedding layer, which maps tokenized inputs to a `D` dimensional vector embedding representation of the original sequence.
- Ordinary embedding layers in transformers act as lookup tables the respective embedding vector for a particular token. Given a token `T`, we select the row/col at index `T` of the matrix which gives us the embedding layer for the particular token. This means that the embedding layer is of size `N * D`, where N is the count of all tokens in the model vocabulary.
- For some models, this embedding layer accounts for a large amount of the model size — the authors cite that the embedding layer for BERT-base accounts for 88% of the overall model size.
- Thus, it follows that if we could replace the embedding layer with a function that can compute the embedding at runtime, we can decrease the size of the a given model.

A bit of napkin math - for [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) the default embedding size is `4096` and the default vocabulary size is `32000`. If we load the model in f16, the embedding layer is of size:

* `4098` (embedding size) * `32000` (vocab size) * `2` (2 bytes per parameter) / `(1024 * 1024)` = ~0.3GB / ~12GB model size.

This isn't actually a large proportion of the overall model size (about 2.5% of all weights) and thus we probably wouldn't have an incentive to remove the embedding layer on Mistral. However, when we consider smaller models, such as [GPT2](https://huggingface.co/gpt2), we see a much larger potential relative size saving:

* `768` (embedding size) * `50000` (vocab size) * `2` (2 bytes per param) / `(1024 * 1024)` = ~73Mb / 245Mb model size.

About 30% of all weights! In this case, it might make sense to dynamically compute the embeddings.

## Properties of Dynamic Embeddings

A dynamic embedding function must satisfy a few properties:

- The embedding function must be deterministic.
- The embedding function must transform a string of arbitrary length to a vector of length `D` in embedding space.
- It follows that for more similar inputs, the embedding function should generate outputs that are closer in embedding space than less similiar inputs (this isn’t specified in the paper, but I think it’s a fair assumption to make).

In order to satisfy the property of determinism, the embedding function is based on a hashing algorithm — in this case, a polynomial rolling hash.


## Implementation

I thought I’d have a go at implementing this function using JAX. First we need to encode our string to a sequence of numerical values, of which I’ve just used the unicode number for each character of the string.


```python
from functools import partial
import jax
from jax import Array
import jax.numpy as jnp


def encode_text(text: str) -> Array:
  """Return string as unicode representation of chars."""
  return jnp.array([ord(i) for i in text])
```

Next we need write a function to compute the [polynomial rolling hash](https://www.geeksforgeeks.org/string-hashing-using-polynomial-rolling-hash-function/) of the encoded string. The algorithm is defined by the following equation:

$$ \text{Hash(String)} = s[0] \times p^{n-1} + s[1] \times p^{n-2} + \ldots + s[n-2] \times p + s[n-1] $$

* `s[i]` is ascii value of the `i-th` char of the string.
* `p` is a chosen prime number used as the base of the polynomial.
* `n` is the number of chars in the string.

And if we scan over the inputs to accumulate state, we’ve got our function to compute our polynomial rolling hash:

```python
@jax.jit
def rolling_hash(encoded: Array, p: int = 31, m: int = 10**9+7) -> Array:
    """Given a piece of text, compute it's rolling hash.
  
  Args:
    encoded: Array
    p: int
    m: int
  
  Returns:
    hash values: Array
  """

  def body_fn(result, elem):
    value, power = result
    value = (value + (elem - 96) * power) % m
    power = (power * p) % m
    return (value, power), elem

  # Compute rolling hash
  ((hash_value, _), _) = jax.lax.scan(body_fn, (encoded[0], encoded.shape[0]), encoded[1:])

  return hash_value
```

Next, we need to implement the full dynamic embedding function from the paper. This requires a computing sliding windows over i-grams of size `1..i`. This was a bit of a pain in JAX, but we can do it with a little bit of `.vmap` trickery.

```python
@partial(jax.jit, static_argnums=(1,))
def sliding_window(a: Array, size: int) -> Array:
  """Get all sliding windows of size over a."""
  starts = jnp.arange(len(a) - size + 1)
  return jax.vmap(lambda start: jax.lax.dynamic_slice(a, (start,), (size,)))(starts)
```

Right, now to implement the author’s full embedding algorithm. The steps are as such:

1. Pre-compute your hash seeds.
2. Get all i-grams of size `1..i` from the input characters.
3. For each set of i-grams of size `i`, compute it's rolling hash.
4. Compute the projection matrix using the outer product of the rolling hashes + hash seeds.
5. Normalize the projection matrix.
6. Average over all i-grams of size `i`.
7. Add to respective partition size.

```python
def hash_embedding(encoded: Array, seed: int = 42, b: int = 10**9+7, n: int = 3, d: int = 768) -> Array:
    """Compute hash embedding of a given piece of text.
    
    Args:
        encoded: Array -  encoded text we're embedding
        seed: int - random seed - MUST be held constant across embeddings.
        b: int - scalar bucket size.
        n: int - maximum size of an i-gram.
        d: int - the dimension of the embedding.
    
    Returns:
        embedding: jnp.array[d,]
    """

    # Initialize h and partitions
    partitions = jnp.sum(jnp.arange(1, n+1))
    h = jax.random.split(jax.random.PRNGKey(seed), d)[:, 0].reshape((d / partitions).astype(int), partitions) # reduce to 1d

    # Initialize loop variables
    embedding = jnp.zeros((d,))
    partition_idx = jnp.arange(0, d+1, int(d / partitions))
    run = 0

    # TODO: It'd be nice to move this to use jax.lax.scan
    for i in range(1, n+1):

        # Compute rolling hash
        igrams = sliding_window(encoded, i)
        s = rolling_hash(igrams.T)

        # Compute projection matrix
        p = jnp.outer(s, h[:, run:run+i]) # select the partition which is equal to run: run + i

        # Normalize
        p = p % b
        p = p - jnp.greater(p, b / 2) * b
        p = p / (b / 2)

        # Average
        igram_embedding = jnp.mean(p, axis = 0)

        # Concat to final embedding
        embedding = embedding.at[partition_idx[run]: partition_idx[run + i]].set(igram_embedding)
        run += i

    return embedding
```

It’d be nice to revisit this to make the main body of the function compatible with jax.lax.scan, at some point - but I couldn't get it to work at the time of writing due to some dynamic shapes flying about.

Anyway, we have a completed dynamic embedding function! I’m always a little bit worried I’ve made a mistake when I’m implementing code from papers with no programmatic reference, so I wrote a couple of tests to ensure the embedding function satisfies the properties we mentioned earlier.

![Hell yeah](/assets/images/testspass.png)


## Epilogue

It was nice to dig a little deeper into the idea behind dynamic embeddings. For smaller models, this could definitely be a useful way of shrinking models to sizes that can fit in-memory on edge devices. If you want to check out the code, have a look [here](https://github.com/harryjulian/hash-embeddings).
