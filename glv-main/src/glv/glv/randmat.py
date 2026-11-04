import jax
import jax.numpy as jnp
import jax.random as jr
import numpy
import functools
import json
import os

def new_seed():
    """
    Draw an int seed from system entropy, which can be printed, saved and passed as `key`
    """
    return int(numpy.random.SeedSequence().generate_state(1, dtype=numpy.uint32)[0])

def default_key(key=None):
    """
    Leaves a jax key untouched, turns an int key into a jax key, and None into
    a jax key based on system entropy 
    """
    def _is_jax_key(k):
        return isinstance(k, jax.Array) and jax.dtypes.issubdtype(k.dtype, jax.dtypes.prng_key)

    if not _is_jax_key(key):
        if key is None:
            key_out = new_seed()
        elif isinstance(key, int):
            key_out = key
        else:
            raise ValueError("key is not interpretable")
        key_out = jr.key(key_out)
    else:
        key_out = key

    return key_out

def gaussian_randmat(N, mu=0., sig=1., gam=0., diag=None, key=None):
    """
    Generate a random Gaussian (square) matrix of shape (N,N). Mean value `mu`, std `sig`, 
    pairwise correlation `gam`, and diagonal set to `diag` (broadcasted). Random generator
    `key` is either int, jax-like key, or None for system entropy.
    """

    if isinstance(gam, (int, float, numpy.ndarray)) and abs(gam) > 1.0:
        raise ValueError("γ > 1 or γ < -1 not allowed for correlation")

    key = default_key(key)
    
    factor = jnp.where(jnp.abs(gam) > 0.0, (1 - jnp.sqrt(1 - gam**2)) / gam, 0)
    
    M = jr.normal(key, (N, N))
    Z = (M + factor * M.T) / jnp.sqrt(1 + factor**2)
    A = mu + sig * Z

    if diag is not None:
        A = A.at[jnp.diag_indices(N)].set(diag)

    return A

jit_gaussian_randmat = functools.partial(jax.jit, static_argnums=(0,))(gaussian_randmat)

def save_randmat(name, seed, N, mu=0., sig=1., gam=0., diag=None, path="saved_matrices.json"):
    """
    Save under `name` everything needed to regenerate a `gaussian_randmat` to the JSON file
    `path`. The int `seed` alone is not enough, as the matrix also depends on the parameters.
    An existing `name` is never overwritten.
    """
    saved = {}
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)

    if name in saved:
        raise ValueError(f"'{name}' is already saved in {path}")

    params = {"seed": seed, "N": N, "mu": mu, "sig": sig, "gam": gam, "diag": diag}
    saved[name] = {k: numpy.asarray(v).tolist() if v is not None else None
                   for k, v in params.items()}

    with open(path, "w") as f:
        json.dump(saved, f, indent=2)

def load_randmat(name, path="saved_matrices.json"):
    """
    Regenerate the matrix saved under `name` by `save_randmat`
    """
    with open(path) as f:
        params = json.load(f)[name]

    key = params.pop("seed")
    params = {k: jnp.array(v) if isinstance(v, list) else v for k, v in params.items()}

    return gaussian_randmat(key=key, **params)


def gaussian_randmat_ensemble(B, N, mu=0., sig=1., gam=0., diag=None, key=None):
    """
    Sample `B` independent matrices; return as `(B,N,N)` array. `mu, sig, gam` can 
    be scalar (same for evert sample) or `(B,)`-shaped (different per sample). `diag`
    can be scalar (same for every sample, constant along diagonal) or `(B,N)` shaped
    to represent the full diagonal independently for each sample. A single `key` is given.   
    """
    key = default_key(key)
    keys = jax.random.split(key, B)

    mu_b  = jnp.broadcast_to(mu,  (B,))
    sig_b = jnp.broadcast_to(sig, (B,))
    gam_b = jnp.broadcast_to(gam, (B,))

    # Resolve how `diag` should be handled: shared scalar (unbatched),
    # per-matrix (B, N) array (batched on axis 0), or ambiguous (error).
    if diag is None or isinstance(diag, (int, float)):
        diag_b, diag_axis = diag, None
    else:
        diag_arr = jnp.asarray(diag)
        if diag_arr.ndim == 0:
            diag_b, diag_axis = diag_arr, None
        elif diag_arr.ndim == 1:
            raise ValueError(
                "diag as a 1D vector is ambiguous: it's unclear whether it should "
                "vary along the matrix diagonal (shared across the batch), be "
                "constant per matrix but vary across the batch, or something else. "
                "Pass a scalar/None for a shared diagonal, or a (B, N) array for "
                "an explicit per-matrix diagonal."
            )
        elif diag_arr.ndim == 2:
            if diag_arr.shape != (B, N):
                raise ValueError(f"diag must have shape (B, N) = ({B}, {N}), got {diag_arr.shape}")
            diag_b, diag_axis = diag_arr, 0
        else:
            raise ValueError("diag must be None, a scalar, or a (B, N) array")

    def _jit_batch_grm(N, diag_axis):
        def wrapped(mu_, sig_, gam_, diag_, key_):
            return gaussian_randmat(N, mu=mu_, sig=sig_, gam=gam_, diag=diag_, key=key_)

        batched = jax.vmap(wrapped, in_axes=(0, 0, 0, diag_axis, 0))
        return jax.jit(batched)

    batched = _jit_batch_grm(N, diag_axis)

    return batched(mu_b, sig_b, gam_b, diag_b, keys)