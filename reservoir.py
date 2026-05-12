"""reservoir.py — Echo State Network (Reservoir Computing) core.

Architecture
------------
- Large fixed random recurrent reservoir (1 000 nodes, sparse 5 % connectivity)
- Leaky-integrator neurons:  x(t) = (1-a)*x(t-1) + a*tanh(W_in*u(t) + W*x(t-1))
- Only the linear readout layer W_out is trained (ridge regression)
- No backpropagation through time — training is a single matrix solve

Key hyper-parameters (all in config.py)
- RESERVOIR_SIZE   : number of recurrent nodes
- SPECTRAL_RADIUS  : rho(W); must be < 1 for echo state property
- SPARSITY         : fraction of non-zero W connections
- INPUT_SCALING    : scales W_in entries
- LEAK_RATE        : leaky integration alpha
- RIDGE_ALPHA      : L2 penalty for W_out ridge regression
- N_ENSEMBLES      : number of independent reservoirs (different random seeds)
"""

from __future__ import annotations

import numpy as np

import config


class EchoStateNetwork:
    """Single-reservoir ESN with leaky integrator neurons and ridge readout.

    Parameters
    ----------
    n_inputs  : int  — dimensionality of u(t)
    n_outputs : int  — dimensionality of y(t) (one per ETF)
    seed      : int  — random seed for reproducibility
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        seed: int = 42,
        reservoir_size: int = config.RESERVOIR_SIZE,
        spectral_radius: float = config.SPECTRAL_RADIUS,
        sparsity: float = config.SPARSITY,
        input_scaling: float = config.INPUT_SCALING,
        leak_rate: float = config.LEAK_RATE,
        ridge_alpha: float = config.RIDGE_ALPHA,
    ) -> None:
        self.n_res   = reservoir_size
        self.n_in    = n_inputs
        self.n_out   = n_outputs
        self.alpha   = leak_rate
        self.ridge   = ridge_alpha
        self.rng     = np.random.default_rng(seed)

        # ── Input weights W_in : shape (n_res, n_inputs) ─────────────────────
        self.W_in = (self.rng.uniform(-1, 1, (self.n_res, n_inputs))
                     * input_scaling).astype(np.float32)

        # ── Reservoir weights W : sparse, rescaled to spectral_radius ─────────
        W = self._make_reservoir(spectral_radius, sparsity)
        self.W = W.astype(np.float32)

        # ── Readout weights (set after fit) ───────────────────────────────────
        self.W_out: np.ndarray | None = None
        self._state = np.zeros(self.n_res, dtype=np.float32)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _make_reservoir(self, spectral_radius: float, sparsity: float) -> np.ndarray:
        """Build a sparse random reservoir matrix scaled to the desired rho(W)."""
        n = self.n_res
        # sparse random matrix
        W = self.rng.uniform(-1, 1, (n, n)).astype(np.float64)
        # apply sparsity mask
        mask = self.rng.uniform(0, 1, (n, n)) > sparsity
        W[mask] = 0.0
        # rescale to target spectral radius
        eigvals = np.linalg.eigvals(W)
        rho = np.max(np.abs(eigvals))
        if rho > 1e-8:
            W *= spectral_radius / rho
        return W

    def _step(self, u: np.ndarray) -> np.ndarray:
        """Advance reservoir one time step.

        x(t) = (1 - a) * x(t-1) + a * tanh(W_in @ u(t) + W @ x(t-1))
        """
        pre = self.W_in @ u + self.W @ self._state
        x_new = (1.0 - self.alpha) * self._state + self.alpha * np.tanh(pre)
        self._state = x_new
        return x_new.copy()

    def _collect_states(
        self, U: np.ndarray, reset: bool = True
    ) -> np.ndarray:
        """Drive reservoir with input sequence U (T, n_inputs).

        Returns state matrix S (T, n_res).
        """
        if reset:
            self._state = np.zeros(self.n_res, dtype=np.float32)
        T = len(U)
        S = np.empty((T, self.n_res), dtype=np.float32)
        for t in range(T):
            S[t] = self._step(U[t])
        return S

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        U_train: np.ndarray,
        Y_train: np.ndarray,
        warmup_steps: int = 100,
    ) -> "EchoStateNetwork":
        """Fit readout W_out using ridge regression.

        Parameters
        ----------
        U_train      : (T, n_inputs)  — input sequence
        Y_train      : (T, n_outputs) — target sequence (1-day ahead returns)
        warmup_steps : rows to discard at start for transient wash-out
        """
        S = self._collect_states(U_train, reset=True)

        # Discard warm-up transient
        S_tr = S[warmup_steps:]
        Y_tr = Y_train[warmup_steps:]

        # Augment state with bias
        ones  = np.ones((len(S_tr), 1), dtype=np.float32)
        S_aug = np.concatenate([S_tr, ones], axis=1)   # (T', n_res+1)

        # Ridge regression: W_out = (S^T S + alpha I)^{-1} S^T Y
        n = S_aug.shape[1]
        A = S_aug.T @ S_aug + self.ridge * np.eye(n, dtype=np.float32)
        B = S_aug.T @ Y_tr.astype(np.float32)
        self.W_out = np.linalg.solve(A, B)             # (n_res+1, n_outputs)
        return self

    def predict_one(self, u: np.ndarray) -> np.ndarray:
        """Predict next step given single input vector u (n_inputs,).

        Advances reservoir state in-place. Call sequentially for walk-forward.
        """
        if self.W_out is None:
            raise RuntimeError("Call fit() before predict_one().")
        s = self._step(u)
        s_aug = np.append(s, 1.0).astype(np.float32)   # bias
        return (s_aug @ self.W_out)                     # (n_outputs,)

    def reset_state(self) -> None:
        """Reset reservoir hidden state to zeros."""
        self._state = np.zeros(self.n_res, dtype=np.float32)


# ── Ensemble of ESNs ──────────────────────────────────────────────────────────

class ESNEnsemble:
    """Ensemble of N independent ESNs; output = mean of member predictions.

    Using multiple random seeds reduces variance from the random reservoir
    initialisation, giving more stable cross-sectional rankings.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        seeds: list[int] = config.RANDOM_SEEDS,
    ) -> None:
        self.members = [
            EchoStateNetwork(n_inputs, n_outputs, seed=s)
            for s in seeds
        ]

    def fit(
        self,
        U_train: np.ndarray,
        Y_train: np.ndarray,
        warmup_steps: int = 100,
    ) -> "ESNEnsemble":
        for m in self.members:
            m.fit(U_train, Y_train, warmup_steps=warmup_steps)
        return self

    def predict_one(self, u: np.ndarray) -> np.ndarray:
        """Mean prediction across all ensemble members."""
        preds = np.stack([m.predict_one(u) for m in self.members], axis=0)
        return preds.mean(axis=0)

    def reset_states(self) -> None:
        for m in self.members:
            m.reset_state()
