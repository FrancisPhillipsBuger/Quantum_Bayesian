#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Variational data re-uploading regressor (Jerbi et al., Nat. Commun. 14, 517, 2023):
f(x) = <0|U(x,theta)^dag O U(x,theta)|0> with trainable encodings, variational
blocks and readout O = sum_i c_i Z_i + b, trained with exact adjoint gradients.

Created on: Mon Jun 30 2026

@author: Chien-Chai Chang (Francis)

"""

#%% 0. Import required libraries
from math import ceil, sqrt

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from common.quantum_kernel import apply_1q, ring_pairs
from common.report import log, progress


#%% 1. Variational data re-uploading regressor
class DataReuploadingRegressor:

    """

    Explicit re-uploading circuit: per layer ry/rz(s*x_f + p) encoding, ry/rz(theta)
    variational block and a CZ ring; trained by mini-batch Adam on standardised y.

    Flat parameter layout: [enc_scale (E) | enc_phase (E) | var_ry (E) | var_rz (E) | c (q) | b],
    with E = q * L gate slots.


    """

    EARLY_STOP_RTOL = 1e-4      # relative epoch-MSE gain below which an epoch stalls
    READOUT_LR_RATIO = 10.0     # default readout / circuit learning-rate ratio

    def __init__(self, n_qubits=4, n_layers=3, scale=1.0, learning_rate=0.05,
                 epochs=40, batch_size=256, l2=0.0, random_state=0,
                 patience=None, readout_learning_rate=None,
                 laplace=True, laplace_prior_precision=1.0):

        """

        :param1 n_qubits:                 qubit cap; per fit q = min(n_features, n_qubits).
        :param2 n_layers:                 uploads per feature; the circuit runs
                                          n_layers * ceil(n_features / q) layers.
        :param3 scale:                    initial (trainable) encoding scale s.
        :param4 learning_rate:            Adam step size for circuit parameters.
        :param5 epochs:                   passes over the training set.
        :param6 batch_size:               mini-batch size.
        :param7 l2:                       L2 penalty on circuit parameters.
        :param8 random_state:             RNG seed for init and shuffling.
        :param9 patience:                 early-stopping patience in epochs; None disables.
        :param10 readout_learning_rate:   Adam step size for (c, b); None → READOUT_LR_RATIO * lr.
        :param11 laplace:                 fit a linearised-Laplace posterior for an
                                          input-dependent predictive std.
        :param12 laplace_prior_precision: Gaussian prior precision of that posterior.

        :return: None.

        """

        self.n_qubits = max(1, int(n_qubits))
        self.n_layers = max(1, int(n_layers))
        self.scale = float(scale)
        self.learning_rate = float(learning_rate)
        self.readout_learning_rate = (
            None if readout_learning_rate is None else float(readout_learning_rate)
        )
        self.epochs = max(1, int(epochs))
        self.batch_size = max(1, int(batch_size))
        self.l2 = float(l2)
        self.random_state = random_state
        self.patience = None if patience is None else max(1, int(patience))
        self.laplace = bool(laplace)
        self.laplace_prior_precision = float(laplace_prior_precision)

        # Set on fit()
        self.q_ = None                # qubits used
        self.n_layers_ = None         # layers used
        self.d_ = None                # input feature count
        self.n_ec_ = None             # gate slots E = q * L
        self.params_ = None           # flat parameter vector
        self.y_mean_ = 0.0
        self.y_std_ = 1.0
        self.loss_history_ = []
        self.train_rmse_ = None
        self.posterior_cov_ = None    # Laplace covariance (standardised space)
        self.noise_var_ = None        # residual variance (standardised space)

    #%% 1.1 Geometry and initialisation
    def circuit_shape(self, d):

        """Return (q, n_layers) for a d-feature input."""

        q = max(1, min(d, self.n_qubits))
        n_layers = max(1, int(self.n_layers)) * ceil(d / q)
        return q, n_layers

    cz_pairs = staticmethod(ring_pairs)

    def configure(self, d):

        """

        Set the per-fit geometry (d_, q_, n_layers_, n_ec_); shared with the kernel view.

        :param1 d: input feature count.

        :return: self.

        """

        self.d_ = int(d)
        self.q_, self.n_layers_ = self.circuit_shape(int(d))
        self.n_ec_ = self.q_ * self.n_layers_
        return self

    def init_params(self, rng):

        """

        Draw the initial flat parameter vector (configure() must have run).

        :param1 rng: numpy RandomState.

        :return: (n_params,) float array.

        """

        E = self.n_ec_
        params = np.empty(self._n_params())
        params[0:E] = self.scale + rng.normal(0.0, 0.1, size=E)   # enc scale
        params[E:2 * E] = rng.normal(0.0, 0.1, size=E)            # enc phase
        params[2 * E:3 * E] = rng.normal(0.0, 0.1, size=E)        # var ry
        params[3 * E:4 * E] = rng.normal(0.0, 0.1, size=E)        # var rz
        params[4 * E:4 * E + self.q_] = rng.normal(
            0.0, 1.0 / sqrt(self.q_), size=self.q_,
        )                                                         # c
        params[4 * E + self.q_] = 0.0                             # b
        return params

    def _n_params(self):

        """Return the parameter count 4E + q + 1."""

        return 4 * self.n_ec_ + self.q_ + 1

    def _unpack(self, params):

        """Return views (enc_scale, enc_phase, var_ry, var_rz, c, b)."""

        E = self.n_ec_
        s = params[0:E]
        p = params[E:2 * E]
        ty = params[2 * E:3 * E]
        tz = params[3 * E:4 * E]
        c = params[4 * E:4 * E + self.q_]
        b = params[4 * E + self.q_]
        return s, p, ty, tz, c, b

    #%% 1.2 Statevector primitives
    _apply_1q = staticmethod(apply_1q)

    @staticmethod
    def _ryrz(theta_y, theta_z):

        """Fused rz(theta_z) @ ry(theta_y) as (n, 2, 2) blocks."""

        theta_y = np.atleast_1d(theta_y)
        c, s = np.cos(theta_y / 2.0), np.sin(theta_y / 2.0)
        ph = np.exp(-0.5j * np.atleast_1d(theta_z))
        U = np.empty((theta_y.shape[0], 2, 2), dtype=np.complex128)
        U[:, 0, 0] = c * ph
        U[:, 0, 1] = -s * ph
        U[:, 1, 0] = s * ph.conj()
        U[:, 1, 1] = c * ph.conj()
        return U

    @staticmethod
    def _dag(U):

        """Return the conjugate transpose of a batch of 2x2 gates."""

        return np.conj(np.swapaxes(U, 1, 2))

    @staticmethod
    def _gen_enc(a):

        """Adjoint generator Z + rz(a) Y rz(a)^dag of the fused encoding gate."""

        a = np.atleast_1d(a)
        ph = np.exp(-1j * a)
        G = np.empty((a.shape[0], 2, 2), dtype=np.complex128)
        G[:, 0, 0] = 1.0
        G[:, 0, 1] = -1j * ph
        G[:, 1, 0] = 1j * ph.conj()
        G[:, 1, 1] = -1.0
        return G

    @staticmethod
    def _gen_y_conj(theta_z):

        """Return rz(tz) Y rz(tz)^dag, the ry generator seen through the fused gate."""

        theta_z = np.atleast_1d(theta_z)
        ph = np.exp(-1j * theta_z)
        G = np.zeros((theta_z.shape[0], 2, 2), dtype=np.complex128)
        G[:, 0, 1] = -1j * ph
        G[:, 1, 0] = 1j * ph.conj()
        return G

    def _cz_mask(self, q, dim):

        """Diagonal (+/-1) mask of the whole CZ ring."""

        idx = np.arange(dim)
        mask = np.ones(dim)
        for cc, tt in self.cz_pairs(q):
            mask *= np.where(((idx >> cc) & 1) & ((idx >> tt) & 1), -1.0, 1.0)
        return mask

    #%% 1.3 Forward pass
    def _statevector(self, X, params):

        """

        Build U(x, params)|0...0> for every row of X.

        :param1 X:      (n, d) feature batch.
        :param2 params: flat parameter vector.

        :return: (n, 2^q) complex statevector batch.

        """

        n, d = X.shape
        q, n_layers = self.q_, self.n_layers_
        dim = 1 << q
        s, p, ty, tz, _, _ = self._unpack(params)

        psi = np.zeros((n, dim), dtype=np.complex128)
        psi[:, 0] = 1.0  # |0...0>

        cz_mask = self._cz_mask(q, dim)

        for layer in range(n_layers):
            for i in range(q):
                e = layer * q + i
                f = e % d
                a = s[e] * X[:, f] + p[e]
                psi = self._apply_1q(psi, self._ryrz(a, a), i, n, q, dim)
            for i in range(q):
                e = layer * q + i
                psi = self._apply_1q(psi, self._ryrz(ty[e], tz[e]), i, n, q, dim)
            psi = psi * cz_mask
        return psi

    def _z_all(self, psi, q):

        """Return <Z_i> for every qubit, shape (n, q)."""

        idx = np.arange(psi.shape[1])
        prob = psi.real ** 2 + psi.imag ** 2
        z = np.empty((psi.shape[0], q))
        for i in range(q):
            z[:, i] = prob @ (1.0 - 2.0 * ((idx >> i) & 1))
        return z

    def _forward(self, X, params=None):

        """Return (standardised prediction, <Z_i> matrix) for X."""

        if params is None:
            params = self.params_
        _, _, _, _, c, b = self._unpack(params)
        psi = self._statevector(X, params)
        z = self._z_all(psi, self.q_)
        pred_std = z @ c + b
        return pred_std, z

    #%% 1.4 Adjoint gradient
    def _grad(self, X, params):

        """

        Standardised prediction and per-sample Jacobian by adjoint differentiation.

        :param1 X:      (n, d) feature batch.
        :param2 params: flat parameter vector.

        :return: (pred_std (n,), z (n, q), d_pred (n, n_params)).

        """

        n, d = X.shape
        q, n_layers = self.q_, self.n_layers_
        dim = 1 << q
        E = self.n_ec_
        s, p, ty, tz, c, b = self._unpack(params)

        idx = np.arange(dim)
        z_masks = [1.0 - 2.0 * ((idx >> i) & 1) for i in range(q)]
        cz_mask = self._cz_mask(q, dim)

        ket = self._statevector(X, params)
        z = self._z_all(ket, q)
        pred_std = z @ c + b

        # lam = O|psi> with the diagonal observable O = sum_i c_i Z_i
        cmask = np.zeros(dim)
        for i in range(q):
            cmask = cmask + c[i] * z_masks[i]
        lam = ket * cmask

        d_s = np.zeros((n, E)); d_p = np.zeros((n, E))
        d_ty = np.zeros((n, E)); d_tz = np.zeros((n, E))

        for layer in reversed(range(n_layers)):
            ket = ket * cz_mask                 # CZ ring is self-inverse
            lam = lam * cz_mask
            # Variational block, reverse qubit order
            for i in reversed(range(q)):
                e = layer * q + i
                pk = ket * z_masks[i]           # rz half: generator Z_i
                d_tz[:, e] = np.imag(np.sum(np.conj(lam) * pk, axis=1))
                pk = self._apply_1q(ket, self._gen_y_conj(tz[e]), i, n, q, dim)
                d_ty[:, e] = np.imag(np.sum(np.conj(lam) * pk, axis=1))
                U_inv = self._dag(self._ryrz(ty[e], tz[e]))
                ket = self._apply_1q(ket, U_inv, i, n, q, dim)
                lam = self._apply_1q(lam, U_inv, i, n, q, dim)
            # Encoding block; the shared angle a = s*x_f + p chains to s and p
            for i in reversed(range(q)):
                e = layer * q + i
                f = e % d
                xf = X[:, f]
                a = s[e] * xf + p[e]
                pk = self._apply_1q(ket, self._gen_enc(a), i, n, q, dim)
                g = np.imag(np.sum(np.conj(lam) * pk, axis=1))
                d_s[:, e] = g * xf
                d_p[:, e] = g
                U_inv = self._dag(self._ryrz(a, a))
                ket = self._apply_1q(ket, U_inv, i, n, q, dim)
                lam = self._apply_1q(lam, U_inv, i, n, q, dim)

        # Readout derivatives: d/dc_i = <Z_i>, d/db = 1
        d_circ = np.concatenate([d_s, d_p, d_ty, d_tz], axis=1)
        d_read_b = np.ones((n, 1))
        d_pred = np.concatenate([d_circ, z, d_read_b], axis=1)
        return pred_std, z, d_pred

    #%% 1.5 Fit
    def fit(self, X, y, show_progress=False):

        """

        Train by MSE minimisation with mini-batch Adam on standardised targets.

        :param1 X:             (n, d) standardised feature matrix.
        :param2 y:             length-n regression target.
        :param3 show_progress: show an epoch progress bar.

        :return: self.

        """

        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, d = X.shape

        self.configure(d)
        n_params = self._n_params()
        E = self.n_ec_

        self.y_mean_ = float(np.mean(y))
        self.y_std_ = float(np.std(y)) if np.std(y) > 0 else 1.0
        ys = (y - self.y_mean_) / self.y_std_

        rng = np.random.RandomState(self.random_state)
        params = self.init_params(rng)
        self.params_ = params

        m = np.zeros(n_params)
        v = np.zeros(n_params)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t_step = 0

        # Circuit parameters use learning_rate, readout (c, b) the readout rate
        readout_lr = (
            self.READOUT_LR_RATIO * self.learning_rate
            if self.readout_learning_rate is None else self.readout_learning_rate
        )
        lr = np.full(n_params, self.learning_rate)
        lr[4 * E:] = readout_lr

        l2_mask = np.zeros(n_params)
        if self.l2 > 0.0:
            l2_mask[0:4 * E] = 1.0

        self.loss_history_ = []
        best_mse = float('inf')
        stall = 0
        epoch_iter = range(self.epochs)
        if show_progress:
            epoch_iter = progress(epoch_iter, "Re-uploading epochs")
        for epoch in epoch_iter:
            perm = rng.permutation(n)
            epoch_se = 0.0
            for start in range(0, n, self.batch_size):
                bidx = perm[start:start + self.batch_size]
                Xb, yb = X[bidx], ys[bidx]
                nb = Xb.shape[0]

                pred, z, d_pred = self._grad(Xb, self.params_)
                err = pred - yb
                epoch_se += float(np.sum(err ** 2))

                grad = (2.0 / nb) * (d_pred.T @ err)
                if self.l2 > 0.0:
                    grad += 2.0 * self.l2 * l2_mask * self.params_

                t_step += 1
                m = beta1 * m + (1.0 - beta1) * grad
                v = beta2 * v + (1.0 - beta2) * grad ** 2
                m_hat = m / (1.0 - beta1 ** t_step)
                v_hat = v / (1.0 - beta2 ** t_step)
                self.params_ -= lr * m_hat / (np.sqrt(v_hat) + eps)

            mse = epoch_se / n
            self.loss_history_.append(mse)
            if show_progress:
                epoch_iter.set_postfix(MSE=f"{mse * self.y_std_ ** 2:.6f}", refresh=False)

            if self.patience is not None:
                if mse < best_mse * (1.0 - self.EARLY_STOP_RTOL):
                    best_mse = mse
                    stall = 0
                else:
                    stall += 1
                    if stall >= self.patience:
                        if show_progress:
                            epoch_iter.close()
                            log("Tier-2", f"Early stop at epoch {epoch + 1}/{self.epochs} "
                                          f"(no improvement in {self.patience} epochs)")
                        break

        train_pred = self.predict(X)
        self.train_rmse_ = float(np.sqrt(np.mean((train_pred - y) ** 2)))

        if self.laplace:
            self.fit_laplace(X, ys, show_progress=show_progress)
        return self

    #%% 1.6 Linearised-Laplace posterior
    def fit_laplace(self, X, ys, show_progress=False):

        """

        Posterior precision A = JᵀJ / sigma^2 + tau I at the trained parameters,
        giving var(x) = sigma^2 + j(x)ᵀ A^-1 j(x).

        :param1 X:             (n, d) standardised training features.
        :param2 ys:            length-n standardised training target.
        :param3 show_progress: report a failure instead of falling back silently.

        :return: self (posterior left None on failure → constant std).

        """

        try:
            n = X.shape[0]
            P = self._n_params()
            A = self.laplace_prior_precision * np.eye(P)
            ssr = 0.0

            for start in range(0, n, self.batch_size):
                Xb = X[start:start + self.batch_size]
                yb = ys[start:start + self.batch_size]
                pred, _, J = self._grad(Xb, self.params_)
                A += J.T @ J
                err = pred - yb
                ssr += float(err @ err)

            # n - P guards against an over-confident sigma when over-parameterised
            noise_var = ssr / max(n - P, 1)
            if not np.isfinite(noise_var) or noise_var <= 0.0:
                noise_var = max(ssr / max(n, 1), 1e-12)

            A = (A - self.laplace_prior_precision * np.eye(P)) / noise_var
            A.flat[::P + 1] += self.laplace_prior_precision

            jitter = 1e-10 * float(np.trace(A)) / P
            for _ in range(6):
                try:
                    c, lower = cho_factor(A, lower=True, check_finite=False)
                    cov = cho_solve((c, lower), np.eye(P), check_finite=False)
                    break
                except Exception:
                    A.flat[::P + 1] += max(jitter, 1e-12)
                    jitter *= 10.0
            else:
                raise np.linalg.LinAlgError("Laplace precision not factorisable")

            self.posterior_cov_ = 0.5 * (cov + cov.T)
            self.noise_var_ = float(noise_var)
        except Exception as exc:
            self.posterior_cov_ = None
            self.noise_var_ = None
            if show_progress:
                log("Warning", f"Laplace posterior failed ({exc}); "
                               f"using the constant training RMSE as std")
        return self

    def predictive_std(self, X):

        """

        Laplace predictive std for X.

        :param1 X: (n, d) feature matrix.

        :return: length-n std in target units, or None when no posterior exists.

        """

        if self.posterior_cov_ is None or self.noise_var_ is None:
            return None

        n = X.shape[0]
        var = np.empty(n)
        for start in range(0, n, self.batch_size):
            Xb = X[start:start + self.batch_size]
            _, _, J = self._grad(Xb, self.params_)
            var[start:start + Xb.shape[0]] = np.einsum(
                'np,np->n', J @ self.posterior_cov_, J, optimize=True,
            )
        var = np.maximum(var + self.noise_var_, 0.0)
        return np.sqrt(var) * self.y_std_

    #%% 1.7 Predict
    def predict(self, X, return_std=False):

        """

        Predict targets for X (un-scaling the standardised readout).

        :param1 X:          (n, d) feature matrix.
        :param2 return_std: also return the predictive std (Laplace, else training RMSE).

        :return: predictions, or (predictions, std).

        """

        if self.params_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        X = np.ascontiguousarray(X, dtype=np.float64)
        pred_std, _ = self._forward(X)
        pred = pred_std * self.y_std_ + self.y_mean_
        if return_std:
            std = self.predictive_std(X)
            if std is None:
                std = np.full(X.shape[0],
                              self.train_rmse_ if self.train_rmse_ else 0.0)
            return pred, std
        return pred
