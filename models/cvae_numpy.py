"""
models/cvae_numpy.py
=====================
NumPy CVAE fallback when PyTorch is unavailable.

Fixes over original:
1. KL annealing with configurable warmup schedule (was missing entirely)
2. Padding masking in reconstruction loss (was missing: padded zeros
   contributed to loss, biasing decoder toward 'A' at all positions)
3. Sequence length stored and used at generation time — no more
   .rstrip('A') hack which silently removed valid alanines
4. Gradient clipping retained (original had it; kept here)

This implementation is suitable for environments where PyTorch cannot
be installed. For better generation quality, use cvae_torch.py.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from features.physicochemical import AA_LIST, AA_INDEX, is_canonical
from features.transformers import seq_to_onehot

logger = logging.getLogger(__name__)


class PeptideCVAENP:
    """
    Pure NumPy CVAE with:
    - Adam optimiser with gradient clipping
    - KL annealing (linear β schedule)
    - Per-position masked cross-entropy loss

    Architecture:
      Encoder : x(D) → h(H, ReLU) → μ(L), logv(L)
      Decoder : z(L) → h(H, ReLU) → logits(D) → per-position softmax
    where D = max_len * 20, H = hidden_dim, L = latent_dim
    """

    def __init__(
        self,
        max_len: int = 50,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        seed: int = 42,
    ):
        self.max_len    = max_len
        self.L          = latent_dim
        self.H          = hidden_dim
        self.lr         = lr
        self.D          = max_len * 20
        np.random.seed(seed)

        # Encoder weights
        self.We1  = self._he(self.D, hidden_dim)
        self.be1  = np.zeros(hidden_dim, dtype=np.float32)
        self.Wmu  = self._he(hidden_dim, latent_dim)
        self.bmu  = np.zeros(latent_dim, dtype=np.float32)
        self.Wlv  = self._he(hidden_dim, latent_dim)
        self.blv  = np.zeros(latent_dim, dtype=np.float32)

        # Decoder weights
        self.Wd1  = self._he(latent_dim, hidden_dim)
        self.bd1  = np.zeros(hidden_dim, dtype=np.float32)
        self.Wd2  = self._he(hidden_dim, self.D)
        self.bd2  = np.zeros(self.D, dtype=np.float32)

        self._param_names = [
            'We1', 'be1', 'Wmu', 'bmu', 'Wlv', 'blv',
            'Wd1', 'bd1', 'Wd2', 'bd2',
        ]

    @staticmethod
    def _he(fan_in, fan_out):
        return (np.random.randn(fan_in, fan_out) *
                np.sqrt(2.0 / fan_in)).astype(np.float32)

    @staticmethod
    def _relu(x):
        return np.maximum(0.0, x)

    def _softmax_per_pos(self, logits):
        """logits: (B, D) → (B, D) per-position softmax."""
        B = logits.shape[0]
        x3 = logits.reshape(B, self.max_len, 20)
        x3 -= x3.max(axis=2, keepdims=True)
        ex = np.exp(x3)
        return (ex / ex.sum(axis=2, keepdims=True)).reshape(B, self.D)

    # -----------------------------------------------------------------------
    # Encoder / Decoder forward passes
    # -----------------------------------------------------------------------

    def _encode(self, X):
        z1   = X @ self.We1 + self.be1
        h1   = self._relu(z1)
        mu   = h1 @ self.Wmu + self.bmu
        logv = np.clip(h1 @ self.Wlv + self.blv, -10.0, 4.0)
        return mu, logv, h1, z1

    def _decode(self, z):
        z1  = z @ self.Wd1 + self.bd1
        h1  = self._relu(z1)
        out = h1 @ self.Wd2 + self.bd2
        probs = self._softmax_per_pos(out)
        return probs, h1, z1

    # -----------------------------------------------------------------------
    # Masked reconstruction loss
    # -----------------------------------------------------------------------

    def _masked_recon_loss(self, probs, targets, masks):
        """
        Cross-entropy loss over real (non-padded) positions only.

        probs   : (B, D) — output probabilities
        targets : (B, D) — one-hot ground truth
        masks   : (B, max_len) — True = real residue

        FIX: original used np.sum(targets * log(probs)) over ALL positions,
        including padding zeros. This made the model learn to assign
        high probability to whatever was encoded at padded positions (zeros
        → index 0 in AA_LIST = 'A'), biasing generation toward 'AAAA...'
        tails. Dividing by B (not by total real positions) further inflated
        the relative loss at padded positions.
        """
        B = probs.shape[0]
        log_probs = np.log(probs + 1e-8).reshape(B, self.max_len, 20)
        tgt_3d    = targets.reshape(B, self.max_len, 20)
        # Per-position cross-entropy: sum over 20 AA classes
        ce_per_pos = -(tgt_3d * log_probs).sum(axis=2)   # (B, max_len)
        # Mask: only count real positions
        mask_f     = masks.astype(np.float32)             # (B, max_len)
        n_real     = mask_f.sum()
        if n_real == 0:
            return 0.0
        return (ce_per_pos * mask_f).sum() / n_real

    # -----------------------------------------------------------------------
    # Single optimiser step (Adam)
    # -----------------------------------------------------------------------

    def _step(self, batch, masks, beta, t, m_state, v_state,
              beta1=0.9, beta2=0.999, eps=1e-8, grad_clip=5.0):
        B = batch.shape[0]

        # Forward
        mu, logv, eh1, ez1 = self._encode(batch)
        noise = np.random.randn(B, self.L).astype(np.float32)
        z     = mu + noise * np.exp(0.5 * logv)
        probs, dh1, dz1 = self._decode(z)

        recon = self._masked_recon_loss(probs, batch, masks)
        kl    = float(-0.5 * np.mean(1.0 + logv - mu ** 2 - np.exp(logv)))
        loss  = recon + beta * kl

        # Backward: decoder
        # Combined softmax + masked CE gradient:
        #   dL/d_logit_j = (probs_j - target_j) / n_real  at real positions
        #                = 0                               at padded positions
        mask_f   = masks.astype(np.float32).reshape(B, self.max_len, 1)  # (B,L,1)
        n_real   = max(masks.sum(), 1)
        probs_3d = probs.reshape(B, self.max_len, 20)
        tgt_3d   = batch.reshape(B, self.max_len, 20)
        d_logit  = ((probs_3d - tgt_3d) * mask_f / n_real).reshape(B, self.D)

        gWd2 = dh1.T @ d_logit
        gbd2 = d_logit.sum(axis=0)
        d_dh1 = d_logit @ self.Wd2.T
        d_dz1 = d_dh1 * (dz1 > 0)
        gWd1  = z.T @ d_dz1
        gbd1  = d_dz1.sum(axis=0)
        d_z   = d_dz1 @ self.Wd1.T

        # Backward: KL + reparameterisation
        d_kl_mu = mu / B
        d_kl_lv = 0.5 * (np.exp(logv) - 1.0) / B
        d_mu    = beta * d_kl_mu + d_z
        d_lv    = beta * d_kl_lv + d_z * noise * 0.5 * np.exp(0.5 * logv)

        gWmu = eh1.T @ d_mu
        gbmu = d_mu.sum(axis=0)
        gWlv = eh1.T @ d_lv
        gblv = d_lv.sum(axis=0)
        d_eh1 = d_mu @ self.Wmu.T + d_lv @ self.Wlv.T
        d_ez1 = d_eh1 * (ez1 > 0)
        gWe1  = batch.T @ d_ez1
        gbe1  = d_ez1.sum(axis=0)

        grads = dict(
            We1=gWe1, be1=gbe1, Wmu=gWmu, bmu=gbmu,
            Wlv=gWlv, blv=gblv, Wd1=gWd1, bd1=gbd1,
            Wd2=gWd2, bd2=gbd2,
        )

        # Adam update
        for p in self._param_names:
            g = np.clip(grads[p], -grad_clip, grad_clip)
            m_state[p] = beta1 * m_state[p] + (1 - beta1) * g
            v_state[p] = beta2 * v_state[p] + (1 - beta2) * g ** 2
            mh = m_state[p] / (1 - beta1 ** t)
            vh = v_state[p] / (1 - beta2 ** t)
            setattr(self, p, getattr(self, p) - self.lr * mh / (np.sqrt(vh) + eps))

        return loss, recon, kl

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train(
        self,
        seqs: List[str],
        epochs: int = 150,
        batch_size: int = 64,
        kl_beta: float = 1.0,
        kl_warmup_epochs: int = 60,
        grad_clip: float = 5.0,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 25,
        resume_from: Optional[str] = None,
    ) -> list:
        # Build one-hot matrix and lengths
        N = len(seqs)
        X = np.vstack([
            seq_to_onehot(s, self.max_len).reshape(-1) for s in seqs
        ])
        lengths = np.array([min(len(s), self.max_len) for s in seqs],
                           dtype=np.int32)
        # Build mask matrix: (N, max_len)
        masks = np.zeros((N, self.max_len), dtype=bool)
        for i, l in enumerate(lengths):
            masks[i, :l] = True

        m_state = {p: np.zeros_like(getattr(self, p)) for p in self._param_names}
        v_state = {p: np.zeros_like(getattr(self, p)) for p in self._param_names}
        t = 0

        start_epoch = 0
        if resume_from and Path(resume_from).exists():
            ckpt = np.load(resume_from, allow_pickle=True).item()
            for p in self._param_names:
                setattr(self, p, ckpt[p])
            m_state = ckpt['m_state']
            v_state = ckpt['v_state']
            t       = ckpt['t']
            start_epoch = ckpt.get('epoch', 0) + 1
            logger.info("Resumed numpy CVAE from epoch %d", start_epoch)

        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        history = []
        for epoch in range(start_epoch, epochs):
            # KL annealing
            beta = kl_beta * min(1.0, (epoch + 1) / max(kl_warmup_epochs, 1))
            idx  = np.random.permutation(N)
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, N, batch_size):
                bi    = idx[start: start + batch_size]
                batch = X[bi]
                bmask = masks[bi]
                if len(bi) < 4:
                    continue
                t += 1
                loss, recon, kl = self._step(
                    batch, bmask, beta, t, m_state, v_state,
                    grad_clip=grad_clip,
                )
                epoch_loss += loss
                n_batches  += 1

            avg = epoch_loss / max(n_batches, 1)
            history.append(avg)

            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(
                    "NP-CVAE  Epoch %3d/%d  loss=%.4f  β=%.3f",
                    epoch + 1, epochs, avg, beta,
                )

            if checkpoint_dir and ((epoch + 1) % checkpoint_every == 0 or
                                    epoch == epochs - 1):
                ckpt = {p: getattr(self, p) for p in self._param_names}
                ckpt.update({
                    'm_state': m_state, 'v_state': v_state,
                    't': t, 'epoch': epoch,
                })
                ckpt_path = (Path(checkpoint_dir) /
                             f"cvae_np_epoch_{epoch+1:04d}.npy")
                np.save(str(ckpt_path), ckpt)
                logger.info("Checkpoint saved: %s", ckpt_path)

        return history

    # -----------------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------------

    def generate(
        self,
        n: int = 500,
        temperature: float = 1.1,
        min_len: int = 8,
        target_lengths: Optional[List[int]] = None,
    ) -> List[str]:
        """
        Generate sequences by sampling from the prior N(0,I).

        FIX: sequence length is now specified explicitly via
        target_lengths (a list of desired lengths, sampled randomly if
        not provided). The original .rstrip('A') approach was removed
        because it introduces two bugs:
          (a) it removes valid C-terminal alanine residues
          (b) the stopping criterion was positional (based on AA_INDEX[0]='A')
              rather than length-based, so it was not portable if AA_LIST
              ordering changed.
        """
        sequences = []
        rng = np.random.default_rng(0)

        while len(sequences) < n:
            nb = min(256, (n - len(sequences)) * 4)
            z  = np.random.randn(nb, self.L).astype(np.float32)
            probs, _, _ = self._decode(z)
            probs = probs.reshape(nb, self.max_len, 20)

            if temperature != 1.0:
                logits = np.log(probs + 1e-8) / temperature
                logits -= logits.max(axis=2, keepdims=True)
                ex = np.exp(logits)
                probs = ex / ex.sum(axis=2, keepdims=True)

            for i in range(nb):
                # Determine length for this sample
                if target_lengths:
                    seq_len = int(rng.choice(target_lengths))
                else:
                    lo = max(min_len, self.max_len // 4)
                    seq_len = int(rng.integers(lo, self.max_len + 1))
                seq_len = min(seq_len, self.max_len)

                chars = [
                    AA_LIST[rng.choice(20, p=probs[i, pos] /
                                       probs[i, pos].sum())]
                    for pos in range(seq_len)
                ]
                seq_str = ''.join(chars)
                if min_len <= len(seq_str) <= self.max_len and is_canonical(seq_str):
                    sequences.append(seq_str)
                if len(sequences) >= n:
                    break

        logger.info("NP-CVAE generated %d sequences", len(sequences))
        return sequences[:n]

    def encode_sequence(self, seq: str) -> np.ndarray:
        x = seq_to_onehot(seq, self.max_len).reshape(1, -1)
        mu, _, _, _ = self._encode(x)
        return mu[0]
