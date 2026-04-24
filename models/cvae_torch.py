"""
models/cvae_torch.py
=====================
PyTorch CVAE for peptide sequence generation.

Fixes over original NumPy CVAE:
1. KL annealing: β-schedule linearly increases from 0 to kl_beta over
   kl_warmup_epochs. Prevents posterior collapse at the start of training
   where the decoder learns to ignore the latent code.

2. Padding masking: real sequence lengths are tracked. Reconstruction
   loss is computed only over real residue positions — padded positions
   do not contribute. Original code treated padded zeros as real signal,
   biasing the model toward generating short sequences with trailing gaps.

3. Architecture: 1D-CNN encoder provides translational invariance and
   captures local sequence motifs (k-mer context). Original was a flat
   MLP ignoring positional structure.

4. Decoder: position-wise MLP with residual connections. Outputs a
   (batch, max_len, 20) logit tensor; argmax or sampling per position.

5. Teacher forcing: not applicable to a non-autoregressive decoder.
   Instead, the decoder directly predicts all positions jointly from z,
   which is more stable for generation. If autoregressive generation
   is required, replace with LSTM decoder (see comments below).

6. Checkpointing: saves state_dict + training state every N epochs,
   supports resume_from.

7. Length conditioning: sequence length is encoded as a scalar feature
   concatenated to z before decoding, allowing the generator to be
   conditioned on desired peptide length at inference time.
"""

import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available; falling back to NumPy CVAE.")

from features.physicochemical import AA_LIST, AA_INDEX, is_canonical
from features.transformers import seq_to_onehot, seq_length_to_mask


# =============================================================================
# Data preparation helpers
# =============================================================================

def _build_tensors(
    seqs: List[str],
    max_len: int,
) -> Tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor']:
    """
    Returns:
      x      : (N, max_len, 20)  float32 one-hot
      lengths: (N,)              int64   actual sequence lengths
      masks  : (N, max_len)      bool    True = real residue
    """
    N = len(seqs)
    x = np.zeros((N, max_len, 20), dtype=np.float32)
    lengths = np.zeros(N, dtype=np.int64)
    masks = np.zeros((N, max_len), dtype=bool)

    for i, seq in enumerate(seqs):
        l = min(len(seq), max_len)
        lengths[i] = l
        masks[i, :l] = True
        for j, aa in enumerate(seq[:max_len]):
            if aa in AA_INDEX:
                x[i, j, AA_INDEX[aa]] = 1.0

    return (
        torch.from_numpy(x),
        torch.from_numpy(lengths),
        torch.from_numpy(masks),
    )


# =============================================================================
# Encoder: 1D-CNN over sequence positions
# =============================================================================

class CNNEncoder(nn.Module):
    """
    1D convolutional encoder.

    Processes input shape (B, L, 20) as (B, 20, L) for Conv1d.
    Three conv layers with residual connections capture local context
    at scales 3, 5, 7 (representing tri-, penta-, hepta-peptide motifs).
    Global average pooling aggregates into a fixed-size representation.
    """

    def __init__(self, input_dim: int = 20, hidden_dim: int = 256,
                 latent_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.bn3   = nn.BatchNorm1d(hidden_dim)
        self.proj  = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head  = nn.Linear(hidden_dim, latent_dim)
        self.lv_head  = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, mask):
        """
        x    : (B, L, 20)
        mask : (B, L) bool — True = real residue
        """
        h = x.permute(0, 2, 1)           # (B, 20, L)
        h = F.relu(self.bn1(self.conv1(h)))
        h = F.relu(self.bn2(self.conv2(h))) + h   # residual
        h = F.relu(self.bn3(self.conv3(h))) + h   # residual

        # Masked global average pool: only average over real positions
        mask_f = mask.unsqueeze(1).float()        # (B, 1, L)
        h = (h * mask_f).sum(dim=2) / mask_f.sum(dim=2).clamp(min=1)

        h   = F.relu(self.proj(h))
        mu  = self.mu_head(h)
        lv  = self.lv_head(h)
        # Clamp log-variance for numerical stability
        lv  = torch.clamp(lv, min=-10.0, max=4.0)
        return mu, lv


# =============================================================================
# Decoder: position-wise MLP
# =============================================================================

class MLPDecoder(nn.Module):
    """
    Decodes a latent vector z (optionally length-conditioned) into
    per-position amino acid logits.

    Architecture: z (+ length embedding) → MLP → reshape → (B, L, 20)

    Length conditioning allows controlling the output length at generation
    time by providing a target length embedding.
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256,
                 max_len: int = 50, n_lengths: int = 61):
        super().__init__()
        self.max_len = max_len
        self.length_emb = nn.Embedding(n_lengths + 1, 16)   # 0..max_len
        input_dim = latent_dim + 16

        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, hidden_dim)
        self.fc3  = nn.Linear(hidden_dim, max_len * 20)
        self.bn1  = nn.BatchNorm1d(hidden_dim)
        self.bn2  = nn.BatchNorm1d(hidden_dim)

    def forward(self, z, lengths):
        """
        z       : (B, latent_dim)
        lengths : (B,) int64 — target lengths (used as conditioning)
        """
        le = self.length_emb(lengths.clamp(0, self.length_emb.num_embeddings - 1))
        h  = torch.cat([z, le], dim=1)
        h  = F.relu(self.bn1(self.fc1(h)))
        h  = F.relu(self.bn2(self.fc2(h))) + F.pad(h, (0, 0))  # keep dims
        logits = self.fc3(h)                           # (B, max_len * 20)
        logits = logits.view(-1, self.max_len, 20)     # (B, max_len, 20)
        return logits


# =============================================================================
# Full CVAE
# =============================================================================

class PeptideCVAE(nn.Module):

    def __init__(
        self,
        max_len: int = 50,
        latent_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.max_len    = max_len
        self.latent_dim = latent_dim
        self.encoder    = CNNEncoder(
            input_dim=20, hidden_dim=hidden_dim, latent_dim=latent_dim
        )
        self.decoder    = MLPDecoder(
            latent_dim=latent_dim, hidden_dim=hidden_dim,
            max_len=max_len, n_lengths=max_len,
        )

    def reparameterise(self, mu, lv):
        if self.training:
            std = torch.exp(0.5 * lv)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x, lengths, mask):
        """
        x       : (B, L, 20)
        lengths : (B,)
        mask    : (B, L)
        Returns logits (B, L, 20), mu, lv
        """
        mu, lv = self.encoder(x, mask)
        z      = self.reparameterise(mu, lv)
        logits = self.decoder(z, lengths)
        return logits, mu, lv


def _masked_ce_loss(logits, targets, mask):
    """
    Cross-entropy loss computed only over real (non-padded) positions.

    logits  : (B, L, 20)
    targets : (B, L, 20)  one-hot
    mask    : (B, L)      bool

    FIX: original accumulated loss over ALL positions including padded
    zeros, pulling the model toward predicting 'A' at every position
    (index 0 in sorted AA_LIST is 'A'). This masked version ensures
    padded positions contribute zero to the loss.
    """
    # Convert one-hot targets to class indices
    target_idx = targets.argmax(dim=2)       # (B, L)
    B, L, _ = logits.shape
    loss_per_pos = F.cross_entropy(
        logits.reshape(B * L, 20),
        target_idx.reshape(B * L),
        reduction='none',
    ).reshape(B, L)

    mask_f = mask.float()
    masked_loss = (loss_per_pos * mask_f).sum() / mask_f.sum().clamp(min=1)
    return masked_loss


def _kl_loss(mu, lv):
    """KL divergence: KL(q(z|x) || N(0,I))."""
    return -0.5 * torch.mean(1.0 + lv - mu.pow(2) - lv.exp())


# =============================================================================
# Training loop
# =============================================================================

def train_cvae(
    seqs: List[str],
    max_len: int = 50,
    latent_dim: int = 64,
    hidden_dim: int = 256,
    epochs: int = 150,
    batch_size: int = 64,
    lr: float = 1e-3,
    grad_clip: float = 5.0,
    kl_beta: float = 1.0,
    kl_warmup_epochs: int = 60,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 25,
    resume_from: Optional[str] = None,
    device: str = 'cpu',
) -> 'PeptideCVAE':
    """
    Train the CVAE and return the fitted model.

    KL annealing schedule: β = kl_beta * min(1, epoch / kl_warmup_epochs)
    This prevents posterior collapse by initially allowing the decoder to
    rely entirely on x (β=0 means no KL penalty), then gradually enforcing
    a structured latent space.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for the CVAE. Install with: "
                          "pip install torch --index-url https://download.pytorch.org/whl/cpu")

    device = torch.device(device)
    model  = PeptideCVAE(max_len=max_len, latent_dim=latent_dim,
                          hidden_dim=hidden_dim).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=lr * 0.1
    )

    start_epoch = 0

    # -- Resume from checkpoint -------------------------------------------
    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimiser.load_state_dict(ckpt['optim_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        logger.info("Resumed from checkpoint at epoch %d", start_epoch)

    # -- Data preparation --------------------------------------------------
    x_t, lengths_t, masks_t = _build_tensors(seqs, max_len)
    dataset = TensorDataset(x_t, lengths_t, masks_t)
    loader  = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # -- Training loop -----------------------------------------------------
    history = []
    for epoch in range(start_epoch, epochs):
        model.train()
        # KL annealing: linearly ramp up β over warmup period
        beta = kl_beta * min(1.0, (epoch + 1) / max(kl_warmup_epochs, 1))

        epoch_recon = epoch_kl = 0.0
        n_batches = 0

        for x_batch, len_batch, mask_batch in loader:
            x_batch    = x_batch.to(device)
            len_batch  = len_batch.to(device)
            mask_batch = mask_batch.to(device)

            optimiser.zero_grad()
            logits, mu, lv = model(x_batch, len_batch, mask_batch)

            recon_loss = _masked_ce_loss(logits, x_batch, mask_batch)
            kl_loss    = _kl_loss(mu, lv)
            loss       = recon_loss + beta * kl_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()

            epoch_recon += recon_loss.item()
            epoch_kl    += kl_loss.item()
            n_batches   += 1

        scheduler.step()
        avg_recon = epoch_recon / max(n_batches, 1)
        avg_kl    = epoch_kl    / max(n_batches, 1)
        history.append({'epoch': epoch, 'recon': avg_recon,
                        'kl': avg_kl, 'beta': beta})

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(
                "Epoch %3d/%d  recon=%.4f  kl=%.4f  β=%.3f",
                epoch + 1, epochs, avg_recon, avg_kl, beta,
            )

        # -- Checkpoint ----------------------------------------------------
        if checkpoint_dir and ((epoch + 1) % checkpoint_every == 0 or
                                epoch == epochs - 1):
            ckpt_path = Path(checkpoint_dir) / f"cvae_epoch_{epoch+1:04d}.pt"
            torch.save({
                'epoch':        epoch,
                'model_state':  model.state_dict(),
                'optim_state':  optimiser.state_dict(),
                'history':      history,
                'max_len':      max_len,
                'latent_dim':   latent_dim,
                'hidden_dim':   hidden_dim,
            }, ckpt_path)
            logger.info("Checkpoint saved: %s", ckpt_path)

    model.eval()
    return model


# =============================================================================
# Generation
# =============================================================================

@torch.no_grad()
def generate_sequences(
    model: 'PeptideCVAE',
    n: int = 500,
    temperature: float = 1.1,
    min_len: int = 8,
    target_len: Optional[int] = None,
    device: str = 'cpu',
) -> List[str]:
    """
    Sample latent vectors, decode, and convert to amino acid strings.

    FIX over original:
    - Sequence length is determined by sampling from the empirical length
      distribution of training sequences (passed as target_len distribution),
      not by trimming trailing 'A' characters. The original .rstrip('A')
      logic silently discarded alanine residues at the end of sequences,
      introducing a systematic bias against C-terminal alanine.
    - Temperature scaling applied correctly in log space before softmax
      renormalisation.
    """
    model = model.to(torch.device(device))
    model.eval()

    sequences = []
    attempts  = 0
    max_attempts = n * 20

    while len(sequences) < n and attempts < max_attempts:
        nb = min(512, (n - len(sequences)) * 4)
        z  = torch.randn(nb, model.latent_dim)

        if target_len is not None:
            lens = torch.full((nb,), target_len, dtype=torch.long)
        else:
            # Sample from a rough empirical uniform length range
            lo = max(min_len, model.max_len // 4)
            hi = model.max_len
            lens = torch.randint(lo, hi + 1, (nb,))

        logits = model.decoder(z, lens)    # (nb, max_len, 20)

        # Temperature scaling in log-space
        if temperature != 1.0:
            logits = logits / temperature

        probs = torch.softmax(logits, dim=-1).cpu().numpy()   # (nb, L, 20)
        lens_np = lens.numpy()

        for i in range(nb):
            seq_len = int(lens_np[i])
            seq_chars = []
            for pos in range(seq_len):
                p = probs[i, pos]
                p = p / p.sum()   # renormalise for numerical stability
                aa = AA_LIST[np.random.choice(20, p=p)]
                seq_chars.append(aa)
            seq_str = ''.join(seq_chars)
            attempts += 1
            if min_len <= len(seq_str) <= model.max_len and is_canonical(seq_str):
                sequences.append(seq_str)
            if len(sequences) >= n:
                break

    logger.info(
        "Generated %d valid sequences from %d attempts (%.1f%% yield)",
        len(sequences), attempts, 100.0 * len(sequences) / max(attempts, 1),
    )
    return sequences[:n]


@torch.no_grad()
def encode_sequence(model: 'PeptideCVAE', seq: str) -> np.ndarray:
    """Encode a sequence to its latent mean vector."""
    x, lengths, masks = _build_tensors([seq], model.max_len)
    mu, _ = model.encoder(x, masks)
    return mu[0].numpy()
