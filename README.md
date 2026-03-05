# FHE-Flicker 2.0

Privacy-Preserving Federated Learning with Fully Homomorphic Encryption for class-imbalance correction on CIFAR-10.

---

## Overview

FHE-Flicker extends the **Flicker** federated rebalancing algorithm with **CKKS fully homomorphic encryption** (via [TenSEAL](https://github.com/OpenMined/TenSEAL)), so that the server never observes plaintext feature vectors from any client at any stage of the training pipeline.

The core problem: real-world federated datasets are often **class-imbalanced** — some clients have far more samples of certain classes than others. Naively training on this data leads to biased global models. Flicker corrects imbalance by:
1. **Oversampling** minority classes on clients where they are underrepresented.
2. **Undersampling** redundant majority-class samples identified via cross-client cosine similarity.

FHE-Flicker makes step 2 privacy-preserving: cross-client similarity comparisons are computed entirely in the encrypted domain. This repository also includes two research extensions (Options 6 and 7) that push the FHE coverage further.

---

## Contents

```
fhe-flicker-2.0/
├── notebooks/
│   ├── fhe-flicker-tenseal-cifar10.ipynb          # Main FHE-Flicker notebook
│   ├── torch-data-flicker-n-clients-cifar10-ipynb.ipynb   # 3-client plaintext baseline
│   └── torch-data-flicker-2n-clients-cifar10-ipynb.ipynb  # 6-client plaintext baseline
├── examples/
│   ├── option6_fhe_fedavg.py                      # Encrypted gradient aggregation
│   └── option7_private_redundancy.py              # Fully private redundancy protocol
└── requirements.txt
```

---

## Background

### Federated Learning

In federated learning, a set of clients each hold a private local dataset. A central server coordinates training without directly accessing any client's data. The standard FedAvg protocol:

1. Server broadcasts the current global model.
2. Each client trains locally and sends **gradient updates** (or model weights) back to the server.
3. Server averages the updates and applies them to the global model.
4. Repeat for multiple rounds.

The central challenge addressed here is that local datasets are often **non-IID** — heterogeneous in both size and class distribution. FHE-Flicker corrects this before training begins.

### The CKKS Scheme

CKKS (Cheon-Kim-Kim-Song) is an approximate arithmetic fully homomorphic encryption scheme suited for real-valued data. Key properties used in this project:

| Operation | Depth consumed | Notes |
|-----------|---------------|-------|
| Addition (enc + enc) | 0 | Exact; noise grows as O(√n) |
| Multiplication (enc × enc) | 1 | Main cost; increases noise |
| Rotation (enc.rotate(k)) | 0 | Uses Galois keys; enables slot-wise reductions |
| Plaintext multiply (enc × plain) | 1 | Same depth cost as enc × enc |

**Context parameters** used throughout this project:

```python
poly_modulus_degree = 8192     # Ring dimension N
                               # → N/2 = 4096 CKKS slots per ciphertext
coeff_mod_bit_sizes = [60, 40, 40, 60]
                               # Two 40-bit middle primes → depth-2 budget
global_scale = 2**40           # ~12 decimal digits of floating-point precision
Security level ≥ 128-bit       # HE-Standard compliant
```

---

## The Flicker Algorithm

### 1. Client Distribution Analysis

Each client computes its **local distribution (LD)** — a 10-element class-count histogram. The server computes the **global distribution (GD)** by summing all LDs.

The **dominant client** for a given round is the one whose LD is most aligned with the GD (highest cosine similarity). This is the client contributing the most to global imbalance.

### 2. Oversampling (plaintext, first domination only)

When a client dominates for the first time, its minority-class samples are oversampled:
- For each class with count < `lin_threshold × max_count`:
  - Compute how many samples are needed to balance.
  - Randomly sample that many existing minority examples.
  - Apply random augmentation (horizontal flip, small rotation).
  - Add augmented copies to the local dataset.

This step involves only the dominant client's own data and requires no cross-client communication.

### 3. Global Redundancy Undersampling (FHE-secured)

When a client has already been oversampled, redundant majority-class samples are removed using cross-client cosine similarity — computed entirely in the encrypted domain.

**Protocol:**

```
1. Dominant client: identify majority-class candidates (buffer)
                    extract L2-normalised 512-dim feature vectors

2. Other clients:   encrypt their feature vectors with CKKS
                    send ciphertexts to the server

3. Server:          for each buffer vector q, for each encrypted feature e:
                        enc_sim = enc_e.dot(q.tolist())
                    (q is a plaintext list in the base version — see Option 7
                     for the fully private upgrade)

4. Key-holder:      decrypt enc_sim → scalar cosine similarity
                    if average_similarity ≥ η: mark q for removal

5. Dominant client: remove flagged buffer samples from local dataset
```

**FHE privacy guarantee:** the server never observes raw feature vectors from non-dominant clients. It receives only CKKS ciphertexts and returns CKKS ciphertexts.

### 4. Iterative Rebalancing

The algorithm repeats for up to `MAX_DOM_ROUNDS = 6` rounds. Each round selects the current dominant client (excluding saturated ones) and applies oversampling or undersampling as appropriate. A client is considered saturated when its local imbalance ratio `LIn = min_class_count / max_class_count` reaches `LIn_MAX = 0.75`.

---

## FHE Operations in the Main Notebook

### Global Redundancy Check

The cross-client cosine similarity between the dominant client's buffer vector `q` and another client's encrypted feature `enc_e` is computed as:

```python
enc_sim = enc_e.dot(q.tolist())   # cipher × plaintext dot product
sim     = enc_sim.decrypt()[0]
```

This uses **depth 1** (one 40-bit prime), leaving one level for other operations within the same CKKS context.

### Encrypted Inference

At test time, the user encrypts their ResNet-18 feature vector and the server evaluates the linear classifier head entirely in the encrypted domain:

```python
enc_feat = ts.ckks_vector(ctx, feat.tolist())          # user encrypts

for i in range(10):                                     # server computes
    enc_logit_i = enc_feat.dot(W[i]) + b[i]            # cipher × plain

logits = [enc_logit_i.decrypt()[0] for ...]            # user decrypts
prediction = np.argmax(logits)
```

The server processes only ciphertexts and returns ciphertexts. It never sees the raw feature vector or the predicted class.

---

## Option 6 — Encrypted Gradient Aggregation (FHE-FedAvg)

**File:** `examples/option6_fhe_fedavg.py`
**Notebook section:** Section 14 of `fhe-flicker-tenseal-cifar10.ipynb`

### The Limitation

Up to this point, FHE protects feature vectors at inference and redundancy-check time. The training phase is still entirely plaintext: if clients send raw gradients to the server, gradient-inversion attacks ([Zhu et al., 2019](https://arxiv.org/abs/1906.08935)) can reconstruct training images from those gradients with high fidelity.

### The Extension

Clients encrypt their gradient vectors before sending to the server. The server aggregates ciphertexts using homomorphic addition and ships the result to the key-holder for decryption and update.

### Protocol

```
Round r:
  1. Server broadcasts global model θ  (plaintext — standard FL)

  2. Each client i:
       train local copy of θ for 1 epoch on local data
       compute gradient ∇_i (5130 params: head.weight + head.bias)
       split into chunks: chunk_0 = ∇_i[0:4096], chunk_1 = ∇_i[4096:5130]
       encrypt: enc_grad_i = [CKKS(chunk_0), CKKS(chunk_1)]
       send enc_grad_i to server  ← server sees 2 ciphertexts, nothing more

  3. Server aggregates (homomorphic addition, depth = 0):
       for k in {0, 1}:
           agg_k = enc_grad_0_k + enc_grad_1_k + … + enc_grad_n_k

  4. Key-holder decrypts:
       flat = [agg_k.decrypt() for agg_k in agg_chunks]
       θ ← θ − lr · (1/n) · flat          (FedAvg update)
```

### Why the Slot Limit Is Not a Problem

The linear head has **5,130 parameters** (10×512 + 10), which exceeds the 4,096 CKKS slots available with `poly_modulus_degree = 8192`. The solution is **gradient chunking**:

```
grad_flat (5130,) float32
    chunk_0 → grad_flat[0    : 4096]   (4096 elements, 1 full CKKS vector)
    chunk_1 → grad_flat[4096 : 5130]   (1034 elements, TenSEAL zero-pads rest)
```

The server aggregates each chunk independently. The key-holder concatenates and trims to 5,130 before applying the update.

### Depth Budget

CKKS **addition** consumes **zero multiplicative levels**. The entire FedAvg aggregation (summing n clients' gradient ciphertexts) uses depth 0. The existing `[60, 40, 40, 60]` context handles it without any parameter changes.

### Running the Example

```bash
python examples/option6_fhe_fedavg.py
```

Expected output (5 rounds, 3 clients, Dirichlet α = 0.3):

```
[1/5] Setting up CKKS context...
[2/5] Loading CIFAR-10...
[3/5] Distributing data across 3 clients...
[4/5] Verifying gradient encryption round-trip...
  Max absolute error: ~1e-7  (negligible for SGD)
[5/5] Running FHE-FedAvg (5 rounds)...

── FHE-FedAvg Round 0 ──────────────────────────────────────
  Client 0: 5130 params → 2 CKKS chunk(s) encrypted
  Client 1: 5130 params → 2 CKKS chunk(s) encrypted
  Client 2: 5130 params → 2 CKKS chunk(s) encrypted
  Server: aggregated 3 clients (homomorphic addition, depth=0)
  Test accuracy: 0.42xx
...
```

---

## Option 7 — Private Redundancy Protocol

**File:** `examples/option7_private_redundancy.py`
**Notebook section:** Section 15 of `fhe-flicker-tenseal-cifar10.ipynb`

### The Limitation

In the base FHE-Flicker global undersampling step, the dominant client's buffer feature vectors are uploaded as a **plaintext list** to the server:

```python
enc_sim = enc_feat.dot(buf_vec.tolist())   # buf_vec: PLAIN Python list
```

The server reads `buf_vec` directly. Even though these are 512-dim ResNet embeddings rather than raw images, feature-inversion attacks can reconstruct recognisable images from such vectors. The **dominant client's privacy is compromised**.

### The Fix

Both the dominant client's buffer vectors and the other clients' feature vectors are CKKS-encrypted before the server touches them. The server computes a fully homomorphic dot product:

```python
enc_sim = fhe_inner_product(enc_buf, enc_feat)   # BOTH sides encrypted
sim     = enc_sim.decrypt()[0]                   # key-holder only
```

### The `fhe_inner_product` Kernel

```python
def fhe_inner_product(enc_a, enc_b, n=512):
    # Step 1: element-wise cipher × cipher multiply  (depth 1)
    enc_prod = enc_a * enc_b

    # Step 2: rotation-based sum into slot 0  (depth 0)
    result = enc_prod
    shift  = 1
    while shift < n:                   # 9 iterations for n=512
        result = result + result.rotate(shift)
        shift <<= 1

    return result                      # result.decrypt()[0] = dot product
```

**Why the rotation-based sum works:**

After `enc_prod = enc_a * enc_b`, slot `i` holds `a[i] * b[i]`. We want `Σᵢ a[i]*b[i]`. The rotation trick:

```
After rotate(1):   slot 0 = p[0] + p[1]
After rotate(2):   slot 0 = p[0] + p[1] + p[2] + p[3]
After rotate(4):   slot 0 = p[0] + … + p[7]
…
After rotate(256): slot 0 = p[0] + p[1] + … + p[511]  ✓
```

Each `.rotate(k)` is a Galois automorphism: it cyclically shifts all slots left by k positions. The Galois keys pre-generated in `setup_ckks_context()` authorise this operation at zero extra depth.

### Depth Budget Comparison

| Method | Step | Depth |
|--------|------|-------|
| Section 9 (enc × plain) | `enc_feat.dot(plain_buf)` internally: element-wise multiply | 1 |
| Option 7 (enc × enc) | `enc_a * enc_b` (cipher × cipher multiply) | 1 |
| Both | Rotation-based sum (log₂(512) = 9 additions) | 0 |
| **Both total** | | **1** |

**The multiplicative depth is identical.** The upgrade from cipher×plain to cipher×cipher costs zero extra CKKS levels. The only change is that cipher×cipher multiplication increases noise slightly more than cipher×plain — but at `scale = 2^40` and `depth = 1`, this is well within the noise budget (verified by the correctness check in the script).

### Privacy Comparison

| Stage | Section 9 | Option 7 |
|-------|-----------|----------|
| Other clients' features | Encrypted ✓ | Encrypted ✓ |
| Dominant client's buffer vectors | **Plaintext ✗** | **Encrypted ✓** |
| Server's view | buf_vec in plaintext | Only ciphertexts |
| Server learns | buf_vec (512 floats each) | At most 1 binary bit per candidate |

### Running the Example

```bash
python examples/option7_private_redundancy.py
```

Expected output:

```
[1/5] Setting up CKKS context...
[2/5] Loading CIFAR-10...
[3/5] Distributing data across 3 clients...
[4/5] Correctness check: fhe_inner_product vs numpy dot product...
  Plaintext dot product  :  0.0234567
  FHE dot product        :  0.0234569
  Absolute error         : 2.4e-07
[5/5a] Section 9 baseline — cipher × PLAINTEXT buffer vectors...
[5/5b] Option 7 — cipher × CIPHER buffer vectors (fully private)...

══════════════════════════════════════════════════════════════
  Option 7 — Private Redundancy Protocol: Summary
  Method                          Removed   Server sees buf_vec?
  Section 9  (cipher × plain)        XX     YES  (plaintext list)
  Option 7   (cipher × cipher)       XX     NO   (encrypted)
══════════════════════════════════════════════════════════════
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch ≥ 1.13 (CPU or CUDA)
- TenSEAL ≥ 0.3.14
- torchvision ≥ 0.14

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
numpy
torch
torchvision
scikit-learn
pandas
matplotlib
jupyter
tenseal
```

### Environment setup (recommended)

```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

---

## Usage

### Interactive notebook

```bash
jupyter notebook notebooks/fhe-flicker-tenseal-cifar10.ipynb
```

Run cells sequentially. Sections 1–13 cover the base FHE-Flicker experiment.
Section 14 adds Option 6 (FHE-FedAvg).
Section 15 adds Option 7 (private redundancy).

### Standalone scripts

**Option 6 — Encrypted Gradient Aggregation:**
```bash
python examples/option6_fhe_fedavg.py
```

**Option 7 — Private Redundancy Protocol:**
```bash
python examples/option7_private_redundancy.py
```

Both scripts download CIFAR-10 automatically on first run (`./data/`).
A GPU is used if available; CPU works but is slower (~5–10 min per FedAvg round on CPU).

---

## Architecture

```
┌────────────────────────────────────────────────────────┐
│  CIFARClassifier                                       │
│  ┌──────────────────────────┐   ┌────────────────────┐ │
│  │  ResNet-18 backbone       │ → │  Linear head       │ │
│  │  (frozen, ImageNet init) │   │  nn.Linear(512,10) │ │
│  │  output: 512-dim vector  │   │  (trainable)       │ │
│  └──────────────────────────┘   └────────────────────┘ │
└────────────────────────────────────────────────────────┘

Input:  (B, 3, 224, 224)  CIFAR-10 images (resized)
Output: (B, 10)           class logits
```

**Why freeze the backbone?**
ResNet-18 was pre-trained on ImageNet. Its features transfer well to CIFAR-10 without fine-tuning. Freezing means:
- Only 5,130 head parameters need gradients (Option 6).
- Inference is fast and deterministic.
- Encrypted inference evaluates only the linear layer (not the full backbone), keeping FHE depth requirements at 1.

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_CLIENTS` | 3 | Number of federated clients |
| `DIRICHLET_ALPHA` | 0.3–0.5 | Imbalance severity (lower = more imbalanced) |
| `FHE_SAMPLE_SIZE` | 200 | Encrypted features sampled per non-dominant client |
| `theta` | 0.2 | Fraction of majority samples added to the buffer |
| `eta` | 0.6 | Cosine similarity threshold for redundancy removal |
| `lin_threshold` | 0.9 | Defines "majority class": count ≥ threshold × max_count |
| `MAX_DOM_ROUNDS` | 6 | Max rounds a client can be dominant before saturation |
| `LIn_MAX` | 0.75 | Local imbalance saturation threshold |

---

## CKKS Depth Accounting

| Operation | Where used | Depth |
|-----------|-----------|-------|
| `enc_feat.dot(plain_vec)` | Global redundancy check (Sec 9) | 1 |
| `enc_feat.dot(W[i]) + b[i]` | Encrypted inference (Sec 12–13) | 1 |
| `enc_grad_0 + enc_grad_1 + …` | FHE-FedAvg aggregation (Option 6) | **0** |
| `enc_buf * enc_feat` | Private redundancy kernel (Option 7) | 1 |
| rotation-based sum | Option 7 inner product | 0 |

All operations fit within the depth-2 budget of `[60, 40, 40, 60]`. No context changes are needed for Options 6 or 7.

---

## References

- **Flicker:** *Flicker: Federated Learning for Class-Imbalanced Data with Application to Healthcare* — original algorithm this project is based on.
- **FedAvg:** McMahan et al. (2017). [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629).
- **CKKS:** Cheon, Kim, Kim, Song (2017). [Homomorphic Encryption for Arithmetic of Approximate Numbers](https://eprint.iacr.org/2016/421).
- **TenSEAL:** [github.com/OpenMined/TenSEAL](https://github.com/OpenMined/TenSEAL) — CKKS implementation used in this project.
- **Gradient inversion:** Zhu et al. (2019). [Deep Leakage from Gradients](https://arxiv.org/abs/1906.08935).
- **Feature inversion:** Mahendran & Vedaldi (2015). [Understanding Deep Image Representations by Inverting Them](https://arxiv.org/abs/1412.0035).
- **HE-Standard:** [HomomorphicEncryption.org](https://homomorphicencryption.org) — security parameter guidelines.

---

## Licence

MIT
