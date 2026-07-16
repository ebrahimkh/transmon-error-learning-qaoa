# %% [markdown]
# # Three-qubit
#
# This notebook contains the final three-qutrit / three-qubit code used for the paper's local-response, pair-probe, leakage-aware, oracle-assisted diagnostic, and non-oracle CDR-style results.

# %% [markdown]
# # Clean final Colab code
#
# Run in Colab with **Runtime → Change runtime type → GPU**. The code uses GPU for PyTorch training and CPU/SciPy for the qutrit/Liouvillian simulation.
## Citation

#If you use this code in academic work, please cite the associated manuscript:

#Ebrahim Khaleghian and Özgür E. Müstecaplıoğlu,
#"Physics-Informed Learning of Effective Error Processes from Limited Noisy
#Transmon Measurements for Robust QAOA Reliability", 2026.
# %% Cell 3
# ============================================================
# CELL 1 — Imports and 3-qutrit global setup
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import product
from scipy.linalg import expm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(7)
torch.manual_seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# We are now interested only in 3 qutrit transmons
N_QUBITS = 3
QUTRIT_DIM = 3
HILBERT_DIM = QUTRIT_DIM ** N_QUBITS

print("N_QUBITS:", N_QUBITS)
print("Hilbert dimension:", HILBERT_DIM)
print("Liouvillian dimension:", HILBERT_DIM**2)

# %% Cell 4
import os, glob

# Check the expected folder
FINAL_DIR = "outputs/three_qubit_qaoa_FINAL_scaled_run"

print("Folder exists:", os.path.exists(FINAL_DIR))
print("Folder path:", FINAL_DIR)

if os.path.exists(FINAL_DIR):
    files = sorted(glob.glob(os.path.join(FINAL_DIR, "*")))
    print("Number of files:", len(files))
    for f in files:
        print(os.path.basename(f))
else:
    print("Expected folder does not exist.")

# %% Cell 5
# ============================================================
# CELL 2 — Basic helpers
# ============================================================

def dagger(A):
    return A.conj().T

def kron_list(ops):
    out = np.array([[1.0 + 0.0j]])
    for op in ops:
        out = np.kron(out, op)
    return out

def basis(dim, idx):
    v = np.zeros((dim, 1), dtype=complex)
    v[idx, 0] = 1.0
    return v

def normalize_ket(psi):
    return psi / np.linalg.norm(psi)

def ket_to_rho(psi):
    return psi @ dagger(psi)

def two_pi(x):
    return 2.0 * np.pi * x

def trace_expectation(rho, O):
    return float(np.real_if_close(np.trace(rho @ O)))

def clean_density_matrix(rho):
    rho = 0.5 * (rho + dagger(rho))
    tr = np.trace(rho)
    if abs(tr) > 1e-12:
        rho = rho / tr
    return rho

def vec(rho):
    return rho.reshape((-1, 1), order="F")

def unvec(v, dim):
    return v.reshape((dim, dim), order="F")

# %% Cell 6
# ============================================================
# CELL 3 — 3-qutrit transmon operators
# ============================================================

I3 = np.eye(3, dtype=complex)
I_sys = np.eye(HILBERT_DIM, dtype=complex)

# Single-qutrit annihilation operator
a_single = np.zeros((3, 3), dtype=complex)
for n in range(1, 3):
    a_single[n - 1, n] = np.sqrt(n)

adag_single = dagger(a_single)
n_single = adag_single @ a_single

def embed_single_qutrit_operator(op, q, N=N_QUBITS):
    """
    Embed a single-qutrit operator on qutrit q.
    q is zero-indexed.
    """
    ops = []
    for j in range(N):
        ops.append(op if j == q else I3)
    return kron_list(ops)

a_ops = [embed_single_qutrit_operator(a_single, q) for q in range(N_QUBITS)]
adag_ops = [dagger(aq) for aq in a_ops]
n_ops = [adag_ops[q] @ a_ops[q] for q in range(N_QUBITS)]

print("Built 3-qutrit operators.")
print("a_ops[0] shape:", a_ops[0].shape)

# %% Cell 7
# ============================================================
# CELL 4 — Computational subspace and embedded Pauli operators
# ============================================================

def qutrit_product_ket(levels):
    kets = [basis(3, ell) for ell in levels]
    return kron_list(kets)

def bitstrings(N):
    return list(product([0, 1], repeat=N))

def comp_ket_from_bitstring(bits):
    return qutrit_product_ket(bits)

comp_bitstrings = bitstrings(N_QUBITS)

comp_kets = {
    "".join(map(str, bits)): comp_ket_from_bitstring(bits)
    for bits in comp_bitstrings
}

P_comp = np.zeros((HILBERT_DIM, HILBERT_DIM), dtype=complex)
for ket in comp_kets.values():
    P_comp += ket @ dagger(ket)

# Single-qubit Pauli matrices in computational basis
I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)

def embedded_pauli_on_qubit(q, P2, N=N_QUBITS):
    """
    Embed a qubit Pauli operator into the N-qutrit Hilbert space,
    acting only inside the computational subspace.

    q is zero-indexed.
    """
    O = np.zeros((3**N, 3**N), dtype=complex)

    for bits_bra in bitstrings(N):
        for bits_ket in bitstrings(N):

            ok = True
            for j in range(N):
                if j != q and bits_bra[j] != bits_ket[j]:
                    ok = False
                    break

            if not ok:
                continue

            coeff = P2[bits_bra[q], bits_ket[q]]

            if abs(coeff) > 1e-15:
                ket_bra = comp_ket_from_bitstring(bits_bra)
                ket_ket = comp_ket_from_bitstring(bits_ket)
                O += coeff * ket_bra @ dagger(ket_ket)

    return O

measurement_ops = {}

for q in range(N_QUBITS):
    measurement_ops[q] = {
        "X": embedded_pauli_on_qubit(q, X2),
        "Y": embedded_pauli_on_qubit(q, Y2),
        "Z": embedded_pauli_on_qubit(q, Z2),
    }

measurement_labels = ["X", "Y", "Z"]

print("Computational subspace dimension:", int(np.real_if_close(np.trace(P_comp))))
print("Example Pauli shape:", measurement_ops[0]["X"].shape)

# %% Cell 8
# ============================================================
# CELL 5 — Local tomography states
# ============================================================

q0 = basis(3, 0)
q1 = basis(3, 1)

q_plus = normalize_ket((q0 + q1) / np.sqrt(2))
q_minus = normalize_ket((q0 - q1) / np.sqrt(2))
q_plus_i = normalize_ket((q0 + 1j * q1) / np.sqrt(2))
q_minus_i = normalize_ket((q0 - 1j * q1) / np.sqrt(2))

local_state_dict = {
    "0": q0,
    "1": q1,
    "+": q_plus,
    "-": q_minus,
    "+i": q_plus_i,
    "-i": q_minus_i,
}

expected_bloch = {
    "0":  np.array([0.0, 0.0, 1.0]),
    "1":  np.array([0.0, 0.0, -1.0]),
    "+":  np.array([1.0, 0.0, 0.0]),
    "-":  np.array([-1.0, 0.0, 0.0]),
    "+i": np.array([0.0, 1.0, 0.0]),
    "-i": np.array([0.0, -1.0, 0.0]),
}

tomography_input_labels = ["0", "1", "+", "-", "+i", "-i"]

def prepare_local_tomography_state(target_qubit, local_label, spectator_label="0", N=N_QUBITS):
    """
    Prepare an N-qutrit product state for local tomography.
    target_qubit is zero-indexed.
    """
    states = []

    for q in range(N):
        if q == target_qubit:
            states.append(local_state_dict[local_label])
        else:
            states.append(local_state_dict[spectator_label])

    return kron_list(states)

psi_test = prepare_local_tomography_state(1, "+")
print("Tomography ket shape:", psi_test.shape)
print("Norm:", np.linalg.norm(psi_test))

# %% Cell 9
# ============================================================
# CELL 6 — 3-transmon hidden device sampler
# ============================================================

def chain_edges(N):
    return [(j, j + 1) for j in range(N - 1)]

def full_edges(N):
    return [(i, j) for i in range(N) for j in range(i + 1, N)]

def sample_hidden_device_3q(rng=None, topology="chain"):
    """
    Hidden 3-qutrit-transmon device parameters.

    Units:
        angular frequencies in rad/ns
        times in ns
    """
    if rng is None:
        rng = np.random.default_rng()

    N = 3
    params = {}
    params["N"] = N

    # Detunings and anharmonicities
    Delta = []
    alpha = []

    # Slight frequency separation across the chain
    base_detunings_GHz = [0.000, 0.080, 0.160]

    for j in range(N):
        Delta_j_GHz = rng.normal(base_detunings_GHz[j], 0.006)
        alpha_j_GHz = rng.normal(0.220 + 0.004 * j, 0.008)

        Delta.append(two_pi(Delta_j_GHz))
        alpha.append(two_pi(alpha_j_GHz))

    params["Delta"] = np.array(Delta, dtype=np.float64)
    params["alpha"] = np.array(alpha, dtype=np.float64)

    if topology == "chain":
        edges = chain_edges(N)
    elif topology == "full":
        edges = full_edges(N)
    else:
        raise ValueError("topology must be 'chain' or 'full'.")

    params["edges"] = edges

    # Couplings and residual ZZ shifts
    g = {}
    zeta = {}

    for (i, j) in edges:
        g[(i, j)] = two_pi(rng.normal(0.0030, 0.0005))
        zeta[(i, j)] = two_pi(rng.normal(0.0008, 0.0002))

    params["g"] = g
    params["zeta"] = zeta

    # Hidden pulse imperfections
    params["amp_scale"] = rng.normal(1.0, 0.025, size=N)
    params["phase_error"] = rng.normal(0.0, 0.025, size=N)

    # Decoherence times
    params["T1"] = rng.uniform(25000.0, 60000.0, size=N)
    params["Tphi"] = rng.uniform(20000.0, 50000.0, size=N)

    # Readout assignment errors
    params["r01"] = rng.uniform(0.005, 0.040, size=N)
    params["r10"] = rng.uniform(0.005, 0.050, size=N)

    # Reserved for later richer noise taxonomy
    params["noise_family"] = "baseline_3q_chain_open_system"

    return params


rng = np.random.default_rng(123)
p = sample_hidden_device_3q(rng=rng, topology="chain")

print("Sampled 3-transmon hidden device")
print("Delta / 2pi GHz:", p["Delta"] / (2*np.pi))
print("alpha / 2pi GHz:", p["alpha"] / (2*np.pi))
print("edges:", p["edges"])
print("g / 2pi GHz:", {k: v/(2*np.pi) for k, v in p["g"].items()})
print("zeta / 2pi GHz:", {k: v/(2*np.pi) for k, v in p["zeta"].items()})
print("T1:", p["T1"])
print("Tphi:", p["Tphi"])

# %% Cell 10
# ============================================================
# CELL 7 — Hamiltonian, Liouvillian, and cached pulse superoperators
# ============================================================

def build_H0(params):
    N = params["N"]
    H = np.zeros((3**N, 3**N), dtype=complex)

    for j in range(N):
        Delta_j = params["Delta"][j]
        alpha_j = params["alpha"][j]
        n_j = n_ops[j]

        H += Delta_j * n_j
        H += -0.5 * alpha_j * (n_j @ (n_j - I_sys))

    for (i, j) in params["edges"]:
        g_ij = params["g"][(i, j)]
        zeta_ij = params["zeta"][(i, j)]

        H += g_ij * (adag_ops[i] @ a_ops[j] + a_ops[i] @ adag_ops[j])
        H += zeta_ij * (n_ops[i] @ n_ops[j])

    return H


def control_hamiltonian(params, qubit, Omega=0.0, phase=0.0):
    """
    Square-pulse control Hamiltonian on one target transmon.
    qubit is zero-indexed.
    """
    amp_scale = params["amp_scale"][qubit]
    phase_error = params["phase_error"][qubit]

    effective_Omega = amp_scale * Omega
    effective_phase = phase + phase_error

    aq = a_ops[qubit]
    adagq = adag_ops[qubit]

    Hx = 0.5 * effective_Omega * np.cos(effective_phase) * (aq + adagq)
    Hy = 0.5 * effective_Omega * np.sin(effective_phase) * (1j * (adagq - aq))

    return Hx + Hy


def collapse_operators(params):
    """
    Markovian open-system collapse operators:
        T1 relaxation + pure dephasing
    """
    N = params["N"]
    cops = []

    for j in range(N):
        gamma1 = 1.0 / params["T1"][j]
        gammaphi = 1.0 / params["Tphi"][j]

        cops.append(np.sqrt(gamma1) * a_ops[j])
        cops.append(np.sqrt(gammaphi) * n_ops[j])

    return cops


def liouvillian(H, cops):
    dim = H.shape[0]
    I = np.eye(dim, dtype=complex)

    L = -1j * (np.kron(I, H) - np.kron(H.T, I))

    for C in cops:
        CdC = dagger(C) @ C

        L += np.kron(C.conj(), C)
        L += -0.5 * np.kron(I, CdC)
        L += -0.5 * np.kron(CdC.T, I)

    return L


def build_pulse_superoperator(
    params,
    qubit,
    theta=np.pi / 2,
    phase=0.0,
    T=40.0,
    include_decoherence=True
):
    """
    Build and cache the superoperator for one target-qubit pulse.

    For 3 qutrits:
        rho dimension = 27
        vec(rho) dimension = 729
        superoperator dimension = 729 x 729
    """
    H0 = build_H0(params)

    Omega = theta / T
    Hc = control_hamiltonian(params, qubit=qubit, Omega=Omega, phase=phase)
    H = H0 + Hc

    if include_decoherence:
        cops = collapse_operators(params)
        L = liouvillian(H, cops)
        S = expm(L * T)
    else:
        U = expm(-1j * H * T)
        S = np.kron(U.conj(), U)

    return S


def build_all_pulse_superoperators(
    params,
    theta=np.pi / 2,
    phase=0.0,
    T=40.0,
    include_decoherence=True
):
    """
    Build one pulse superoperator per target qubit.
    """
    S_list = []

    for q in range(params["N"]):
        S_q = build_pulse_superoperator(
            params=params,
            qubit=q,
            theta=theta,
            phase=phase,
            T=T,
            include_decoherence=include_decoherence
        )
        S_list.append(S_q)

    return S_list


def evolve_rho_with_superoperator(rho0, S):
    rho_vec = S @ vec(rho0)
    rho = unvec(rho_vec, rho0.shape[0])
    return clean_density_matrix(rho)


def leakage_probability(rho):
    """
    Probability outside computational subspace.
    """
    p_comp = float(np.real_if_close(np.trace(P_comp @ rho)))
    return float(np.clip(1.0 - p_comp, 0.0, 1.0))


# Sanity check
rng = np.random.default_rng(222)
p = sample_hidden_device_3q(rng=rng)

t0 = time.time()
S_list = build_all_pulse_superoperators(p, theta=np.pi/2, T=40.0)
print("Built", len(S_list), "pulse superoperators in", time.time() - t0, "s")

psi0 = prepare_local_tomography_state(0, "0")
rho0 = ket_to_rho(psi0)
rho1 = evolve_rho_with_superoperator(rho0, S_list[0])

print("Trace:", np.trace(rho1))
print("Leakage:", leakage_probability(rho1))

# %% Cell 11
# ============================================================
# CELL 8 — Measurement, readout error, finite shots
# ============================================================

def ideal_measurement_probabilities(rho, qubit=0, basis_label="Z"):
    O = measurement_ops[qubit][basis_label]
    m = trace_expectation(rho, O)
    m = float(np.clip(m, -1.0, 1.0))

    p_plus = 0.5 * (1.0 + m)
    p_minus = 0.5 * (1.0 - m)

    return p_plus, p_minus, m


def apply_readout_error(p_plus, p_minus, params, qubit=0):
    r01 = params["r01"][qubit]
    r10 = params["r10"][qubit]

    p_plus_meas = (1.0 - r01) * p_plus + r10 * p_minus
    p_minus_meas = r01 * p_plus + (1.0 - r10) * p_minus

    p_plus_meas = float(np.clip(p_plus_meas, 0.0, 1.0))
    p_minus_meas = float(np.clip(p_minus_meas, 0.0, 1.0))

    s = p_plus_meas + p_minus_meas
    if s <= 1e-12:
        return 0.5, 0.5

    return p_plus_meas / s, p_minus_meas / s


def exact_measured_expectation_with_readout(rho, params, qubit=0, basis_label="Z"):
    p_plus, p_minus, _ = ideal_measurement_probabilities(rho, qubit, basis_label)
    p_plus_meas, p_minus_meas = apply_readout_error(p_plus, p_minus, params, qubit)
    return float(p_plus_meas - p_minus_meas)


def sample_binary_measurement(rho, params, qubit=0, basis_label="Z", shots=1024, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    p_plus, p_minus, _ = ideal_measurement_probabilities(rho, qubit, basis_label)
    p_plus_meas, p_minus_meas = apply_readout_error(p_plus, p_minus, params, qubit)

    counts = rng.multinomial(shots, [p_plus_meas, p_minus_meas])
    n_plus, n_minus = counts

    m_hat = (n_plus - n_minus) / shots

    return float(m_hat)

# %% Cell 12
# ============================================================
# CELL 9 — 3-qubit tomography and local-channel labels
# ============================================================

FULL_TOMO_DIM = N_QUBITS * len(tomography_input_labels) * len(measurement_labels)
LABEL_DIM = N_QUBITS * 12

print("FULL_TOMO_DIM:", FULL_TOMO_DIM)
print("LABEL_DIM:", LABEL_DIM)


def finite_shot_output_bloch_for_local_input_full_cached(
    params,
    S_list,
    target_qubit=0,
    local_label="0",
    spectator_label="0",
    shots=1024,
    rng=None
):
    """
    Use cached pulse superoperator S_list[target_qubit].
    """
    if rng is None:
        rng = np.random.default_rng()

    psi0 = prepare_local_tomography_state(
        target_qubit=target_qubit,
        local_label=local_label,
        spectator_label=spectator_label
    )
    rho0 = ket_to_rho(psi0)

    rho_out = evolve_rho_with_superoperator(rho0, S_list[target_qubit])

    r_hat = []

    for basis_label in measurement_labels:
        m_hat = sample_binary_measurement(
            rho=rho_out,
            params=params,
            qubit=target_qubit,
            basis_label=basis_label,
            shots=shots,
            rng=rng
        )
        r_hat.append(m_hat)

    return np.array(r_hat, dtype=np.float32)


def generate_full_tomography_vector_cached(
    params,
    S_list,
    shots=1024,
    spectator_label="0",
    rng=None
):
    """
    Full finite-shot local tomography vector for 3 qubits.

    Dimension:
        3 qubits * 6 states * 3 bases = 54
    """
    if rng is None:
        rng = np.random.default_rng()

    features = []

    for target_qubit in range(N_QUBITS):
        for state_label in tomography_input_labels:
            r_hat = finite_shot_output_bloch_for_local_input_full_cached(
                params=params,
                S_list=S_list,
                target_qubit=target_qubit,
                local_label=state_label,
                spectator_label=spectator_label,
                shots=shots,
                rng=rng
            )
            features.extend(list(r_hat))

    return np.array(features, dtype=np.float32)


def exact_output_bloch_for_local_input_cached(
    params,
    S_list,
    target_qubit=0,
    local_label="0",
    spectator_label="0",
    include_readout=True
):
    psi0 = prepare_local_tomography_state(
        target_qubit=target_qubit,
        local_label=local_label,
        spectator_label=spectator_label
    )

    rho0 = ket_to_rho(psi0)
    rho_out = evolve_rho_with_superoperator(rho0, S_list[target_qubit])

    r_out = []

    for basis_label in measurement_labels:
        if include_readout:
            m = exact_measured_expectation_with_readout(
                rho=rho_out,
                params=params,
                qubit=target_qubit,
                basis_label=basis_label
            )
        else:
            _, _, m = ideal_measurement_probabilities(
                rho=rho_out,
                qubit=target_qubit,
                basis_label=basis_label
            )

        r_out.append(m)

    return np.array(r_out, dtype=np.float64)


def fit_affine_bloch_map(r_in_list, r_out_list):
    """
    Fit r_out = A r_in + b.
    """
    R_in = np.asarray(r_in_list, dtype=np.float64)
    R_out = np.asarray(r_out_list, dtype=np.float64)

    X_design = np.hstack([R_in, np.ones((R_in.shape[0], 1))])
    W, *_ = np.linalg.lstsq(X_design, R_out, rcond=None)

    A = W[:3, :].T
    b = W[3, :]

    return A, b


def reference_local_affine_channel_cached(
    params,
    S_list,
    target_qubit=0,
    spectator_label="0",
    include_readout=True
):
    r_in_list = []
    r_out_list = []

    for local_label in tomography_input_labels:
        r_in = expected_bloch[local_label]

        r_out = exact_output_bloch_for_local_input_cached(
            params=params,
            S_list=S_list,
            target_qubit=target_qubit,
            local_label=local_label,
            spectator_label=spectator_label,
            include_readout=include_readout
        )

        r_in_list.append(r_in)
        r_out_list.append(r_out)

    A, b = fit_affine_bloch_map(r_in_list, r_out_list)
    return A, b


def flatten_local_channels(A_list, b_list):
    pieces = []

    for A, b in zip(A_list, b_list):
        pieces.append(A.reshape(-1))
        pieces.append(b.reshape(-1))

    return np.concatenate(pieces).astype(np.float32)


def unflatten_local_channels(y, N=N_QUBITS):
    y = np.asarray(y)
    A_list = []
    b_list = []

    offset = 0
    for _ in range(N):
        A = y[offset:offset+9].reshape(3, 3)
        offset += 9

        b = y[offset:offset+3]
        offset += 3

        A_list.append(A)
        b_list.append(b)

    return A_list, b_list


def reference_all_local_channels_label_cached(
    params,
    S_list,
    spectator_label="0",
    include_readout=True
):
    """
    Output dimension for 3 qubits:
        3 * 12 = 36
    """
    A_list = []
    b_list = []

    for target_qubit in range(N_QUBITS):
        A, b = reference_local_affine_channel_cached(
            params=params,
            S_list=S_list,
            target_qubit=target_qubit,
            spectator_label=spectator_label,
            include_readout=include_readout
        )

        A_list.append(A)
        b_list.append(b)

    return flatten_local_channels(A_list, b_list)

# %% Cell 13
# ============================================================
# CELL 10 — 3-qubit masked measurement protocol
# ============================================================

def full_tomo_feature_index(target_qubit, state_label, measurement_label):
    """
    target_qubit is zero-indexed.
    """
    q_block = len(tomography_input_labels) * len(measurement_labels)
    q_offset = target_qubit * q_block

    s_idx = tomography_input_labels.index(state_label)
    m_idx = measurement_labels.index(measurement_label)

    return q_offset + len(measurement_labels) * s_idx + m_idx


def decode_full_tomo_index(idx):
    idx = int(idx)

    q_block = len(tomography_input_labels) * len(measurement_labels)

    q = idx // q_block
    local_idx = idx % q_block

    s_idx = local_idx // len(measurement_labels)
    m_idx = local_idx % len(measurement_labels)

    return q, tomography_input_labels[s_idx], measurement_labels[m_idx]


# Fixed partial protocol
fixed_partial_state_labels = ["0", "+", "+i"]
fixed_partial_measurement_labels = ["X", "Z"]

fixed_partial_indices = []

for q in range(N_QUBITS):
    for state_label in fixed_partial_state_labels:
        for meas_label in fixed_partial_measurement_labels:
            fixed_partial_indices.append(
                full_tomo_feature_index(q, state_label, meas_label)
            )

fixed_partial_indices = np.array(fixed_partial_indices, dtype=int)


def fixed_partial_mask_indices():
    return fixed_partial_indices.copy()


def random_mask_indices(K, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    return np.sort(rng.choice(np.arange(FULL_TOMO_DIM), size=K, replace=False))


def complete_mask_indices():
    return np.arange(FULL_TOMO_DIM, dtype=int)


def make_masked_tomography_input(x_full_tomo, observed_indices, shots):
    """
    Masked representation:
        [values_with_missing_zero, mask_bits, log10(shots)]

    Dimension for 3 qubits:
        54 + 54 + 1 = 109
    """
    x_full_tomo = np.asarray(x_full_tomo, dtype=np.float32)
    observed_indices = np.asarray(observed_indices, dtype=int)

    values = np.zeros(FULL_TOMO_DIM, dtype=np.float32)
    mask = np.zeros(FULL_TOMO_DIM, dtype=np.float32)

    values[observed_indices] = x_full_tomo[observed_indices]
    mask[observed_indices] = 1.0

    shot_feature = np.array([np.log10(shots)], dtype=np.float32)

    return np.concatenate([values, mask, shot_feature]).astype(np.float32)


print("FULL_TOMO_DIM:", FULL_TOMO_DIM)
print("LABEL_DIM:", LABEL_DIM)
print("Masked input dimension:", 2 * FULL_TOMO_DIM + 1)

print("\nFixed partial indices:", fixed_partial_indices)
print("Number fixed partial measurements:", len(fixed_partial_indices))

print("\nDecoded fixed partial settings:")
for idx in fixed_partial_indices:
    print(idx, "->", decode_full_tomo_index(idx))

# %% Cell 14
# ============================================================
# CELL 13 — Standardization helpers
# ============================================================
# This cell must only define helper functions. Dataset-specific
# standardization is done later in the FINAL SCALED RUN cells,
# after X_train/Y_train have actually been generated.

def compute_standardizer(X, eps=1e-8):
    X = np.asarray(X)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return mean.astype(np.float32), std.astype(np.float32)

def apply_standardizer(X, mean, std):
    return ((np.asarray(X) - mean) / std).astype(np.float32)

def invert_standardizer(Xs, mean, std):
    return (np.asarray(Xs) * std + mean).astype(np.float32)

print("Standardization helpers are defined.")


def mse_np(a, b):
    """Mean squared error for numpy arrays."""
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def per_example_l2_error(Y_est, Y_true):
    """Per-example Euclidean error along the output-label dimension."""
    return np.linalg.norm(np.asarray(Y_est) - np.asarray(Y_true), axis=1)


def improvement_ratio(noisy_error, corrected_error, eps=1e-12):
    """Improvement ratio used in QAOA tables."""
    return float(noisy_error / (corrected_error + eps))

# %% Cell 15
# ============================================================
# CELL 14 — Direct masked-tomography baseline helpers
# ============================================================
# This cell only defines the direct baseline. It does not execute
# it here because the final train/val/test datasets are generated
# later in the FINAL SCALED RUN block.

def direct_masked_channel_from_input_3q(
    x_masked,
    mean_channel,
    ridge=1e-6
):
    """
    Direct reconstruction from masked tomography input.

    Starts from mean_channel and updates only measured components.

    x_masked:
        [values, mask, log10(shots)]
        shape = (109,)

    mean_channel:
        shape = (36,)
    """
    values = np.asarray(x_masked[:FULL_TOMO_DIM], dtype=np.float64)
    mask = np.asarray(x_masked[FULL_TOMO_DIM:2*FULL_TOMO_DIM], dtype=np.float64)

    A_mean_list, b_mean_list = unflatten_local_channels(mean_channel, N=N_QUBITS)

    A_list = [A.copy() for A in A_mean_list]
    b_list = [b.copy() for b in b_mean_list]

    rows_by_q_comp = {}
    vals_by_q_comp = {}

    for q in range(N_QUBITS):
        for comp in range(3):
            rows_by_q_comp[(q, comp)] = []
            vals_by_q_comp[(q, comp)] = []

    observed_indices = np.where(mask > 0.5)[0]

    for idx in observed_indices:
        q, state_label, meas_label = decode_full_tomo_index(idx)
        comp = measurement_labels.index(meas_label)
        r_in = expected_bloch[state_label]
        design_row = np.concatenate([r_in, [1.0]])
        m_val = values[idx]

        rows_by_q_comp[(q, comp)].append(design_row)
        vals_by_q_comp[(q, comp)].append(m_val)

    for q in range(N_QUBITS):
        for comp in range(3):
            rows = rows_by_q_comp[(q, comp)]
            vals = vals_by_q_comp[(q, comp)]

            if len(rows) == 0:
                continue

            Xd = np.stack(rows)
            yd = np.array(vals)

            XtX = Xd.T @ Xd
            Xty = Xd.T @ yd

            w = np.linalg.solve(XtX + ridge * np.eye(4), Xty)

            A_list[q][comp, :] = w[:3]
            b_list[q][comp] = w[3]

    return flatten_local_channels(A_list, b_list)


def direct_masked_channels_for_dataset_3q(X_masked, mean_channel):
    Y_direct = []
    for i in range(len(X_masked)):
        y_direct = direct_masked_channel_from_input_3q(
            x_masked=X_masked[i],
            mean_channel=mean_channel
        )
        Y_direct.append(y_direct)
    return np.stack(Y_direct).astype(np.float32)

print("Direct masked-tomography baseline helpers are defined.")

# %% Cell 16
# ============================================================
# CELL 16 — Optional ChannelNet class for 3-qubit masked tomography
# ============================================================
# The final scaled run defines and uses FinalChannelNet3Q later.
# This lightweight class is kept only for compatibility with earlier cells.

class ChannelNet3Q(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

print("ChannelNet3Q class is defined. The final run uses FinalChannelNet3Q later.")

# %% Cell 17
# ============================================================
# CELL 17 — Physicality helper functions for local affine channels
# ============================================================
# This cell defines reusable helpers only. It no longer depends on
# Y_mean_3q/Y_std_3q, because those are created later after data generation.

bloch_test_np = np.array([
    [ 1,  0,  0],
    [-1,  0,  0],
    [ 0,  1,  0],
    [ 0, -1,  0],
    [ 0,  0,  1],
    [ 0,  0, -1],
    [ 1,  1,  1],
    [ 1, -1,  1],
    [-1,  1,  1],
    [ 1,  1, -1],
], dtype=np.float32)

for i in range(6, len(bloch_test_np)):
    bloch_test_np[i] /= np.linalg.norm(bloch_test_np[i])

bloch_test_t = torch.tensor(bloch_test_np, dtype=torch.float32).to(device)


def split_channels_torch_3q(y_phys):
    """
    y_phys shape: (batch, 36)
    Return:
        A shape: (batch, N_QUBITS, 3, 3)
        b shape: (batch, N_QUBITS, 3)
    """
    batch = y_phys.shape[0]
    A_all = []
    b_all = []
    offset = 0
    for _ in range(N_QUBITS):
        A = y_phys[:, offset:offset+9].reshape(batch, 3, 3)
        offset += 9
        b = y_phys[:, offset:offset+3]
        offset += 3
        A_all.append(A)
        b_all.append(b)
    return torch.stack(A_all, dim=1), torch.stack(b_all, dim=1)


def physicality_loss_3q_from_standardized(y_pred_s, Y_mean_t, Y_std_t):
    y_phys = y_pred_s * Y_std_t + Y_mean_t
    A, b = split_channels_torch_3q(y_phys)
    r = bloch_test_t[None, None, :, :]
    penalties = []
    for q in range(N_QUBITS):
        A_q = A[:, q, :, :]
        b_q = b[:, q, :]
        out = torch.matmul(r, A_q.transpose(1, 2)[:, None, :, :])
        out = out.squeeze(1) + b_q[:, None, :]
        norms = torch.linalg.norm(out, dim=-1)
        penalties.append((torch.relu(norms - 1.0) ** 2).mean())
    return sum(penalties) / len(penalties)

print("Physicality helper functions are defined.")

# %% Cell 18
# ============================================================
# CELL 23 — 3-qubit QAOA chain operators
# ============================================================

# QAOA now lives in the computational qubit Hilbert space: 2^3 = 8
N_QAOA = 3
DIM_QAOA = 2 ** N_QAOA

I2_q = np.eye(2, dtype=complex)
X_q = np.array([[0, 1], [1, 0]], dtype=complex)
Y_q = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_q = np.array([[1, 0], [0, -1]], dtype=complex)

I_qaoa = np.eye(DIM_QAOA, dtype=complex)

def kron_qubit_ops(ops):
    out = np.array([[1.0 + 0.0j]])
    for op in ops:
        out = np.kron(out, op)
    return out

def embed_qubit_operator(P, q, N=N_QAOA):
    """
    Embed a one-qubit operator P on qubit q.
    q is zero-indexed.
    """
    ops = []
    for j in range(N):
        ops.append(P if j == q else I2_q)
    return kron_qubit_ops(ops)

X_ops_qaoa = [embed_qubit_operator(X_q, q) for q in range(N_QAOA)]
Y_ops_qaoa = [embed_qubit_operator(Y_q, q) for q in range(N_QAOA)]
Z_ops_qaoa = [embed_qubit_operator(Z_q, q) for q in range(N_QAOA)]

# 3-qubit chain graph: 1-2-3
qaoa_edges_chain = [(0, 1), (1, 2)]

def build_maxcut_cost_hamiltonian(edges, N=N_QAOA):
    """
    H_C = sum_edges 1/2 (I - Zi Zj)
    """
    H = np.zeros((2**N, 2**N), dtype=complex)

    for (i, j) in edges:
        H += 0.5 * (np.eye(2**N, dtype=complex) - Z_ops_qaoa[i] @ Z_ops_qaoa[j])

    return H

H_cost_chain = build_maxcut_cost_hamiltonian(qaoa_edges_chain)
H_mixer_chain = sum(X_ops_qaoa)

# Initial |+++>
ket0_q = basis(2, 0)
ket1_q = basis(2, 1)
ket_plus_q = normalize_ket((ket0_q + ket1_q) / np.sqrt(2))
ket_plus_all = kron_qubit_ops([ket_plus_q for _ in range(N_QAOA)])

def qaoa_state_p1(gamma, beta):
    """
    p=1 QAOA state:
        exp(-i beta H_M) exp(-i gamma H_C) |+++>
    """
    U_cost = expm(-1j * gamma * H_cost_chain)
    U_mixer = expm(-1j * beta * H_mixer_chain)

    psi = U_mixer @ U_cost @ ket_plus_all
    return normalize_ket(psi)

def ideal_qaoa_cost_p1(gamma, beta):
    psi = qaoa_state_p1(gamma, beta)
    val = dagger(psi) @ H_cost_chain @ psi
    return float(np.real_if_close(val[0, 0]))

# Bitstring utilities
qaoa_bitstrings = list(product([0, 1], repeat=N_QAOA))
qaoa_bitstring_labels = ["".join(map(str, b)) for b in qaoa_bitstrings]

def qubit_product_ket(bits):
    kets = [basis(2, b) for b in bits]
    return kron_qubit_ops(kets)

qaoa_basis_kets = {
    "".join(map(str, bits)): qubit_product_ket(bits)
    for bits in qaoa_bitstrings
}

def maxcut_value_for_bitstring(bits, edges=qaoa_edges_chain):
    """
    Classical MaxCut value for a bitstring.
    For chain 1-2-3, maximum is 2.
    """
    value = 0
    for (i, j) in edges:
        if bits[i] != bits[j]:
            value += 1
    return value

bitstring_costs = {
    "".join(map(str, bits)): maxcut_value_for_bitstring(bits)
    for bits in qaoa_bitstrings
}

C_max_chain = max(bitstring_costs.values())
optimal_bitstrings_chain = [
    label for label, c in bitstring_costs.items()
    if c == C_max_chain
]

print("QAOA bitstring costs:")
for label in qaoa_bitstring_labels:
    print(label, "cost =", bitstring_costs[label])

print("C_max_chain:", C_max_chain)
print("Optimal bitstrings:", optimal_bitstrings_chain)

# %% Cell 19
# ============================================================
# CELL 24 — Ideal QAOA landscape and P_opt grid
# ============================================================
# Build the ideal 3-qubit QAOA grid needed by all later evaluations.

# Use moderate grid first. Later increase to 41 or 61 if desired.
n_gamma_3q = 31
n_beta_3q = 31

gamma_grid_3q = np.linspace(0, np.pi, n_gamma_3q)
beta_grid_3q = np.linspace(0, np.pi / 2, n_beta_3q)

rho_ideal_grid_3q = [[None for _ in range(n_beta_3q)] for _ in range(n_gamma_3q)]
C_ideal_3q = np.zeros((n_gamma_3q, n_beta_3q), dtype=np.float64)
Popt_ideal_3q = np.zeros((n_gamma_3q, n_beta_3q), dtype=np.float64)

def probs_from_rho_qaoa(rho):
    probs = {}
    for label, ket in qaoa_basis_kets.items():
        P = ket @ dagger(ket)
        p = float(np.real_if_close(np.trace(rho @ P)))
        probs[label] = p

    arr = np.array([probs[k] for k in qaoa_bitstring_labels], dtype=np.float64)
    arr = np.clip(arr, 0.0, 1.0)

    if arr.sum() <= 1e-12:
        arr = np.ones_like(arr) / len(arr)
    else:
        arr = arr / arr.sum()

    return {k: float(v) for k, v in zip(qaoa_bitstring_labels, arr)}


def maxcut_cost_from_probs(probs):
    return float(sum(bitstring_costs[label] * probs[label] for label in qaoa_bitstring_labels))


def p_opt_from_probs(probs):
    return float(sum(probs[label] for label in optimal_bitstrings_chain))

for i, gamma in enumerate(gamma_grid_3q):
    for j, beta in enumerate(beta_grid_3q):
        psi = qaoa_state_p1(gamma, beta)
        rho = ket_to_rho(psi)
        rho_ideal_grid_3q[i][j] = rho
        probs = probs_from_rho_qaoa(rho)
        C_ideal_3q[i, j] = maxcut_cost_from_probs(probs)
        Popt_ideal_3q[i, j] = p_opt_from_probs(probs)

print("Ideal QAOA chain max cost on grid:", C_ideal_3q.max())
print("Ideal QAOA chain max P_opt on grid:", Popt_ideal_3q.max())

# %% Cell 20
# ============================================================
# CELL 25 — Apply 3 local affine channels to 3-qubit QAOA states
# ============================================================

paulis_1q = [I2_q, X_q, Y_q, Z_q]
pauli_labels_1q = ["I", "X", "Y", "Z"]

pauli_strings_3q = []
pauli_string_labels_3q = []

for labels in product(range(4), repeat=N_QAOA):
    ops = [paulis_1q[k] for k in labels]
    labs = [pauli_labels_1q[k] for k in labels]

    pauli_strings_3q.append(kron_qubit_ops(ops))
    pauli_string_labels_3q.append("".join(labs))

def rho_to_pauli_coeffs_3q(rho):
    """
    rho = 1/2^N sum_P c_P P
    c_P = Tr(rho P)
    """
    coeffs = []
    for P in pauli_strings_3q:
        coeffs.append(float(np.real_if_close(np.trace(rho @ P))))
    return np.array(coeffs, dtype=np.float64)

def pauli_coeffs_to_rho_3q(coeffs):
    rho = np.zeros((DIM_QAOA, DIM_QAOA), dtype=complex)

    for c, P in zip(coeffs, pauli_strings_3q):
        rho += c * P

    rho = rho / (2 ** N_QAOA)
    return clean_density_matrix(rho)

def affine_channel_to_ptm(A, b):
    """
    Convert r_out = A r_in + b to 4x4 Pauli transfer matrix
    acting on [1, rx, ry, rz]^T.
    """
    R = np.zeros((4, 4), dtype=np.float64)

    R[0, 0] = 1.0
    R[1:4, 0] = b
    R[1:4, 1:4] = A

    return R

def product_ptm_from_channel_label_3q(y_channel):
    """
    y_channel shape: (36,)
    contains [A1,b1,A2,b2,A3,b3]
    """
    A_list, b_list = unflatten_local_channels(y_channel, N=N_QUBITS)

    R_list = [
        affine_channel_to_ptm(A_list[q], b_list[q])
        for q in range(N_QAOA)
    ]

    R_prod = R_list[0]
    for q in range(1, N_QAOA):
        R_prod = np.kron(R_prod, R_list[q])

    return R_prod

def apply_product_local_channel_3q(rho, y_channel):
    """
    Apply product local channel to a 3-qubit density matrix.
    """
    R_prod = product_ptm_from_channel_label_3q(y_channel)

    c_in = rho_to_pauli_coeffs_3q(rho)
    c_out = R_prod @ c_in

    rho_out = pauli_coeffs_to_rho_3q(c_out)

    return rho_out

def qaoa_cost_landscape_from_channel_3q(
    y_channel,
    rho_ideal_grid,
    shots=None,
    seed=0
):
    """
    Compute noisy QAOA landscape generated by product local channels.

    If shots is None:
        exact expectation from probabilities.

    If shots is not None:
        sampled finite-shot estimate of MaxCut cost.
    """
    rng = np.random.default_rng(seed)

    C = np.zeros((n_gamma_3q, n_beta_3q), dtype=np.float64)
    Popt = np.zeros((n_gamma_3q, n_beta_3q), dtype=np.float64)

    for i in range(n_gamma_3q):
        for j in range(n_beta_3q):
            rho_ideal = rho_ideal_grid[i][j]

            rho_noisy = apply_product_local_channel_3q(
                rho_ideal,
                y_channel
            )

            probs = probs_from_rho_qaoa(rho_noisy)

            if shots is None:
                C[i, j] = maxcut_cost_from_probs(probs)
            else:
                p_arr = np.array([probs[k] for k in qaoa_bitstring_labels], dtype=np.float64)
                p_arr = p_arr / p_arr.sum()

                counts = rng.multinomial(shots, p_arr)

                cost_hat = 0.0
                for count, label in zip(counts, qaoa_bitstring_labels):
                    cost_hat += count * bitstring_costs[label]

                C[i, j] = cost_hat / shots

            Popt[i, j] = p_opt_from_probs(probs)

    return C, Popt

def mitigate_landscape_by_bias_correction(C_measured_noisy, C_ideal, C_model_noisy, clip=True):
    """
    C_mit = C_measured_noisy - (C_model_noisy - C_ideal)
    """
    bias_hat = C_model_noisy - C_ideal
    C_mit = C_measured_noisy - bias_hat

    if clip:
        C_mit = np.clip(C_mit, 0.0, C_max_chain)

    return C_mit, bias_hat

# %% Cell 21
# ============================================================
# CELL 26 — QAOA reliability metrics
# ============================================================

def landscape_error_metrics_3q(C_est, C_ideal):
    abs_err = np.abs(C_est - C_ideal)

    return {
        "MAE": float(abs_err.mean()),
        "MedianAE": float(np.median(abs_err)),
        "MaxAE": float(abs_err.max()),
        "RMSE": float(np.sqrt(np.mean(abs_err**2))),
    }

def best_grid_point(C_grid):
    idx = np.unravel_index(np.argmax(C_grid), C_grid.shape)

    return {
        "idx": idx,
        "gamma": float(gamma_grid_3q[idx[0]]),
        "beta": float(beta_grid_3q[idx[1]]),
        "cost": float(C_grid[idx])
    }

def selected_point_metrics(C_select_grid, C_ideal_grid, Popt_ideal_grid):
    """
    Select best point according to C_select_grid,
    evaluate it on ideal landscape.
    """
    best = best_grid_point(C_select_grid)
    idx = best["idx"]

    ideal_selected_cost = float(C_ideal_grid[idx])
    ideal_best_cost = float(C_ideal_grid.max())

    regret = ideal_best_cost - ideal_selected_cost
    approximation_ratio = ideal_selected_cost / (C_max_chain + 1e-12)

    p_opt_selected = float(Popt_ideal_grid[idx])

    ideal_best = best_grid_point(C_ideal_grid)
    gamma_displacement = best["gamma"] - ideal_best["gamma"]
    beta_displacement = best["beta"] - ideal_best["beta"]

    parameter_displacement = float(
        np.sqrt(gamma_displacement**2 + beta_displacement**2)
    )

    return {
        "selected_gamma": best["gamma"],
        "selected_beta": best["beta"],
        "selected_est_cost": best["cost"],
        "ideal_selected_cost": ideal_selected_cost,
        "ideal_best_cost": ideal_best_cost,
        "regret": regret,
        "approximation_ratio": approximation_ratio,
        "p_opt_selected": p_opt_selected,
        "parameter_displacement": parameter_displacement
    }

def improvement_ratio(noisy_error, mitigated_error, eps=1e-12):
    return float(noisy_error / (mitigated_error + eps))

# %% Cell 22
# ============================================================
# CELL 28 — Multi-device 3-qubit QAOA evaluation
# ============================================================
"""
prediction_methods_3q = {
    "Direct": Y_test_direct_3q,
    "Ridge": Y_test_ridge_3q,
    "RandomForest": Y_test_rf_3q,
    "ChannelNet": Y_test_pred_3q,
}"""

def evaluate_qaoa_device_3q(
    idx_device,
    Y_true_all,
    prediction_methods,
    qaoa_shots=2048,
    seed=0
):

    """
    Evaluate one hidden device.

    Returns:
        dictionary with noisy metrics and mitigated metrics for each method.
    """
    y_true = Y_true_all[idx_device]

    C_noisy_sampled, _ = qaoa_cost_landscape_from_channel_3q(
        y_channel=y_true,
        rho_ideal_grid=rho_ideal_grid_3q,
        shots=qaoa_shots,
        seed=seed + idx_device
    )

    results = {}

    # Noisy baseline
    results["Noisy"] = {
        "C": C_noisy_sampled,
        "error": landscape_error_metrics_3q(C_noisy_sampled, C_ideal_3q),
        "selected": selected_point_metrics(C_noisy_sampled, C_ideal_3q, Popt_ideal_3q)
    }

    # Mitigated methods
    for method_name, Y_pred_all in prediction_methods.items():
        y_model = Y_pred_all[idx_device]

        C_model, _ = qaoa_cost_landscape_from_channel_3q(
            y_channel=y_model,
            rho_ideal_grid=rho_ideal_grid_3q,
            shots=None,
            seed=seed + 1000 + idx_device
        )

        C_mit, _ = mitigate_landscape_by_bias_correction(
            C_noisy_sampled,
            C_ideal_3q,
            C_model,
            clip=True
        )

        results[method_name] = {
            "C": C_mit,
            "error": landscape_error_metrics_3q(C_mit, C_ideal_3q),
            "selected": selected_point_metrics(C_mit, C_ideal_3q, Popt_ideal_3q)
        }

    return results

"""
n_eval_devices_3q = min(20, len(Y_test_3q))

results_qaoa_3q = []

for k in range(n_eval_devices_3q):
    res = evaluate_qaoa_device_3q(
        idx_device=k,
        Y_true_all=Y_test_3q,
        prediction_methods=prediction_methods_3q,
        qaoa_shots=qaoa_shots_3q,
        seed=20000
    )

    results_qaoa_3q.append(res)

    msg = f"Device {k:02d} | Noisy MAE {res['Noisy']['error']['MAE']:.4f}"

    for method_name in prediction_methods_3q.keys():
        msg += f" | {method_name} {res[method_name]['error']['MAE']:.4f}"

    print(msg)
  """

# %% Cell 23
# ============================================================
# CELL 29 — Legacy summary plotting cell disabled
# ============================================================
# The original version of this notebook tried to summarize variables
# such as prediction_methods_3q and results_qaoa_3q before they existed.
# The final scaled run below now creates and saves all summary tables.
print("Legacy summary cell skipped. Final summaries are generated in the FINAL SCALED RUN cells.")

# %% Cell 24
# Global QAOA finite-shot setting used by final evaluations
qaoa_shots_3q = 2048
print("qaoa_shots_3q =", qaoa_shots_3q)

# %% [markdown]
# ### PRA Step 1: leakage-explicit observables
#
# This block adds explicit leakage observables to the three-qutrit workflow. The logical Pauli readout remains binary, but the dataset now also contains per-transmon leakage labels, \(p_{\mathrm{leak},i}=\mathrm{Tr}[
# ho |2_i
# angle\langle 2_i|]\), summarized by mean and maximum leakage over the local tomography probes. Run this block after the simulator-definition cells.

# %% Cell 26
# ============================================================
# PRA STEP 1 — Leakage-explicit observables and labels
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import os
import pickle

# Use the already selected device if available, otherwise define it.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------------------------------
# Required simulator functions check
# ------------------------------------------------------------

required_for_pra_step1 = [
    "sample_hidden_device_3q",
    "build_all_pulse_superoperators",
    "generate_full_tomography_vector_cached",
    "reference_all_local_channels_label_cached",
    "prepare_local_tomography_state",
    "ket_to_rho",
    "evolve_rho_with_superoperator",
    "make_masked_tomography_input",
    "FULL_TOMO_DIM",
    "LABEL_DIM",
    "N_QUBITS",
    "HILBERT_DIM",
    "basis",
    "dagger",
    "embed_single_qutrit_operator",
    "P_comp",
    "qutrit_product_ket",
    "tomography_input_labels",
]

missing = [name for name in required_for_pra_step1 if name not in globals()]

if missing:
    raise RuntimeError(
        "Missing simulator functions/variables for PRA Step 1: "
        + str(missing)
        + "\nRun the simulator-definition cells first."
    )

print("PRA Step 1 simulator requirements are available.")

# ------------------------------------------------------------
# Leakage projectors
# ------------------------------------------------------------
# The old helper leakage_probability(rho) measured total population outside
# the full computational subspace.  For a leakage-explicit journal version,
# we also need per-transmon leakage observables:
#     p_leak,i = Tr[rho |2_i><2_i|].
# These are genuine qutrit-level leakage observables, while the existing
# Pauli readout can remain binary for the logical X/Y/Z measurements.

ket2_single = basis(3, 2)
P2_single = ket2_single @ dagger(ket2_single)
P_leak_ops = [embed_single_qutrit_operator(P2_single, q) for q in range(N_QUBITS)]


def leakage_probability_any(rho):
    """
    Total probability outside the full computational subspace.
    This is one minus the population in {|0>,|1>}^N.
    """
    p_comp = float(np.real_if_close(np.trace(P_comp @ rho)))
    return float(np.clip(1.0 - p_comp, 0.0, 1.0))


def leakage_probability_per_qubit(rho):
    """
    Per-transmon leakage probabilities.

    Returns an array of shape (N_QUBITS,), where component q is
        p_leak,q = Tr[rho |2_q><2_q|].
    For states with simultaneous multi-qubit leakage, the sum of these
    components can exceed the total outside-computational-subspace probability,
    because it counts leakage per transmon.
    """
    vals = []
    for q in range(N_QUBITS):
        p2_q = float(np.real_if_close(np.trace(P_leak_ops[q] @ rho)))
        vals.append(np.clip(p2_q, 0.0, 1.0))
    return np.array(vals, dtype=np.float32)


# Keep the original name for backwards compatibility in old stress-test code.
# It still means "any population outside the computational subspace".
leakage_probability = leakage_probability_any


def leakage_labels_per_qubit_cached(params, S_list):
    """
    Compute leakage-observable labels for each transmon.

    For each target transmon q_target, we run the six local tomography input
    states, apply the q_target pulse superoperator, and measure the per-transmon
    leakage observables
        p_leak,k = Tr[rho_out |2_k><2_k|],  k = 0,1,2.

    Output shape: (6,)
        [mean p_leak,0, mean p_leak,1, mean p_leak,2,
         max  p_leak,0, max  p_leak,1, max  p_leak,2]

    These labels make leakage explicit without changing the existing binary
    logical Pauli readout model.
    """
    leak_rows = []

    for q_target in range(N_QUBITS):
        for state_label in tomography_input_labels:
            psi0 = prepare_local_tomography_state(
                target_qubit=q_target,
                local_label=state_label,
                spectator_label="0"
            )

            rho0 = ket_to_rho(psi0)
            rho_out = evolve_rho_with_superoperator(rho0, S_list[q_target])

            leak_rows.append(leakage_probability_per_qubit(rho_out))

    leak_rows = np.stack(leak_rows).astype(np.float32)  # shape (N_QUBITS*6, N_QUBITS)

    leakage_mean = leak_rows.mean(axis=0)
    leakage_max = leak_rows.max(axis=0)

    return np.concatenate([leakage_mean, leakage_max]).astype(np.float32)


def augment_channel_with_leakage(y_channel, leakage_label):
    """
    y_channel shape: (36,)
    leakage_label shape: (6,)
    output shape: (42,)
    """
    return np.concatenate([
        np.asarray(y_channel, dtype=np.float32),
        np.asarray(leakage_label, dtype=np.float32)
    ]).astype(np.float32)


def split_augmented_channel_leakage(y_aug):
    """
    Split augmented target/prediction.

    Returns:
        y_channel: shape (...,36)
        y_leak: shape (...,6)
    """
    y_aug = np.asarray(y_aug)
    return y_aug[..., :LABEL_DIM], y_aug[..., LABEL_DIM:LABEL_DIM+6]


# Quick projector sanity check on the maximally leaked state |222>.
psi_222 = qutrit_product_ket([2] * N_QUBITS)
rho_222 = ket_to_rho(psi_222)
print("Sanity check, any leakage for |222>:", leakage_probability_any(rho_222))
print("Sanity check, per-qubit leakage for |222>:", leakage_probability_per_qubit(rho_222))

# %% Cell 27
# ============================================================
# PRA STEP 1B — Leakage-enriched device sampler
# ============================================================

def perturb_for_leakage_learning_3q(params, rng=None):
    """
    Create a richer leakage-aware training distribution.

    This does NOT replace the full stress tests.
    It only enriches the training data with a mixture of:
        nominal devices,
        lower anharmonicity,
        stronger pulse-amplitude disorder,
        worse decoherence,
        correlated dephasing.

    This gives the learner meaningful leakage variation.
    """
    if rng is None:
        rng = np.random.default_rng()

    p = copy.deepcopy(params) if "copy" in globals() else __import__("copy").deepcopy(params)

    family = rng.choice([
        "nominal",
        "mild_leakage",
        "strong_leakage",
        "decoherence",
        "correlated_dephasing",
    ], p=[0.25, 0.25, 0.20, 0.15, 0.15])

    if family == "nominal":
        pass

    elif family == "mild_leakage":
        # reduce anharmonicity mildly
        alpha_scale = rng.uniform(0.78, 0.92)
        p["alpha"] = p["alpha"] * alpha_scale
        p["amp_scale"] = p["amp_scale"] + rng.normal(0.0, 0.035, size=p["N"])

    elif family == "strong_leakage":
        # stronger leakage stress, but keep stable
        alpha_scale = rng.uniform(0.55, 0.75)
        p["alpha"] = p["alpha"] * alpha_scale
        p["amp_scale"] = p["amp_scale"] + rng.normal(0.0, 0.060, size=p["N"])

    elif family == "decoherence":
        factor = rng.uniform(1.5, 3.0)
        p["T1"] = p["T1"] / factor
        p["Tphi"] = p["Tphi"] / factor

    elif family == "correlated_dephasing":
        p["correlated_dephasing_rate"] = rng.uniform(0.5e-5, 6.0e-5)

    p["noise_family"] = "leakage_learning_" + family

    return p


def generate_one_leakage_augmented_example_3q(
    rng,
    K=18,
    mask_mode="random",
    topology="chain",
    shot_choices=np.array([128, 512, 1024, 4096], dtype=int),
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    leakage_enriched=True
):
    """
    Generate one supervised example with augmented target:
        X_masked: partial tomography input
        Y_aug: [channel_label(36), leakage_label(6)]
    """
    shots = int(rng.choice(shot_choices))

    params = sample_hidden_device_3q(rng=rng, topology=topology)

    if leakage_enriched:
        params = perturb_for_leakage_learning_3q(params, rng=rng)

    S_list = build_all_pulse_superoperators(
        params=params,
        theta=pulse_theta,
        phase=pulse_phase,
        T=pulse_T,
        include_decoherence=True
    )

    x_full = generate_full_tomography_vector_cached(
        params=params,
        S_list=S_list,
        shots=shots,
        rng=rng
    )

    y_channel = reference_all_local_channels_label_cached(
        params=params,
        S_list=S_list,
        include_readout=True
    )

    leakage_label = leakage_labels_per_qubit_cached(
        params=params,
        S_list=S_list
    )

    y_aug = augment_channel_with_leakage(
        y_channel=y_channel,
        leakage_label=leakage_label
    )

    if mask_mode == "random":
        observed_indices = random_mask_indices(K=K, rng=rng)

    elif mask_mode == "fixed":
        observed_indices = fixed_partial_mask_indices_step9() if "fixed_partial_mask_indices_step9" in globals() else fixed_partial_indices

    elif mask_mode == "complete":
        observed_indices = np.arange(FULL_TOMO_DIM, dtype=int)

    else:
        raise ValueError("mask_mode must be random, fixed, or complete.")

    x_masked = make_masked_tomography_input(
        x_full_tomo=x_full,
        observed_indices=observed_indices,
        shots=shots
    )

    return (
        x_masked.astype(np.float32),
        y_aug.astype(np.float32),
        y_channel.astype(np.float32),
        leakage_label.astype(np.float32),
        x_full.astype(np.float32),
        params,
        shots,
        observed_indices.astype(int)
    )


def build_leakage_augmented_dataset_3q(
    n_examples,
    seed=0,
    K=18,
    mask_mode="random",
    topology="chain",
    leakage_enriched=True,
    print_every=5
):
    """
    Build leakage-aware dataset.

    X shape:       (N,109)
    Y_aug shape:   (N,42)
    Y_channel:     (N,36)
    Y_leakage:     (N,6)
    """
    rng = np.random.default_rng(seed)

    X = []
    Y_aug = []
    Y_channel = []
    Y_leakage = []
    X_full = []
    params_list = []
    shots_list = []
    masks_list = []

    t0 = time.time()

    for i in range(n_examples):
        x, y_aug, y_ch, y_leak, x_full, params, shots, obs = generate_one_leakage_augmented_example_3q(
            rng=rng,
            K=K,
            mask_mode=mask_mode,
            topology=topology,
            leakage_enriched=leakage_enriched
        )

        X.append(x)
        Y_aug.append(y_aug)
        Y_channel.append(y_ch)
        Y_leakage.append(y_leak)
        X_full.append(x_full)
        params_list.append(params)
        shots_list.append(shots)
        masks_list.append(obs)

        if (i + 1) % print_every == 0 or (i + 1) == n_examples:
            print(
                f"Leakage dataset {i+1:4d}/{n_examples} "
                f"| elapsed {time.time()-t0:.1f} s"
            )

    return {
        "X": np.stack(X).astype(np.float32),
        "Y_aug": np.stack(Y_aug).astype(np.float32),
        "Y_channel": np.stack(Y_channel).astype(np.float32),
        "Y_leakage": np.stack(Y_leakage).astype(np.float32),
        "X_full": np.stack(X_full).astype(np.float32),
        "params": params_list,
        "shots": np.array(shots_list, dtype=int),
        "masks": masks_list,
    }

# %% Cell 28
# ============================================================
# PRA STEP 1C — Build leakage-aware train/val/test datasets
# ============================================================

K_LEAK_3Q = 18

# Start moderate. If runtime is okay, increase to 160/40/40.
N_TRAIN_LEAK_3Q = 80
N_VAL_LEAK_3Q = 20
N_TEST_LEAK_3Q = 20

leak_train_3q = build_leakage_augmented_dataset_3q(
    n_examples=N_TRAIN_LEAK_3Q,
    seed=51000,
    K=K_LEAK_3Q,
    mask_mode="random",
    leakage_enriched=True,
    print_every=5
)

leak_val_3q = build_leakage_augmented_dataset_3q(
    n_examples=N_VAL_LEAK_3Q,
    seed=52000,
    K=K_LEAK_3Q,
    mask_mode="random",
    leakage_enriched=True,
    print_every=5
)

leak_test_3q = build_leakage_augmented_dataset_3q(
    n_examples=N_TEST_LEAK_3Q,
    seed=53000,
    K=K_LEAK_3Q,
    mask_mode="random",
    leakage_enriched=True,
    print_every=5
)

print("Leakage-aware dataset shapes:")
print("X train:", leak_train_3q["X"].shape)
print("Y_aug train:", leak_train_3q["Y_aug"].shape)
print("Y_channel train:", leak_train_3q["Y_channel"].shape)
print("Y_leakage train:", leak_train_3q["Y_leakage"].shape)

print("\nLeakage label range:")
print("train min:", leak_train_3q["Y_leakage"].min())
print("train max:", leak_train_3q["Y_leakage"].max())
print("train mean:", leak_train_3q["Y_leakage"].mean())

plt.figure(figsize=(7, 4))
plt.hist(leak_train_3q["Y_leakage"][:, :3].flatten(), bins=20, alpha=0.7, label="Mean leakage")
plt.hist(leak_train_3q["Y_leakage"][:, 3:].flatten(), bins=20, alpha=0.7, label="Max leakage")
plt.xlabel("Leakage probability")
plt.ylabel("Count")
plt.title("Leakage-label distribution")
plt.legend()
plt.grid(True)
plt.show()

# %% Cell 29
# ============================================================
# PRA STEP 1D — Leakage-aware ChannelNet
# ============================================================

class LeakageAwareChannelNet3Q(nn.Module):
    def __init__(self, input_dim, channel_dim=36, leakage_dim=6):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.channel_head = nn.Linear(128, channel_dim)
        self.leakage_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, leakage_dim)
        )

    def forward(self, x):
        h = self.backbone(x)

        y_channel = self.channel_head(h)
        y_leakage = self.leakage_head(h)

        return torch.cat([y_channel, y_leakage], dim=-1)


class ChannelOnlyNet3Q(nn.Module):
    def __init__(self, input_dim, output_dim=36):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# %% Cell 30
# ============================================================
# PRA STEP 1E — Standardization and losses for augmented targets
# ============================================================

def compute_standardizer(X, eps=1e-8):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return mean.astype(np.float32), std.astype(np.float32)

def apply_standardizer(X, mean, std):
    return ((X - mean) / std).astype(np.float32)

def invert_standardizer(Xs, mean, std):
    return (Xs * std + mean).astype(np.float32)

X_mean_leak, X_std_leak = compute_standardizer(leak_train_3q["X"])
Y_aug_mean_leak, Y_aug_std_leak = compute_standardizer(leak_train_3q["Y_aug"])
Y_ch_mean_leak, Y_ch_std_leak = compute_standardizer(leak_train_3q["Y_channel"])

X_train_leak_s = apply_standardizer(leak_train_3q["X"], X_mean_leak, X_std_leak)
X_val_leak_s = apply_standardizer(leak_val_3q["X"], X_mean_leak, X_std_leak)
X_test_leak_s = apply_standardizer(leak_test_3q["X"], X_mean_leak, X_std_leak)

Y_aug_train_s = apply_standardizer(leak_train_3q["Y_aug"], Y_aug_mean_leak, Y_aug_std_leak)
Y_aug_val_s = apply_standardizer(leak_val_3q["Y_aug"], Y_aug_mean_leak, Y_aug_std_leak)
Y_aug_test_s = apply_standardizer(leak_test_3q["Y_aug"], Y_aug_mean_leak, Y_aug_std_leak)

Y_ch_train_s = apply_standardizer(leak_train_3q["Y_channel"], Y_ch_mean_leak, Y_ch_std_leak)
Y_ch_val_s = apply_standardizer(leak_val_3q["Y_channel"], Y_ch_mean_leak, Y_ch_std_leak)
Y_ch_test_s = apply_standardizer(leak_test_3q["Y_channel"], Y_ch_mean_leak, Y_ch_std_leak)


bloch_test_np_leak = np.array([
    [ 1,  0,  0],
    [-1,  0,  0],
    [ 0,  1,  0],
    [ 0, -1,  0],
    [ 0,  0,  1],
    [ 0,  0, -1],
    [ 1,  1,  1],
    [ 1, -1,  1],
    [-1,  1,  1],
    [ 1,  1, -1],
], dtype=np.float32)

for i in range(6, len(bloch_test_np_leak)):
    bloch_test_np_leak[i] /= np.linalg.norm(bloch_test_np_leak[i])

bloch_test_t_leak = torch.tensor(bloch_test_np_leak, dtype=torch.float32).to(device)


def split_channels_torch_from_phys(y_channel_phys):
    batch = y_channel_phys.shape[0]

    A_all = []
    b_all = []

    offset = 0

    for q in range(N_QUBITS):
        A_q = y_channel_phys[:, offset:offset+9].reshape(batch, 3, 3)
        offset += 9

        b_q = y_channel_phys[:, offset:offset+3]
        offset += 3

        A_all.append(A_q)
        b_all.append(b_q)

    A = torch.stack(A_all, dim=1)
    b = torch.stack(b_all, dim=1)

    return A, b


def physicality_loss_channel_part(y_channel_s, Y_ch_mean_t, Y_ch_std_t):
    """
    Physicality loss on the channel part only.
    """
    y_phys = y_channel_s * Y_ch_std_t + Y_ch_mean_t

    A, b = split_channels_torch_from_phys(y_phys)

    r = bloch_test_t_leak[None, None, :, :]

    penalties = []

    for q in range(N_QUBITS):
        A_q = A[:, q, :, :]
        b_q = b[:, q, :]

        out = torch.matmul(r, A_q.transpose(1, 2)[:, None, :, :])
        out = out.squeeze(1) + b_q[:, None, :]

        norms = torch.linalg.norm(out, dim=-1)
        penalty = torch.relu(norms - 1.0) ** 2

        penalties.append(penalty.mean())

    return sum(penalties) / len(penalties)


Y_aug_mean_t = torch.tensor(Y_aug_mean_leak, dtype=torch.float32).to(device)
Y_aug_std_t = torch.tensor(Y_aug_std_leak, dtype=torch.float32).to(device)

Y_ch_mean_t = torch.tensor(Y_ch_mean_leak, dtype=torch.float32).to(device)
Y_ch_std_t = torch.tensor(Y_ch_std_leak, dtype=torch.float32).to(device)

# %% Cell 31
# ============================================================
# PRA STEP 1F — Train leakage-aware model and channel-only baseline
# ============================================================

def make_loader(X_s, Y_s, batch_size=16, shuffle=True):
    return DataLoader(
        TensorDataset(
            torch.tensor(X_s, dtype=torch.float32),
            torch.tensor(Y_s, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=shuffle
    )


train_loader_aug = make_loader(X_train_leak_s, Y_aug_train_s, batch_size=16, shuffle=True)
val_loader_aug = make_loader(X_val_leak_s, Y_aug_val_s, batch_size=64, shuffle=False)
test_loader_aug = make_loader(X_test_leak_s, Y_aug_test_s, batch_size=64, shuffle=False)

train_loader_ch = make_loader(X_train_leak_s, Y_ch_train_s, batch_size=16, shuffle=True)
val_loader_ch = make_loader(X_val_leak_s, Y_ch_val_s, batch_size=64, shuffle=False)
test_loader_ch = make_loader(X_test_leak_s, Y_ch_test_s, batch_size=64, shuffle=False)


def train_leakage_aware_model_3q(
    model,
    train_loader,
    val_loader,
    n_epochs=400,
    lr=8e-4,
    weight_decay=1e-5,
    lambda_phys=0.03,
    lambda_leak=1.0
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    mse = nn.MSELoss()

    history = {
        "train_channel_mse": [],
        "train_leak_mse": [],
        "val_channel_mse": [],
        "val_leak_mse": [],
        "val_phys": [],
    }

    best_val = np.inf
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()

        tr_ch_total = 0.0
        tr_leak_total = 0.0
        n_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)

            pred_ch = pred[:, :LABEL_DIM]
            pred_leak = pred[:, LABEL_DIM:LABEL_DIM+6]

            y_ch = yb[:, :LABEL_DIM]
            y_leak = yb[:, LABEL_DIM:LABEL_DIM+6]

            ch_loss = mse(pred_ch, y_ch)
            leak_loss = mse(pred_leak, y_leak)
            phys_loss = physicality_loss_channel_part(pred_ch, Y_ch_mean_t, Y_ch_std_t)

            loss = ch_loss + lambda_leak * leak_loss + lambda_phys * phys_loss

            loss.backward()
            optimizer.step()

            bs = xb.shape[0]
            tr_ch_total += ch_loss.item() * bs
            tr_leak_total += leak_loss.item() * bs
            n_total += bs

        train_ch = tr_ch_total / n_total
        train_leak = tr_leak_total / n_total

        model.eval()

        val_ch_total = 0.0
        val_leak_total = 0.0
        val_phys_total = 0.0
        val_n = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)

                pred_ch = pred[:, :LABEL_DIM]
                pred_leak = pred[:, LABEL_DIM:LABEL_DIM+6]

                y_ch = yb[:, :LABEL_DIM]
                y_leak = yb[:, LABEL_DIM:LABEL_DIM+6]

                ch_loss = mse(pred_ch, y_ch)
                leak_loss = mse(pred_leak, y_leak)
                phys_loss = physicality_loss_channel_part(pred_ch, Y_ch_mean_t, Y_ch_std_t)

                bs = xb.shape[0]
                val_ch_total += ch_loss.item() * bs
                val_leak_total += leak_loss.item() * bs
                val_phys_total += phys_loss.item() * bs
                val_n += bs

        val_ch = val_ch_total / val_n
        val_leak = val_leak_total / val_n
        val_phys = val_phys_total / val_n

        score = val_ch + lambda_leak * val_leak

        history["train_channel_mse"].append(train_ch)
        history["train_leak_mse"].append(train_leak)
        history["val_channel_mse"].append(val_ch)
        history["val_leak_mse"].append(val_leak)
        history["val_phys"].append(val_phys)

        if score < best_val:
            best_val = score
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"val ch {val_ch:.4e} | "
                f"val leak {val_leak:.4e} | "
                f"val phys {val_phys:.4e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def train_channel_only_baseline_3q(
    model,
    train_loader,
    val_loader,
    n_epochs=400,
    lr=8e-4,
    weight_decay=1e-5,
    lambda_phys=0.03
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    mse = nn.MSELoss()

    history = {
        "train_mse": [],
        "val_mse": [],
        "val_phys": []
    }

    best_val = np.inf
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()

        tr_total = 0.0
        n_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)

            ch_loss = mse(pred, yb)
            phys_loss = physicality_loss_channel_part(pred, Y_ch_mean_t, Y_ch_std_t)

            loss = ch_loss + lambda_phys * phys_loss

            loss.backward()
            optimizer.step()

            bs = xb.shape[0]
            tr_total += ch_loss.item() * bs
            n_total += bs

        train_mse = tr_total / n_total

        model.eval()

        val_total = 0.0
        val_phys_total = 0.0
        val_n = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)

                ch_loss = mse(pred, yb)
                phys_loss = physicality_loss_channel_part(pred, Y_ch_mean_t, Y_ch_std_t)

                bs = xb.shape[0]
                val_total += ch_loss.item() * bs
                val_phys_total += phys_loss.item() * bs
                val_n += bs

        val_mse = val_total / val_n
        val_phys = val_phys_total / val_n

        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["val_phys"].append(val_phys)

        if val_mse < best_val:
            best_val = val_mse
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"val ch {val_mse:.4e} | val phys {val_phys:.4e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


leakage_model_3q = LeakageAwareChannelNet3Q(
    input_dim=leak_train_3q["X"].shape[1],
    channel_dim=LABEL_DIM,
    leakage_dim=6
).to(device)

channel_only_model_leak_3q = ChannelOnlyNet3Q(
    input_dim=leak_train_3q["X"].shape[1],
    output_dim=LABEL_DIM
).to(device)

print("Training leakage-aware model...")
history_leakage_model_3q = train_leakage_aware_model_3q(
    leakage_model_3q,
    train_loader_aug,
    val_loader_aug,
    n_epochs=400,
    lr=8e-4,
    weight_decay=1e-5,
    lambda_phys=0.03,
    lambda_leak=1.0
)

print("\nTraining channel-only baseline on same leakage dataset...")
history_channel_only_leak_3q = train_channel_only_baseline_3q(
    channel_only_model_leak_3q,
    train_loader_ch,
    val_loader_ch,
    n_epochs=400,
    lr=8e-4,
    weight_decay=1e-5,
    lambda_phys=0.03
)

plt.figure(figsize=(8, 5))
plt.plot(history_leakage_model_3q["val_channel_mse"], label="Leakage-aware val channel MSE")
plt.plot(history_channel_only_leak_3q["val_mse"], label="Channel-only val MSE")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Validation standardized channel MSE")
plt.title("Leakage-aware multitask learning vs channel-only baseline")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history_leakage_model_3q["val_leak_mse"], label="Leakage val MSE")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Validation standardized leakage MSE")
plt.title("Leakage prediction training curve")
plt.legend()
plt.grid(True)
plt.show()

# %% Cell 32
# ============================================================
# PRA STEP 1G — Evaluate leakage-aware model
# ============================================================

def predict_leakage_aware_model_3q(model, X_s):
    model.eval()

    X_t = torch.tensor(X_s, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_s = model(X_t).cpu().numpy()

    pred_aug = invert_standardizer(pred_s, Y_aug_mean_leak, Y_aug_std_leak)

    pred_ch, pred_leak = split_augmented_channel_leakage(pred_aug)

    pred_leak = np.clip(pred_leak, 0.0, 1.0)

    return pred_aug.astype(np.float32), pred_ch.astype(np.float32), pred_leak.astype(np.float32)


def predict_channel_only_leak_3q(model, X_s):
    model.eval()

    X_t = torch.tensor(X_s, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_s = model(X_t).cpu().numpy()

    pred_ch = invert_standardizer(pred_s, Y_ch_mean_leak, Y_ch_std_leak)

    return pred_ch.astype(np.float32)


Y_aug_pred_test_leak_3q, Y_ch_pred_test_leakaware_3q, Y_leak_pred_test_3q = predict_leakage_aware_model_3q(
    leakage_model_3q,
    X_test_leak_s
)

Y_ch_pred_test_channelonly_3q = predict_channel_only_leak_3q(
    channel_only_model_leak_3q,
    X_test_leak_s
)

Y_ch_true_test_leak_3q = leak_test_3q["Y_channel"]
Y_leak_true_test_3q = leak_test_3q["Y_leakage"]

# Channel metrics
errs_leakaware_ch = per_example_l2_error(
    Y_ch_pred_test_leakaware_3q,
    Y_ch_true_test_leak_3q
)

errs_channelonly_ch = per_example_l2_error(
    Y_ch_pred_test_channelonly_3q,
    Y_ch_true_test_leak_3q
)

# Leakage metrics
leak_mae_total = np.mean(np.abs(Y_leak_pred_test_3q - Y_leak_true_test_3q))
leak_rmse_total = np.sqrt(np.mean((Y_leak_pred_test_3q - Y_leak_true_test_3q)**2))

leak_mae_meanpart = np.mean(np.abs(Y_leak_pred_test_3q[:, :3] - Y_leak_true_test_3q[:, :3]))
leak_mae_maxpart = np.mean(np.abs(Y_leak_pred_test_3q[:, 3:] - Y_leak_true_test_3q[:, 3:]))

leakage_eval_rows = [
    {
        "Model": "Channel-only",
        "Channel MSE": mse_np(Y_ch_pred_test_channelonly_3q, Y_ch_true_test_leak_3q),
        "Mean channel L2": errs_channelonly_ch.mean(),
        "Median channel L2": np.median(errs_channelonly_ch),
        "Leakage MAE": np.nan,
        "Leakage RMSE": np.nan,
        "Mean-leakage MAE": np.nan,
        "Max-leakage MAE": np.nan,
    },
    {
        "Model": "Leakage-aware multitask",
        "Channel MSE": mse_np(Y_ch_pred_test_leakaware_3q, Y_ch_true_test_leak_3q),
        "Mean channel L2": errs_leakaware_ch.mean(),
        "Median channel L2": np.median(errs_leakaware_ch),
        "Leakage MAE": leak_mae_total,
        "Leakage RMSE": leak_rmse_total,
        "Mean-leakage MAE": leak_mae_meanpart,
        "Max-leakage MAE": leak_mae_maxpart,
    }
]

leakage_eval_table_3q = pd.DataFrame(leakage_eval_rows)

display(leakage_eval_table_3q)

# Predicted vs true leakage
plt.figure(figsize=(6, 6))
plt.scatter(
    Y_leak_true_test_3q.flatten(),
    Y_leak_pred_test_3q.flatten(),
    alpha=0.7
)
lim_max = max(Y_leak_true_test_3q.max(), Y_leak_pred_test_3q.max()) * 1.05
plt.plot([0, lim_max], [0, lim_max], linestyle="--")
plt.xlabel("True leakage label")
plt.ylabel("Predicted leakage label")
plt.title("Leakage-aware model: predicted vs true leakage")
plt.grid(True)
plt.show()

# Per-qubit mean leakage
plt.figure(figsize=(7, 4))
labels = ["mean q1", "mean q2", "mean q3", "max q1", "max q2", "max q3"]
mae_per_component = np.mean(np.abs(Y_leak_pred_test_3q - Y_leak_true_test_3q), axis=0)

plt.bar(np.arange(6), mae_per_component)
plt.xticks(np.arange(6), labels, rotation=25)
plt.ylabel("MAE")
plt.title("Leakage prediction error by component")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# %% Cell 33
# ============================================================
# PRA STEP 1H — Leakage as reliability indicator
# ============================================================

# Define true and predicted total leakage scores
true_total_leakage_test_3q = Y_leak_true_test_3q[:, :3].mean(axis=1)
pred_total_leakage_test_3q = Y_leak_pred_test_3q[:, :3].mean(axis=1)

channel_error_leakaware = per_example_l2_error(
    Y_ch_pred_test_leakaware_3q,
    Y_ch_true_test_leak_3q
)

channel_error_channelonly = per_example_l2_error(
    Y_ch_pred_test_channelonly_3q,
    Y_ch_true_test_leak_3q
)

from scipy.stats import pearsonr, spearmanr

pear_true_leak_ch = pearsonr(true_total_leakage_test_3q, channel_error_leakaware)
spear_true_leak_ch = spearmanr(true_total_leakage_test_3q, channel_error_leakaware)

pear_pred_leak_ch = pearsonr(pred_total_leakage_test_3q, channel_error_leakaware)
spear_pred_leak_ch = spearmanr(pred_total_leakage_test_3q, channel_error_leakaware)

print("True leakage vs channel error:")
print("Pearson:", pear_true_leak_ch.statistic, "p =", pear_true_leak_ch.pvalue)
print("Spearman:", spear_true_leak_ch.statistic, "p =", spear_true_leak_ch.pvalue)

print("\nPredicted leakage vs channel error:")
print("Pearson:", pear_pred_leak_ch.statistic, "p =", pear_pred_leak_ch.pvalue)
print("Spearman:", spear_pred_leak_ch.statistic, "p =", spear_pred_leak_ch.pvalue)

plt.figure(figsize=(6, 5))
plt.scatter(true_total_leakage_test_3q, channel_error_leakaware, label="True leakage", alpha=0.75)
plt.scatter(pred_total_leakage_test_3q, channel_error_leakaware, label="Predicted leakage", alpha=0.75)
plt.xlabel("Leakage score")
plt.ylabel("Channel-label L2 error")
plt.title("Leakage as an error-process reliability indicator")
plt.legend()
plt.grid(True)
plt.show()

# High-leakage triage
threshold_leak = np.quantile(pred_total_leakage_test_3q, 0.75)

high_pred_leak = pred_total_leakage_test_3q >= threshold_leak
low_pred_leak = pred_total_leakage_test_3q < threshold_leak

leakage_triage_table_3q = pd.DataFrame([
    {
        "Group": "low/medium predicted leakage",
        "n": int(low_pred_leak.sum()),
        "Mean predicted leakage": pred_total_leakage_test_3q[low_pred_leak].mean(),
        "Mean true leakage": true_total_leakage_test_3q[low_pred_leak].mean(),
        "Mean channel L2": channel_error_leakaware[low_pred_leak].mean(),
    },
    {
        "Group": "high predicted leakage",
        "n": int(high_pred_leak.sum()),
        "Mean predicted leakage": pred_total_leakage_test_3q[high_pred_leak].mean(),
        "Mean true leakage": true_total_leakage_test_3q[high_pred_leak].mean(),
        "Mean channel L2": channel_error_leakaware[high_pred_leak].mean(),
    },
])

display(leakage_triage_table_3q)

# %% Cell 34
# ============================================================
# PRA STEP 1I — QAOA evaluation for leakage-aware model
# ============================================================

if "evaluate_qaoa_device_3q" not in globals():
    raise RuntimeError("QAOA functions are missing. Load/run QAOA function definitions first.")

leakage_qaoa_prediction_methods_3q = {
    "Channel-only": Y_ch_pred_test_channelonly_3q,
    "Leakage-aware channels": Y_ch_pred_test_leakaware_3q,
}

N_QAOA_LEAK_EVAL_3Q = min(10, len(Y_ch_true_test_leak_3q))

results_qaoa_leakage_3q = []

for k in range(N_QAOA_LEAK_EVAL_3Q):
    res = evaluate_qaoa_device_3q(
        idx_device=k,
        Y_true_all=Y_ch_true_test_leak_3q,
        prediction_methods=leakage_qaoa_prediction_methods_3q,
        qaoa_shots=qaoa_shots_3q,
        seed=110000
    )

    results_qaoa_leakage_3q.append(res)

    msg = f"Device {k:02d} | Noisy MAE {res['Noisy']['error']['MAE']:.4f}"

    for method in leakage_qaoa_prediction_methods_3q.keys():
        msg += f" | {method} {res[method]['error']['MAE']:.4f}"

    print(msg)

# Summary table
leakage_qaoa_rows = []

methods_leak_qaoa = ["Noisy"] + list(leakage_qaoa_prediction_methods_3q.keys())

noisy_mae_mean_leak = np.array([
    r["Noisy"]["error"]["MAE"]
    for r in results_qaoa_leakage_3q
]).mean()

for method in methods_leak_qaoa:
    mae_arr = np.array([r[method]["error"]["MAE"] for r in results_qaoa_leakage_3q])
    rmse_arr = np.array([r[method]["error"]["RMSE"] for r in results_qaoa_leakage_3q])
    regret_arr = np.array([r[method]["selected"]["regret"] for r in results_qaoa_leakage_3q])
    approx_arr = np.array([r[method]["selected"]["approximation_ratio"] for r in results_qaoa_leakage_3q])
    popt_arr = np.array([r[method]["selected"]["p_opt_selected"] for r in results_qaoa_leakage_3q])
    disp_arr = np.array([r[method]["selected"]["parameter_displacement"] for r in results_qaoa_leakage_3q])

    improvement = 1.0 if method == "Noisy" else improvement_ratio(noisy_mae_mean_leak, mae_arr.mean())

    leakage_qaoa_rows.append({
        "Method": method,
        "Mean MAE": mae_arr.mean(),
        "Median MAE": np.median(mae_arr),
        "Mean RMSE": rmse_arr.mean(),
        "Mean regret": regret_arr.mean(),
        "Mean approximation ratio": approx_arr.mean(),
        "Mean P_opt": popt_arr.mean(),
        "Mean parameter displacement": disp_arr.mean(),
        "Improvement ratio": improvement
    })

leakage_qaoa_table_3q = pd.DataFrame(leakage_qaoa_rows)

display(leakage_qaoa_table_3q)

plt.figure(figsize=(7, 4))
xpos = np.arange(len(leakage_qaoa_table_3q))
plt.bar(xpos, leakage_qaoa_table_3q["Mean MAE"].values)
plt.xticks(xpos, leakage_qaoa_table_3q["Method"].values, rotation=25)
plt.ylabel("Mean MAE vs ideal QAOA landscape")
plt.title("Leakage-aware representation: QAOA reliability")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# %% Cell 35
# ============================================================
# PRA STEP 1J — Leakage-aware summary
# ============================================================

print("=" * 90)
print("PRA STEP 1: LEAKAGE-AWARE ERROR REPRESENTATION SUMMARY")
print("=" * 90)

display(leakage_eval_table_3q)
display(leakage_triage_table_3q)
display(leakage_qaoa_table_3q)

# Channel comparison
ch_only_l2 = leakage_eval_table_3q[
    leakage_eval_table_3q["Model"] == "Channel-only"
]["Mean channel L2"].values[0]

leak_l2 = leakage_eval_table_3q[
    leakage_eval_table_3q["Model"] == "Leakage-aware multitask"
]["Mean channel L2"].values[0]

print("\nChannel comparison:")
print("Channel-only Mean L2:", ch_only_l2)
print("Leakage-aware Mean L2:", leak_l2)

if leak_l2 < ch_only_l2:
    print("Result: multitask leakage prediction improves channel inference.")
else:
    print("Result: multitask leakage prediction does not improve channel inference yet.")

# Leakage prediction
leak_mae = leakage_eval_table_3q[
    leakage_eval_table_3q["Model"] == "Leakage-aware multitask"
]["Leakage MAE"].values[0]

print("\nLeakage prediction MAE:", leak_mae)

# QAOA comparison
df_q = leakage_qaoa_table_3q[leakage_qaoa_table_3q["Method"] != "Noisy"].copy()
best_idx = df_q["Mean MAE"].idxmin()

print("\nBest QAOA method:")
print(df_q.loc[best_idx, ["Method", "Mean MAE", "Improvement ratio", "Mean approximation ratio"]])

if "Leakage-aware channels" in df_q["Method"].values and "Channel-only" in df_q["Method"].values:
    leak_mae_q = df_q[df_q["Method"] == "Leakage-aware channels"]["Mean MAE"].values[0]
    ch_mae_q = df_q[df_q["Method"] == "Channel-only"]["Mean MAE"].values[0]

    print("\nQAOA comparison:")
    print("Channel-only QAOA MAE:", ch_mae_q)
    print("Leakage-aware QAOA MAE:", leak_mae_q)

    if leak_mae_q < ch_mae_q:
        print("Result: leakage-aware representation improves downstream QAOA mitigation.")
    else:
        print("Result: leakage-aware representation is useful diagnostically, but not yet better for QAOA mitigation.")

# %% Cell 36
# ============================================================
# CELL 106 — Save Step 11 leakage-aware results
# ============================================================


STEP11_DIR = "outputs/three_qubit_qaoa_step11_leakage_aware"
os.makedirs(STEP11_DIR, exist_ok=True)

# Save tables
leakage_eval_table_3q.to_csv(
    os.path.join(STEP11_DIR, "leakage_eval_table_3q.csv"),
    index=False
)

leakage_triage_table_3q.to_csv(
    os.path.join(STEP11_DIR, "leakage_triage_table_3q.csv"),
    index=False
)

leakage_qaoa_table_3q.to_csv(
    os.path.join(STEP11_DIR, "leakage_qaoa_table_3q.csv"),
    index=False
)

# Save models
torch.save(
    leakage_model_3q.state_dict(),
    os.path.join(STEP11_DIR, "leakage_aware_channelnet_3q.pt")
)

torch.save(
    channel_only_model_leak_3q.state_dict(),
    os.path.join(STEP11_DIR, "channel_only_baseline_leak_3q.pt")
)

# Save arrays/results
step11_pack = {
    "K_LEAK_3Q": K_LEAK_3Q,
    "N_TRAIN_LEAK_3Q": N_TRAIN_LEAK_3Q,
    "N_VAL_LEAK_3Q": N_VAL_LEAK_3Q,
    "N_TEST_LEAK_3Q": N_TEST_LEAK_3Q,
    "leak_train_3q": leak_train_3q,
    "leak_val_3q": leak_val_3q,
    "leak_test_3q": leak_test_3q,
    "X_mean_leak": X_mean_leak,
    "X_std_leak": X_std_leak,
    "Y_aug_mean_leak": Y_aug_mean_leak,
    "Y_aug_std_leak": Y_aug_std_leak,
    "Y_ch_mean_leak": Y_ch_mean_leak,
    "Y_ch_std_leak": Y_ch_std_leak,
    "Y_ch_pred_test_leakaware_3q": Y_ch_pred_test_leakaware_3q,
    "Y_ch_pred_test_channelonly_3q": Y_ch_pred_test_channelonly_3q,
    "Y_leak_pred_test_3q": Y_leak_pred_test_3q,
    "Y_leak_true_test_3q": Y_leak_true_test_3q,
    "leakage_eval_table_3q": leakage_eval_table_3q,
    "leakage_triage_table_3q": leakage_triage_table_3q,
    "leakage_qaoa_table_3q": leakage_qaoa_table_3q,
    "results_qaoa_leakage_3q": results_qaoa_leakage_3q,
    "history_leakage_model_3q": history_leakage_model_3q,
    "history_channel_only_leak_3q": history_channel_only_leak_3q,
}

with open(os.path.join(STEP11_DIR, "step11_leakage_aware_results.pkl"), "wb") as f:
    pickle.dump(step11_pack, f, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ Saved Step 11 leakage-aware results to:")
print(STEP11_DIR)

# %% Cell 37
# ============================================================
# CELL 107 — Pairwise correlated-error labels
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import os
import pickle
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

PAIR_EDGES_3Q = [(0, 1), (1, 2)]
PAIR_PAULI_LABELS = ["X", "Y", "Z"]

print("Local channel dim:", LABEL_DIM)
print("Pair-edge setup loaded. PRA Step 2 below defines the process-relative pair label dimension.")

required_for_step12 = [
    "sample_hidden_device_3q",
    "build_all_pulse_superoperators",
    "generate_full_tomography_vector_cached",
    "reference_all_local_channels_label_cached",
    "prepare_local_tomography_state",
    "ket_to_rho",
    "evolve_rho_with_superoperator",
    "measurement_ops",
    "make_masked_tomography_input",
    "FULL_TOMO_DIM",
    "LABEL_DIM",
    "N_QUBITS",
]

missing = [name for name in required_for_step12 if name not in globals()]
if missing:
    raise RuntimeError("Missing required simulator objects for Step 12: " + str(missing))

print("Step 12 requirements are available.")

# %% Cell 38
# ============================================================
# CELL 108 — Pair probes and pair expectations
# ============================================================

def local_plus_state_for_pauli(label):
    """
    +1 eigenstate of X, Y, or Z inside the qutrit computational subspace.
    """
    if label == "X":
        return local_state_dict["+"]
    elif label == "Y":
        return local_state_dict["+i"]
    elif label == "Z":
        return local_state_dict["0"]
    else:
        raise ValueError("label must be X, Y, or Z.")


def prepare_pair_pauli_probe_state(edge, label_i, label_j, spectator_label="0"):
    """
    Prepare a product state where the two qubits in edge are +1 eigenstates
    of Pauli labels label_i and label_j.
    """
    qi, qj = edge
    states = []

    for q in range(N_QUBITS):
        if q == qi:
            states.append(local_plus_state_for_pauli(label_i))
        elif q == qj:
            states.append(local_plus_state_for_pauli(label_j))
        else:
            states.append(local_state_dict[spectator_label])

    return kron_list(states)


def exact_pair_expectation_with_readout(rho, params, qi, qj, label_i, label_j):
    """
    Pair expectation with independent readout errors.

    If ideal measured variables are s_i, s_j in ±1,
    readout transforms:
        m_i' = a_i m_i + b_i
        m_j' = a_j m_j + b_j
        e_ij' = a_i a_j e_ij + a_i b_j m_i + b_i a_j m_j + b_i b_j

    where:
        a = 1 - r01 - r10
        b = r10 - r01
    """
    Oi = measurement_ops[qi][label_i]
    Oj = measurement_ops[qj][label_j]
    Oij = Oi @ Oj

    mi = trace_expectation(rho, Oi)
    mj = trace_expectation(rho, Oj)
    eij = trace_expectation(rho, Oij)

    ai = 1.0 - params["r01"][qi] - params["r10"][qi]
    bi = params["r10"][qi] - params["r01"][qi]

    aj = 1.0 - params["r01"][qj] - params["r10"][qj]
    bj = params["r10"][qj] - params["r01"][qj]

    eij_meas = ai * aj * eij + ai * bj * mi + bi * aj * mj + bi * bj

    return float(np.clip(eij_meas, -1.0, 1.0))


def local_product_pair_prediction_from_channels(y_channel, edge, label_i, label_j):
    """
    Predict pair expectation from product of local affine channels.

    For the pair probe input, the input Bloch vector is +1 along the
    requested Pauli axis for each qubit.
    """
    A_list, b_list = unflatten_local_channels(y_channel, N=N_QUBITS)

    qi, qj = edge

    axis_map = {"X": 0, "Y": 1, "Z": 2}

    r_i_in = np.zeros(3)
    r_j_in = np.zeros(3)

    r_i_in[axis_map[label_i]] = 1.0
    r_j_in[axis_map[label_j]] = 1.0

    r_i_out = A_list[qi] @ r_i_in + b_list[qi]
    r_j_out = A_list[qj] @ r_j_in + b_list[qj]

    return float(r_i_out[axis_map[label_i]] * r_j_out[axis_map[label_j]])


def pairwise_residual_labels_cached(params, S_list, y_channel):
    """
    Compute pair residual labels.

    For each neighboring edge and Pauli pair ab:
        exact pair response under sequential pulses on qi then qj
        minus product-local prediction.

    delta_ij^ab = exact_pair_expectation - local_product_prediction

    Output shape: 18
    """
    residuals = []

    for edge in PAIR_EDGES_3Q:
        qi, qj = edge

        for label_i in PAIR_PAULI_LABELS:
            for label_j in PAIR_PAULI_LABELS:

                psi0 = prepare_pair_pauli_probe_state(
                    edge=edge,
                    label_i=label_i,
                    label_j=label_j,
                    spectator_label="0"
                )

                rho0 = ket_to_rho(psi0)

                # Sequential local pulse probes on the two qubits.
                rho1 = evolve_rho_with_superoperator(rho0, S_list[qi])
                rho2 = evolve_rho_with_superoperator(rho1, S_list[qj])

                exact_pair = exact_pair_expectation_with_readout(
                    rho=rho2,
                    params=params,
                    qi=qi,
                    qj=qj,
                    label_i=label_i,
                    label_j=label_j
                )

                local_pair = local_product_pair_prediction_from_channels(
                    y_channel=y_channel,
                    edge=edge,
                    label_i=label_i,
                    label_j=label_j
                )

                residuals.append(exact_pair - local_pair)

    return np.array(residuals, dtype=np.float32)


def augment_channel_with_pair_residual(y_channel, y_pair):
    return np.concatenate([
        np.asarray(y_channel, dtype=np.float32),
        np.asarray(y_pair, dtype=np.float32)
    ]).astype(np.float32)


def split_channel_pair(y_aug):
    y_aug = np.asarray(y_aug)
    return y_aug[..., :LABEL_DIM], y_aug[..., LABEL_DIM:LABEL_DIM+PAIR_LABEL_DIM]

# %% Cell 39
# ============================================================
# PRA STEP 2 — Process-relative pair residual labels
# ============================================================

# This cell replaces the earlier scalar pair-correlation residual
#     <sigma_i^a sigma_j^b>_true - <sigma_i^a>_local <sigma_j^b>_local
# by a process-relative residual evaluated on the same pair-probe input ensemble.
# For each hardware edge, we compare the measured two-qubit response to the
# response predicted by the tensor product of the two learned local affine maps.

PAIR_AXIS_PAIRS = [
    (li, lj)
    for li in PAIR_PAULI_LABELS
    for lj in PAIR_PAULI_LABELS
]

PAIR_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}

# One compact edge-process block has 9 output Pauli-pair observables by
# 9 input Pauli-pair probe preparations. There are two edges in the chain.
PAIR_PROCESS_BLOCK_DIM = len(PAIR_AXIS_PAIRS) * len(PAIR_AXIS_PAIRS)  # 9 x 9 = 81
PAIR_PROCESS_LABEL_DIM = len(PAIR_EDGES_3Q) * PAIR_PROCESS_BLOCK_DIM

# Keep the old variable name so the rest of the notebook remains compatible.
PAIR_LABEL_DIM = PAIR_PROCESS_LABEL_DIM
PAIR_AUG_DIM = LABEL_DIM + PAIR_LABEL_DIM
PAIR_RESIDUAL_KIND = "process_relative_edge_response_9x9_per_edge"

print("PRA Step 2 residual kind:", PAIR_RESIDUAL_KIND)
print("PAIR_AXIS_PAIRS:", PAIR_AXIS_PAIRS)
print("Pair residual label dimension:", PAIR_LABEL_DIM)
print("Augmented output dimension:", PAIR_AUG_DIM)


def local_input_bloch_vector_for_pauli(label):
    """Bloch vector of the +1 eigenstate of X, Y, or Z."""
    r = np.zeros(3, dtype=np.float64)
    r[PAIR_AXIS_INDEX[label]] = 1.0
    return r


def local_product_pair_prediction_from_channels_process_relative(
    y_channel,
    edge,
    input_label_i,
    input_label_j,
    output_label_i,
    output_label_j
):
    """
    Tensor-product prediction from the two learned local response maps.

    The local maps act on the same pair-probe input ensemble as the true
    hidden edge process. We then read out the requested output Pauli-pair
    observable from the product of the two local output Bloch vectors.
    """
    A_list, b_list = unflatten_local_channels(y_channel, N=N_QUBITS)
    qi, qj = edge

    r_i_in = local_input_bloch_vector_for_pauli(input_label_i)
    r_j_in = local_input_bloch_vector_for_pauli(input_label_j)

    r_i_out = A_list[qi] @ r_i_in + b_list[qi]
    r_j_out = A_list[qj] @ r_j_in + b_list[qj]

    ai = PAIR_AXIS_INDEX[output_label_i]
    aj = PAIR_AXIS_INDEX[output_label_j]

    return float(r_i_out[ai] * r_j_out[aj])


def edge_process_relative_residual_block_cached(params, S_list, y_channel, edge):
    """
    Return one 9 x 9 process-relative residual block for a hardware edge.

    Rows correspond to output Pauli-pair observables (XX, XY, ..., ZZ).
    Columns correspond to input pair-probe preparations (+X/+X, +X/+Y, ..., +Z/+Z).

    residual[out_pair, in_pair] =
        true edge response on the qutrit simulator
        - tensor-product response from the two learned local maps

    This compares an edge response and a product-local response on the same
    probe input ensemble, rather than subtracting products of QAOA-state
    expectation values.
    """
    qi, qj = edge
    residual_block = np.zeros((len(PAIR_AXIS_PAIRS), len(PAIR_AXIS_PAIRS)), dtype=np.float64)

    for in_idx, (input_label_i, input_label_j) in enumerate(PAIR_AXIS_PAIRS):
        psi0 = prepare_pair_pauli_probe_state(
            edge=edge,
            label_i=input_label_i,
            label_j=input_label_j,
            spectator_label="0"
        )
        rho0 = ket_to_rho(psi0)

        # Same edge-probe evolution as before: sequential local pulses on the two qubits.
        rho1 = evolve_rho_with_superoperator(rho0, S_list[qi])
        rho2 = evolve_rho_with_superoperator(rho1, S_list[qj])

        for out_idx, (output_label_i, output_label_j) in enumerate(PAIR_AXIS_PAIRS):
            true_edge_response = exact_pair_expectation_with_readout(
                rho=rho2,
                params=params,
                qi=qi,
                qj=qj,
                label_i=output_label_i,
                label_j=output_label_j
            )

            product_local_response = local_product_pair_prediction_from_channels_process_relative(
                y_channel=y_channel,
                edge=edge,
                input_label_i=input_label_i,
                input_label_j=input_label_j,
                output_label_i=output_label_i,
                output_label_j=output_label_j
            )

            residual_block[out_idx, in_idx] = true_edge_response - product_local_response

    return residual_block.astype(np.float32)


def process_relative_pair_residual_labels_cached(params, S_list, y_channel):
    """
    Process-relative pair residual labels for all hardware edges.

    Output shape:
        len(PAIR_EDGES_3Q) * 9 * 9 = 162 for the 3-qubit chain.
    """
    blocks = []
    for edge in PAIR_EDGES_3Q:
        block = edge_process_relative_residual_block_cached(
            params=params,
            S_list=S_list,
            y_channel=y_channel,
            edge=edge
        )
        blocks.append(block.reshape(-1))

    return np.concatenate(blocks).astype(np.float32)


# Backward-compatible name used by the dataset builders.
def pairwise_residual_labels_cached(params, S_list, y_channel):
    return process_relative_pair_residual_labels_cached(
        params=params,
        S_list=S_list,
        y_channel=y_channel
    )


def augment_channel_with_pair_residual(y_channel, y_pair):
    return np.concatenate([
        np.asarray(y_channel, dtype=np.float32),
        np.asarray(y_pair, dtype=np.float32)
    ]).astype(np.float32)


def split_channel_pair(y_aug):
    y_aug = np.asarray(y_aug)
    return y_aug[..., :LABEL_DIM], y_aug[..., LABEL_DIM:LABEL_DIM+PAIR_LABEL_DIM]

# %% Cell 40
# Hard guard: the final notebook must keep the PRA Step 2 process-relative residual.
assert PAIR_LABEL_DIM == len(PAIR_EDGES_3Q) * 9 * 9, (PAIR_LABEL_DIM, len(PAIR_EDGES_3Q) * 9 * 9)
assert PAIR_RESIDUAL_KIND == "process_relative_edge_response_9x9_per_edge"
print("PRA Step 2 guard passed: process-relative residual dimension =", PAIR_LABEL_DIM)

# %% Cell 41
# ============================================================
# CELL 109 — Pairwise-augmented dataset
# ============================================================

def perturb_for_pair_learning_3q(params, rng=None):
    """
    Enrich pairwise-correlated-error variation.

    This emphasizes coupling, residual ZZ, detuning disorder,
    and correlated dephasing.
    """
    if rng is None:
        rng = np.random.default_rng()

    p = copy.deepcopy(params)

    family = rng.choice(
        ["nominal", "strong_coupling", "strong_zz", "detuning_disorder", "correlated_dephasing"],
        p=[0.25, 0.25, 0.20, 0.15, 0.15]
    )

    if family == "nominal":
        pass

    elif family == "strong_coupling":
        factor = rng.uniform(1.3, 2.2)
        for edge in p["edges"]:
            p["g"][edge] *= factor

    elif family == "strong_zz":
        factor = rng.uniform(1.5, 3.0)
        for edge in p["edges"]:
            p["zeta"][edge] *= factor

    elif family == "detuning_disorder":
        extra_sigma_GHz = rng.uniform(0.006, 0.020)
        p["Delta"] = p["Delta"] + two_pi(rng.normal(0.0, extra_sigma_GHz, size=p["N"]))

    elif family == "correlated_dephasing":
        p["correlated_dephasing_rate"] = rng.uniform(0.5e-5, 8.0e-5)

    p["noise_family"] = "pair_learning_" + family

    return p


def generate_one_pair_augmented_example_3q(
    rng,
    K=18,
    mask_mode="random",
    topology="chain",
    shot_choices=np.array([128, 512, 1024, 4096], dtype=int),
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    pair_enriched=True
):
    shots = int(rng.choice(shot_choices))

    params = sample_hidden_device_3q(rng=rng, topology=topology)

    if pair_enriched:
        params = perturb_for_pair_learning_3q(params, rng=rng)

    S_list = build_all_pulse_superoperators(
        params=params,
        theta=pulse_theta,
        phase=pulse_phase,
        T=pulse_T,
        include_decoherence=True
    )

    x_full = generate_full_tomography_vector_cached(
        params=params,
        S_list=S_list,
        shots=shots,
        rng=rng
    )

    y_channel = reference_all_local_channels_label_cached(
        params=params,
        S_list=S_list,
        include_readout=True
    )

    y_pair = pairwise_residual_labels_cached(
        params=params,
        S_list=S_list,
        y_channel=y_channel
    )

    y_aug = augment_channel_with_pair_residual(
        y_channel=y_channel,
        y_pair=y_pair
    )

    if mask_mode == "random":
        observed_indices = random_mask_indices(K=K, rng=rng)
    elif mask_mode == "fixed":
        observed_indices = fixed_partial_indices
    elif mask_mode == "complete":
        observed_indices = np.arange(FULL_TOMO_DIM, dtype=int)
    else:
        raise ValueError("mask_mode must be random, fixed, or complete.")

    x_masked = make_masked_tomography_input(
        x_full_tomo=x_full,
        observed_indices=observed_indices,
        shots=shots
    )

    return (
        x_masked.astype(np.float32),
        y_aug.astype(np.float32),
        y_channel.astype(np.float32),
        y_pair.astype(np.float32),
        x_full.astype(np.float32),
        params,
        shots,
        observed_indices.astype(int)
    )


def build_pair_augmented_dataset_3q(
    n_examples,
    seed=0,
    K=18,
    mask_mode="random",
    topology="chain",
    pair_enriched=True,
    print_every=5
):
    rng = np.random.default_rng(seed)

    X = []
    Y_aug = []
    Y_channel = []
    Y_pair = []
    X_full = []
    params_list = []
    shots_list = []
    masks_list = []

    t0 = time.time()

    for i in range(n_examples):
        x, y_aug, y_ch, y_pair, x_full, params, shots, obs = generate_one_pair_augmented_example_3q(
            rng=rng,
            K=K,
            mask_mode=mask_mode,
            topology=topology,
            pair_enriched=pair_enriched
        )

        X.append(x)
        Y_aug.append(y_aug)
        Y_channel.append(y_ch)
        Y_pair.append(y_pair)
        X_full.append(x_full)
        params_list.append(params)
        shots_list.append(shots)
        masks_list.append(obs)

        if (i + 1) % print_every == 0 or (i + 1) == n_examples:
            print(
                f"Pair dataset {i+1:4d}/{n_examples} "
                f"| elapsed {time.time()-t0:.1f} s"
            )

    return {
        "X": np.stack(X).astype(np.float32),
        "Y_aug": np.stack(Y_aug).astype(np.float32),
        "Y_channel": np.stack(Y_channel).astype(np.float32),
        "Y_pair": np.stack(Y_pair).astype(np.float32),
        "X_full": np.stack(X_full).astype(np.float32),
        "params": params_list,
        "shots": np.array(shots_list, dtype=int),
        "masks": masks_list,
    }

# %% Cell 42
# ============================================================
# CLEAN PAIR HELPER DEFINITIONS — required by Step 5 and final pair models
# ============================================================

bloch_test_np_pair = np.array([
    [ 1,  0,  0],
    [-1,  0,  0],
    [ 0,  1,  0],
    [ 0, -1,  0],
    [ 0,  0,  1],
    [ 0,  0, -1],
    [ 1,  1,  1],
    [ 1, -1,  1],
    [-1,  1,  1],
    [ 1,  1, -1],
], dtype=np.float32)

for _i in range(6, len(bloch_test_np_pair)):
    bloch_test_np_pair[_i] /= np.linalg.norm(bloch_test_np_pair[_i])

bloch_test_t_pair = torch.tensor(bloch_test_np_pair, dtype=torch.float32).to(device)


def split_channels_torch_pair(y_channel_phys):
    """Split a batched 36-vector into A and b for three local affine maps."""
    batch = y_channel_phys.shape[0]
    A_all = []
    b_all = []
    offset = 0
    for q in range(N_QUBITS):
        A_q = y_channel_phys[:, offset:offset+9].reshape(batch, 3, 3)
        offset += 9
        b_q = y_channel_phys[:, offset:offset+3]
        offset += 3
        A_all.append(A_q)
        b_all.append(b_q)
    return torch.stack(A_all, dim=1), torch.stack(b_all, dim=1)

print("Clean pair torch helpers loaded.")

# %% Cell 43
# ============================================================
# CELL 120 — Train/evaluate pair-probe-aware model
# ============================================================

class PairProbeAwareNet3Q(nn.Module):
    def __init__(self, input_dim, channel_dim=36, pair_dim=18):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 320),
            nn.ReLU(),

            nn.Linear(320, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.channel_head = nn.Linear(128, channel_dim)

        self.pair_head = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, pair_dim)
        )

    def forward(self, x):
        h = self.backbone(x)

        y_channel = self.channel_head(h)
        y_pair = self.pair_head(h)

        return torch.cat([y_channel, y_pair], dim=-1)


def compute_standardizer(X, eps=1e-8):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return mean.astype(np.float32), std.astype(np.float32)

def apply_standardizer(X, mean, std):
    return ((X - mean) / std).astype(np.float32)

def invert_standardizer(Xs, mean, std):
    return (Xs * std + mean).astype(np.float32)

def make_loader_pp(X_s, Y_s, batch_size=16, shuffle=True):
    return DataLoader(
        TensorDataset(
            torch.tensor(X_s, dtype=torch.float32),
            torch.tensor(Y_s, dtype=torch.float32)
        ),
        batch_size=batch_size,
        shuffle=shuffle
    )


def train_pairprobe_model_for_K(
    K_pair,
    n_epochs=400,
    lambda_pair=1.0,
    lambda_phys=0.03,
    seed=0
):
    """
    Train one model for a given number of pair probes.
    """
    print("\n" + "=" * 90)
    print(f"Training pair-probe-aware model with K_pair = {K_pair}")
    print("=" * 90)

    ds_train = build_local_pairprobe_dataset(
        pair_dataset=pair_train_3q,
        Xpair_probe=Xpair_probe_train_3q,
        K_pair=K_pair,
        seed=seed + 10
    )

    ds_val = build_local_pairprobe_dataset(
        pair_dataset=pair_val_3q,
        Xpair_probe=Xpair_probe_val_3q,
        K_pair=K_pair,
        seed=seed + 20
    )

    ds_test = build_local_pairprobe_dataset(
        pair_dataset=pair_test_3q,
        Xpair_probe=Xpair_probe_test_3q,
        K_pair=K_pair,
        seed=seed + 30
    )

    X_mean, X_std = compute_standardizer(ds_train["X"])
    Y_mean, Y_std = compute_standardizer(ds_train["Y_aug"])
    Y_ch_mean, Y_ch_std = compute_standardizer(ds_train["Y_channel"])

    X_train_s = apply_standardizer(ds_train["X"], X_mean, X_std)
    X_val_s = apply_standardizer(ds_val["X"], X_mean, X_std)
    X_test_s = apply_standardizer(ds_test["X"], X_mean, X_std)

    Y_train_s = apply_standardizer(ds_train["Y_aug"], Y_mean, Y_std)
    Y_val_s = apply_standardizer(ds_val["Y_aug"], Y_mean, Y_std)

    train_loader = make_loader_pp(X_train_s, Y_train_s, batch_size=16, shuffle=True)
    val_loader = make_loader_pp(X_val_s, Y_val_s, batch_size=64, shuffle=False)

    model = PairProbeAwareNet3Q(
        input_dim=ds_train["X"].shape[1],
        channel_dim=LABEL_DIM,
        pair_dim=PAIR_LABEL_DIM
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=8e-4,
        weight_decay=1e-5
    )

    mse = nn.MSELoss()

    Y_ch_mean_t_local = torch.tensor(Y_ch_mean, dtype=torch.float32).to(device)
    Y_ch_std_t_local = torch.tensor(Y_ch_std, dtype=torch.float32).to(device)

    def physicality_loss_local(y_channel_s):
        y_phys = y_channel_s * Y_ch_std_t_local + Y_ch_mean_t_local
        A, b = split_channels_torch_pair(y_phys)

        r = bloch_test_t_pair[None, None, :, :]

        penalties = []

        for q in range(N_QUBITS):
            A_q = A[:, q, :, :]
            b_q = b[:, q, :]

            out = torch.matmul(r, A_q.transpose(1, 2)[:, None, :, :])
            out = out.squeeze(1) + b_q[:, None, :]

            norms = torch.linalg.norm(out, dim=-1)
            penalties.append((torch.relu(norms - 1.0) ** 2).mean())

        return sum(penalties) / len(penalties)

    history = {
        "train_channel_mse": [],
        "train_pair_mse": [],
        "val_channel_mse": [],
        "val_pair_mse": [],
        "val_phys": [],
    }

    best_score = np.inf
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()

        tr_ch = 0.0
        tr_pair = 0.0
        n_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)

            pred_ch = pred[:, :LABEL_DIM]
            pred_pair = pred[:, LABEL_DIM:LABEL_DIM+PAIR_LABEL_DIM]

            y_ch = yb[:, :LABEL_DIM]
            y_pair = yb[:, LABEL_DIM:LABEL_DIM+PAIR_LABEL_DIM]

            ch_loss = mse(pred_ch, y_ch)
            pair_loss = mse(pred_pair, y_pair)
            phys_loss = physicality_loss_local(pred_ch)

            loss = ch_loss + lambda_pair * pair_loss + lambda_phys * phys_loss

            loss.backward()
            optimizer.step()

            bs = xb.shape[0]
            tr_ch += ch_loss.item() * bs
            tr_pair += pair_loss.item() * bs
            n_total += bs

        model.eval()

        val_ch = 0.0
        val_pair = 0.0
        val_phys = 0.0
        val_n = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)

                pred_ch = pred[:, :LABEL_DIM]
                pred_pair = pred[:, LABEL_DIM:LABEL_DIM+PAIR_LABEL_DIM]

                y_ch = yb[:, :LABEL_DIM]
                y_pair = yb[:, LABEL_DIM:LABEL_DIM+PAIR_LABEL_DIM]

                ch_loss = mse(pred_ch, y_ch)
                pair_loss = mse(pred_pair, y_pair)
                phys_loss = physicality_loss_local(pred_ch)

                bs = xb.shape[0]
                val_ch += ch_loss.item() * bs
                val_pair += pair_loss.item() * bs
                val_phys += phys_loss.item() * bs
                val_n += bs

        val_ch /= val_n
        val_pair /= val_n
        val_phys /= val_n

        history["train_channel_mse"].append(tr_ch / n_total)
        history["train_pair_mse"].append(tr_pair / n_total)
        history["val_channel_mse"].append(val_ch)
        history["val_pair_mse"].append(val_pair)
        history["val_phys"].append(val_phys)

        score = val_ch + lambda_pair * val_pair

        if score < best_score:
            best_score = score
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"val ch {val_ch:.4e} | "
                f"val pair {val_pair:.4e} | "
                f"val phys {val_phys:.4e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    # Prediction on test set
    model.eval()

    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_s = model(X_test_t).cpu().numpy()

    pred_aug = invert_standardizer(pred_s, Y_mean, Y_std)
    pred_ch, pred_pair = split_channel_pair(pred_aug)

    result = {
        "K_pair": K_pair,
        "model": model,
        "history": history,
        "ds_train": ds_train,
        "ds_val": ds_val,
        "ds_test": ds_test,
        "X_mean": X_mean,
        "X_std": X_std,
        "Y_mean": Y_mean,
        "Y_std": Y_std,
        "Y_ch_mean": Y_ch_mean,
        "Y_ch_std": Y_ch_std,
        "Y_pred_aug": pred_aug.astype(np.float32),
        "Y_pred_channel": pred_ch.astype(np.float32),
        "Y_pred_pair": pred_pair.astype(np.float32),
    }

    return result

# %% Cell 44
# ============================================================
# CELL 118 — Generate pair-probe vectors for train/val/test
# ============================================================

def build_pair_probe_matrix_from_dataset(pair_dataset, seed=0, print_every=5):
    """
    Build pair-probe measurement matrix for an existing Step 12 dataset.

    Uses the same hidden params stored in pair_dataset["params"].
    """
    rng = np.random.default_rng(seed)

    X_pair_probe = []

    t0 = time.time()

    for i, params in enumerate(pair_dataset["params"]):
        shots = int(pair_dataset["shots"][i])

        S_list = build_all_pulse_superoperators(
            params=params,
            theta=np.pi / 2,
            phase=0.0,
            T=40.0,
            include_decoherence=True
        )

        x_pair = generate_pair_probe_vector_cached(
            params=params,
            S_list=S_list,
            shots=shots,
            rng=rng
        )

        X_pair_probe.append(x_pair)

        if (i + 1) % print_every == 0 or (i + 1) == len(pair_dataset["params"]):
            print(
                f"Pair probes {i+1:4d}/{len(pair_dataset['params'])} "
                f"| elapsed {time.time()-t0:.1f} s"
            )

    return np.stack(X_pair_probe).astype(np.float32)

# %% Cell 45
# ============================================================
# CELL 119 — Build local + pair-probe input datasets
# ============================================================

def random_pair_probe_indices(K_pair, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if K_pair == 0:
        return np.array([], dtype=int)

    return np.sort(
        rng.choice(np.arange(PAIR_PROBE_DIM_3Q), size=K_pair, replace=False)
    )


def make_local_plus_pairprobe_input(
    x_local_masked,
    x_pair_full,
    observed_pair_indices
):
    """
    Combine local masked tomography input with pair-probe masked input.

    x_local_masked shape:
        109 = 54 local values + 54 local mask + 1 shot feature

    x_pair_full shape:
        18

    appended pair part:
        18 pair values with missing=0
        18 pair mask bits

    output shape:
        109 + 18 + 18 = 145
    """
    x_local_masked = np.asarray(x_local_masked, dtype=np.float32)
    x_pair_full = np.asarray(x_pair_full, dtype=np.float32)
    observed_pair_indices = np.asarray(observed_pair_indices, dtype=int)

    pair_values = np.zeros(PAIR_PROBE_DIM_3Q, dtype=np.float32)
    pair_mask = np.zeros(PAIR_PROBE_DIM_3Q, dtype=np.float32)

    pair_values[observed_pair_indices] = x_pair_full[observed_pair_indices]
    pair_mask[observed_pair_indices] = 1.0

    return np.concatenate([
        x_local_masked,
        pair_values,
        pair_mask
    ]).astype(np.float32)


def build_local_pairprobe_dataset(
    pair_dataset,
    Xpair_probe,
    K_pair=6,
    seed=0
):
    """
    Build model input:
        local partial tomography + limited pair probes

    Targets:
        Y_aug = [local channel, pair residual]
    """
    rng = np.random.default_rng(seed)

    X_aug = []
    pair_masks = []

    for i in range(len(pair_dataset["X"])):
        obs_pair = random_pair_probe_indices(K_pair, rng=rng)

        x_aug = make_local_plus_pairprobe_input(
            x_local_masked=pair_dataset["X"][i],
            x_pair_full=Xpair_probe[i],
            observed_pair_indices=obs_pair
        )

        X_aug.append(x_aug)
        pair_masks.append(obs_pair)

    return {
        "X": np.stack(X_aug).astype(np.float32),
        "Y_aug": pair_dataset["Y_aug"].astype(np.float32),
        "Y_channel": pair_dataset["Y_channel"].astype(np.float32),
        "Y_pair": pair_dataset["Y_pair"].astype(np.float32),
        "pair_masks": pair_masks,
    }

# %% Cell 46
# ============================================================
# CELL 113 — QAOA with local + pair residual correction
# ============================================================

if "pauli_string_labels_3q" not in globals():
    raise RuntimeError("pauli_string_labels_3q not found. Run QAOA Pauli-basis definition cells first.")

pauli_label_to_axis = {"X": 1, "Y": 2, "Z": 3}

def pauli_string_index_for_edge_pair(edge, label_i, label_j):
    """
    Return index of Pauli string with given labels on edge and I elsewhere.
    """
    labels = ["I", "I", "I"]
    qi, qj = edge
    labels[qi] = label_i
    labels[qj] = label_j
    target = "".join(labels)
    return pauli_string_labels_3q.index(target)


PAIR_COEFF_INDICES = []

for edge in PAIR_EDGES_3Q:
    for label_i in PAIR_PAULI_LABELS:
        for label_j in PAIR_PAULI_LABELS:
            idx = pauli_string_index_for_edge_pair(edge, label_i, label_j)
            PAIR_COEFF_INDICES.append(idx)

PAIR_COEFF_INDICES = np.array(PAIR_COEFF_INDICES, dtype=int)

print("PAIR_COEFF_INDICES:", PAIR_COEFF_INDICES)


def apply_local_plus_pair_residual_channel_3q(rho, y_channel, y_pair_residual):
    """
    Heuristic structured pair-aware error model.

    1. Apply product local channels.
    2. Add learned pair residual correction to edge two-body Pauli coefficients:
           c_out[P_ab^edge] += delta_ab^edge * c_in[P_ab^edge]
    """
    # Local product channel
    rho_local = apply_product_local_channel_3q(rho, y_channel)

    c_in = rho_to_pauli_coeffs_3q(rho)
    c_out = rho_to_pauli_coeffs_3q(rho_local)

    y_pair_residual = np.asarray(y_pair_residual, dtype=np.float64)

    for k, coeff_idx in enumerate(PAIR_COEFF_INDICES):
        c_out[coeff_idx] += y_pair_residual[k] * c_in[coeff_idx]

    rho_out = pauli_coeffs_to_rho_3q(c_out)

    return rho_out


def qaoa_cost_landscape_from_local_pair_3q(
    y_channel,
    y_pair_residual,
    rho_ideal_grid,
    shots=None,
    seed=0
):
    rng = np.random.default_rng(seed)

    n_gamma = len(gamma_grid_3q)
    n_beta = len(beta_grid_3q)

    C = np.zeros((n_gamma, n_beta), dtype=np.float64)
    Popt = np.zeros((n_gamma, n_beta), dtype=np.float64)

    for i in range(n_gamma):
        for j in range(n_beta):
            rho_ideal = rho_ideal_grid[i][j]

            rho_noisy = apply_local_plus_pair_residual_channel_3q(
                rho_ideal,
                y_channel,
                y_pair_residual
            )

            probs = probs_from_rho_qaoa(rho_noisy)

            if shots is None:
                C[i, j] = maxcut_cost_from_probs(probs)
            else:
                p_arr = np.array([probs[k] for k in qaoa_bitstring_labels], dtype=np.float64)
                p_arr = p_arr / p_arr.sum()

                counts = rng.multinomial(shots, p_arr)

                cost_hat = 0.0
                for count, label in zip(counts, qaoa_bitstring_labels):
                    cost_hat += count * bitstring_costs[label]

                C[i, j] = cost_hat / shots

            Popt[i, j] = p_opt_from_probs(probs)

    return C, Popt


def evaluate_qaoa_pair_device_3q(
    idx_device,
    Y_ch_true_all,
    Y_pair_true_all,
    prediction_methods,
    qaoa_shots=2048,
    seed=0
):
    """
    Pair-correlated QAOA benchmark.

    True noisy QAOA uses true local + true pair residual.
    Each method predicts local + pair residual, or pair residual = 0.
    """
    y_ch_true = Y_ch_true_all[idx_device]
    y_pair_true = Y_pair_true_all[idx_device]

    C_noisy_sampled, _ = qaoa_cost_landscape_from_local_pair_3q(
        y_channel=y_ch_true,
        y_pair_residual=y_pair_true,
        rho_ideal_grid=rho_ideal_grid_3q,
        shots=qaoa_shots,
        seed=seed + idx_device
    )

    results = {}

    results["Noisy"] = {
        "C": C_noisy_sampled,
        "error": landscape_error_metrics_3q(C_noisy_sampled, C_ideal_3q),
        "selected": selected_point_metrics(C_noisy_sampled, C_ideal_3q, Popt_ideal_3q)
    }

    for method_name, pred_pack in prediction_methods.items():
        y_ch_model = pred_pack["Y_channel"][idx_device]

        if "Y_pair" in pred_pack and pred_pack["Y_pair"] is not None:
            y_pair_model = pred_pack["Y_pair"][idx_device]
        else:
            y_pair_model = np.zeros(PAIR_LABEL_DIM, dtype=np.float32)

        C_model, _ = qaoa_cost_landscape_from_local_pair_3q(
            y_channel=y_ch_model,
            y_pair_residual=y_pair_model,
            rho_ideal_grid=rho_ideal_grid_3q,
            shots=None,
            seed=seed + 1000 + idx_device
        )

        C_mit, _ = mitigate_landscape_by_bias_correction(
            C_noisy_sampled,
            C_ideal_3q,
            C_model,
            clip=True
        )

        results[method_name] = {
            "C": C_mit,
            "error": landscape_error_metrics_3q(C_mit, C_ideal_3q),
            "selected": selected_point_metrics(C_mit, C_ideal_3q, Popt_ideal_3q)
        }

    return results

# %% Cell 47
# ============================================================
# PRA STEP 2B — Apply process-relative edge residuals in QAOA
# ============================================================

# The pair label is now a compact edge-process residual block:
#     for each edge: residual_block[out_pair, in_pair], shape 9 x 9.
# In the QAOA benchmark, we add this residual response to the edge two-body
# Pauli coefficients after the product-local channel has acted.

if "pauli_string_labels_3q" not in globals():
    raise RuntimeError("pauli_string_labels_3q not found. Run QAOA Pauli-basis definition cells first.")


def pair_coeff_indices_for_edge(edge):
    indices = []
    for label_i, label_j in PAIR_AXIS_PAIRS:
        labels = ["I", "I", "I"]
        qi, qj = edge
        labels[qi] = label_i
        labels[qj] = label_j
        target = "".join(labels)
        indices.append(pauli_string_labels_3q.index(target))
    return np.array(indices, dtype=int)


PAIR_COEFF_INDICES_BY_EDGE = np.stack([
    pair_coeff_indices_for_edge(edge)
    for edge in PAIR_EDGES_3Q
])

# Flat version kept for compatibility with plotting/debugging code.
PAIR_COEFF_INDICES = PAIR_COEFF_INDICES_BY_EDGE.reshape(-1)

print("PAIR_COEFF_INDICES_BY_EDGE shape:", PAIR_COEFF_INDICES_BY_EDGE.shape)
print("PAIR_LABEL_DIM:", PAIR_LABEL_DIM)


def split_pair_process_residual_blocks(y_pair_residual):
    y_pair_residual = np.asarray(y_pair_residual, dtype=np.float64)
    expected_dim = len(PAIR_EDGES_3Q) * len(PAIR_AXIS_PAIRS) * len(PAIR_AXIS_PAIRS)

    if y_pair_residual.size != expected_dim:
        raise ValueError(
            f"Expected process-relative pair residual dimension {expected_dim}, "
            f"got {y_pair_residual.size}."
        )

    return y_pair_residual.reshape(
        len(PAIR_EDGES_3Q),
        len(PAIR_AXIS_PAIRS),   # output Pauli-pair index
        len(PAIR_AXIS_PAIRS),   # input Pauli-pair index
    )


def apply_local_plus_pair_residual_channel_3q(rho, y_channel, y_pair_residual):
    """
    Product-local response plus process-relative edge residual.

    1. Apply the tensor product of learned local affine maps.
    2. For each hardware edge, add the learned process-relative residual block
       to the corresponding two-body Pauli coefficients:

           c_out_edge += Delta_edge @ c_in_edge

       where Delta_edge[out_pair, in_pair] is learned from the same pair-probe
       input ensemble. This avoids defining the residual as a QAOA-state
       correlation minus a product of QAOA-state local means.
    """
    rho_local = apply_product_local_channel_3q(rho, y_channel)

    c_in = rho_to_pauli_coeffs_3q(rho)
    c_out = rho_to_pauli_coeffs_3q(rho_local)

    residual_blocks = split_pair_process_residual_blocks(y_pair_residual)

    for e_idx in range(len(PAIR_EDGES_3Q)):
        coeff_indices = PAIR_COEFF_INDICES_BY_EDGE[e_idx]
        edge_input_coeffs = c_in[coeff_indices]
        edge_correction = residual_blocks[e_idx] @ edge_input_coeffs
        c_out[coeff_indices] += edge_correction

    rho_out = pauli_coeffs_to_rho_3q(c_out)
    return rho_out

# %% Cell 48
# ============================================================
# CELL 117 — Pair-probe measurement definitions
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import os
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

required_for_step13 = [
    "PAIR_EDGES_3Q",
    "PAIR_PAULI_LABELS",
    "PAIR_LABEL_DIM",
    "LABEL_DIM",
    "FULL_TOMO_DIM",
    "N_QUBITS",
    "build_all_pulse_superoperators",
    "prepare_pair_pauli_probe_state",
    "ket_to_rho",
    "evolve_rho_with_superoperator",
    "exact_pair_expectation_with_readout",
    "make_masked_tomography_input",
]

missing = [name for name in required_for_step13 if name not in globals()]

if missing:
    raise RuntimeError(
        "Missing required objects for Step 13: " + str(missing) +
        "\nRun Step 12 definitions/results first."
    )

print("Step 13 pair-probe measurement definitions requirements are available.")

# ------------------------------------------------------------
# Pair-probe settings
# For each edge (1-2, 2-3), measure all 9 Pauli-pair settings:
# XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ
# Total: 2 edges * 9 = 18 pair probes
# ------------------------------------------------------------

PAIR_PROBE_SETTINGS_3Q = []

for edge in PAIR_EDGES_3Q:
    for li in PAIR_PAULI_LABELS:
        for lj in PAIR_PAULI_LABELS:
            PAIR_PROBE_SETTINGS_3Q.append({
                "edge": edge,
                "label_i": li,
                "label_j": lj
            })

PAIR_PROBE_DIM_3Q = len(PAIR_PROBE_SETTINGS_3Q)

print("PAIR_PROBE_DIM_3Q:", PAIR_PROBE_DIM_3Q)
for k, s in enumerate(PAIR_PROBE_SETTINGS_3Q):
    print(k, s)


def sample_pair_product_measurement(
    rho,
    params,
    qi,
    qj,
    label_i,
    label_j,
    shots=1024,
    rng=None
):
    """
    Finite-shot estimate of the pair-product observable.

    We compute the readout-corrected expectation e = <sigma_i sigma_j>,
    then sample the product variable in ±1.

    This is a compact measurement model for pair-product tomography.
    """
    if rng is None:
        rng = np.random.default_rng()

    e = exact_pair_expectation_with_readout(
        rho=rho,
        params=params,
        qi=qi,
        qj=qj,
        label_i=label_i,
        label_j=label_j
    )

    p_plus = 0.5 * (1.0 + e)
    p_plus = float(np.clip(p_plus, 0.0, 1.0))

    n_plus = rng.binomial(shots, p_plus)
    n_minus = shots - n_plus

    e_hat = (n_plus - n_minus) / shots

    return float(e_hat)


def generate_pair_probe_vector_cached(
    params,
    S_list,
    shots=1024,
    rng=None
):
    """
    Generate finite-shot pair-probe measurement vector.

    For each pair setting:
        prepare pair Pauli +/+ input state
        apply sequential local pulses on the two qubits in the edge
        measure pair-product expectation

    Output shape:
        (18,)
    """
    if rng is None:
        rng = np.random.default_rng()

    values = []

    for setting in PAIR_PROBE_SETTINGS_3Q:
        edge = setting["edge"]
        qi, qj = edge
        li = setting["label_i"]
        lj = setting["label_j"]

        psi0 = prepare_pair_pauli_probe_state(
            edge=edge,
            label_i=li,
            label_j=lj,
            spectator_label="0"
        )

        rho0 = ket_to_rho(psi0)

        # Sequential local pulse probes, same as Step 12 pair-label generation
        rho1 = evolve_rho_with_superoperator(rho0, S_list[qi])
        rho2 = evolve_rho_with_superoperator(rho1, S_list[qj])

        e_hat = sample_pair_product_measurement(
            rho=rho2,
            params=params,
            qi=qi,
            qj=qj,
            label_i=li,
            label_j=lj,
            shots=shots,
            rng=rng
        )

        values.append(e_hat)

    return np.array(values, dtype=np.float32)

# %% [markdown]
# ## Final scaled run
#
# The following cells build the final train/validation/test sets, train local and pair-probe models, and save the baseline/oracle-diagnostic tables. The non-oracle estimator and ablation are run in the final dashboard.

# %% Cell 50
# ============================================================
# CELL 135 — FINAL SCALED RUN CONFIG + LOCAL DATASET
# ============================================================


import os
import time
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

FINAL_DIR = "outputs/three_qubit_qaoa_FINAL_scaled_run"
os.makedirs(FINAL_DIR, exist_ok=True)

# ------------------------------------------------------------
# Final scaled sizes
# ------------------------------------------------------------

FINAL_RUN_CONFIG = {
    # Local random-mask experiment
    "N_LOCAL_TRAIN": 500,
    "N_LOCAL_VAL": 100,
    "N_LOCAL_TEST": 100,
    "LOCAL_K_LIST": [6, 12, 18, 24, 36, 54],
    "LOCAL_EPOCHS": 300,
    "LOCAL_RF_ESTIMATORS": 120,

    # Pair-probe experiment
    "N_PAIR_TRAIN": 400,
    "N_PAIR_VAL": 80,
    "N_PAIR_TEST": 80,
    "PAIR_K_LIST": [0, 6, 12, 18],
    "PAIR_EPOCHS": 350,

    # QAOA evaluation
    "N_QAOA_LOCAL_EVAL": 50,
    "N_QAOA_PAIR_EVAL": 50,

    # Training
    "BATCH_SIZE": 32,
    "LR": 8e-4,
    "WEIGHT_DECAY": 1e-5,
    "LAMBDA_PHYS": 0.03,
    "LAMBDA_PAIR": 1.0,

    # Reproducibility
    "BASE_SEED": 20260522,
}

with open(os.path.join(FINAL_DIR, "final_run_config.json"), "w") as f:
    json.dump(FINAL_RUN_CONFIG, f, indent=2)

print("Final run directory:", FINAL_DIR)
print(json.dumps(FINAL_RUN_CONFIG, indent=2))


# ------------------------------------------------------------
# Required functions/objects check
# ------------------------------------------------------------

required_names_final = [
    "sample_hidden_device_3q",
    "build_all_pulse_superoperators",
    "generate_full_tomography_vector_cached",
    "reference_all_local_channels_label_cached",
    "make_masked_tomography_input",
    "random_mask_indices",
    "direct_masked_channels_for_dataset_3q",
    "FULL_TOMO_DIM",
    "LABEL_DIM",
    "N_QUBITS",
    "evaluate_qaoa_device_3q",
    "qaoa_shots_3q",
]

missing = [name for name in required_names_final if name not in globals()]
if missing:
    raise RuntimeError("Missing required previous definitions: " + str(missing))

print("Core requirements are available.")


# ------------------------------------------------------------
# Common helpers
# ------------------------------------------------------------

def final_save_pickle(obj, filename):
    path = os.path.join(FINAL_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved:", path)

def final_load_pickle(filename):
    path = os.path.join(FINAL_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)

def compute_standardizer(X, eps=1e-8):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return mean.astype(np.float32), std.astype(np.float32)

def apply_standardizer(X, mean, std):
    return ((X - mean) / std).astype(np.float32)

def invert_standardizer(Xs, mean, std):
    return (Xs * std + mean).astype(np.float32)

def mse_np(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))

def per_example_l2_error(Y_est, Y_true):
    return np.linalg.norm(np.asarray(Y_est) - np.asarray(Y_true), axis=1)

def improvement_ratio(noisy_error, corrected_error, eps=1e-12):
    return float(noisy_error / (corrected_error + eps))


# ------------------------------------------------------------
# ChannelNet model and physicality loss
# ------------------------------------------------------------

class FinalChannelNet3Q(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


bloch_test_np_final = np.array([
    [ 1,  0,  0],
    [-1,  0,  0],
    [ 0,  1,  0],
    [ 0, -1,  0],
    [ 0,  0,  1],
    [ 0,  0, -1],
    [ 1,  1,  1],
    [ 1, -1,  1],
    [-1,  1,  1],
    [ 1,  1, -1],
], dtype=np.float32)

for i in range(6, len(bloch_test_np_final)):
    bloch_test_np_final[i] /= np.linalg.norm(bloch_test_np_final[i])

bloch_test_t_final = torch.tensor(bloch_test_np_final, dtype=torch.float32).to(device)


def split_channels_torch_final(y_phys, N=3):
    batch = y_phys.shape[0]
    A_all = []
    b_all = []

    offset = 0
    for q in range(N):
        A_q = y_phys[:, offset:offset+9].reshape(batch, 3, 3)
        offset += 9

        b_q = y_phys[:, offset:offset+3]
        offset += 3

        A_all.append(A_q)
        b_all.append(b_q)

    return torch.stack(A_all, dim=1), torch.stack(b_all, dim=1)


def physicality_loss_final(y_pred_s, Y_mean_t, Y_std_t):
    y_phys = y_pred_s * Y_std_t + Y_mean_t
    A, b = split_channels_torch_final(y_phys, N=N_QUBITS)

    r = bloch_test_t_final[None, None, :, :]
    penalties = []

    for q in range(N_QUBITS):
        A_q = A[:, q, :, :]
        b_q = b[:, q, :]

        out = torch.matmul(r, A_q.transpose(1, 2)[:, None, :, :])
        out = out.squeeze(1) + b_q[:, None, :]

        norms = torch.linalg.norm(out, dim=-1)
        penalties.append((torch.relu(norms - 1.0) ** 2).mean())

    return sum(penalties) / len(penalties)


def train_final_channelnet(
    X_train,
    Y_train,
    X_val,
    Y_val,
    n_epochs=300,
    batch_size=32,
    lr=8e-4,
    weight_decay=1e-5,
    lambda_phys=0.03,
    seed=0
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_mean, X_std = compute_standardizer(X_train)
    Y_mean, Y_std = compute_standardizer(Y_train)

    X_train_s = apply_standardizer(X_train, X_mean, X_std)
    X_val_s = apply_standardizer(X_val, X_mean, X_std)

    Y_train_s = apply_standardizer(Y_train, Y_mean, Y_std)
    Y_val_s = apply_standardizer(Y_val, Y_mean, Y_std)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_s, dtype=torch.float32),
            torch.tensor(Y_train_s, dtype=torch.float32)
        ),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val_s, dtype=torch.float32),
            torch.tensor(Y_val_s, dtype=torch.float32)
        ),
        batch_size=128,
        shuffle=False
    )

    model = FinalChannelNet3Q(
        input_dim=X_train.shape[1],
        output_dim=Y_train.shape[1]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    mse = nn.MSELoss()

    Y_mean_t = torch.tensor(Y_mean, dtype=torch.float32).to(device)
    Y_std_t = torch.tensor(Y_std, dtype=torch.float32).to(device)

    history = {
        "train_mse": [],
        "val_mse": [],
        "val_phys": [],
    }

    best_val = np.inf
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_total = 0.0
        n_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)
            mse_loss = mse(pred, yb)
            phys_loss = physicality_loss_final(pred, Y_mean_t, Y_std_t)

            loss = mse_loss + lambda_phys * phys_loss
            loss.backward()
            optimizer.step()

            bs = xb.shape[0]
            train_total += mse_loss.item() * bs
            n_total += bs

        train_mse = train_total / n_total

        model.eval()
        val_total = 0.0
        val_phys_total = 0.0
        val_n = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                mse_loss = mse(pred, yb)
                phys_loss = physicality_loss_final(pred, Y_mean_t, Y_std_t)

                bs = xb.shape[0]
                val_total += mse_loss.item() * bs
                val_phys_total += phys_loss.item() * bs
                val_n += bs

        val_mse = val_total / val_n
        val_phys = val_phys_total / val_n

        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["val_phys"].append(val_phys)

        if val_mse < best_val:
            best_val = val_mse
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"train MSE {train_mse:.4e} | "
                f"val MSE {val_mse:.4e} | "
                f"val phys {val_phys:.4e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "history": history,
        "X_mean": X_mean,
        "X_std": X_std,
        "Y_mean": Y_mean,
        "Y_std": Y_std,
        "best_val_mse": best_val,
    }


def predict_final_channelnet(pack, X):
    X_s = apply_standardizer(X, pack["X_mean"], pack["X_std"])
    X_t = torch.tensor(X_s, dtype=torch.float32).to(device)

    pack["model"].eval()
    with torch.no_grad():
        pred_s = pack["model"](X_t).cpu().numpy()

    return invert_standardizer(pred_s, pack["Y_mean"], pack["Y_std"]).astype(np.float32)


# ------------------------------------------------------------
# Generate local full-tomography dataset
# ------------------------------------------------------------

def generate_one_final_local_example_3q(
    rng,
    topology="chain",
    shot_choices=np.array([128, 512, 1024, 4096], dtype=int),
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0
):
    shots = int(rng.choice(shot_choices))

    params = sample_hidden_device_3q(
        rng=rng,
        topology=topology
    )

    S_list = build_all_pulse_superoperators(
        params=params,
        theta=pulse_theta,
        phase=pulse_phase,
        T=pulse_T,
        include_decoherence=True
    )

    x_full = generate_full_tomography_vector_cached(
        params=params,
        S_list=S_list,
        shots=shots,
        rng=rng
    )

    y = reference_all_local_channels_label_cached(
        params=params,
        S_list=S_list,
        include_readout=True
    )

    return x_full.astype(np.float32), y.astype(np.float32), params, shots


def build_final_local_full_dataset_3q(
    n_examples,
    seed=0,
    print_every=25
):
    rng = np.random.default_rng(seed)

    Xfull = []
    Y = []
    params_list = []
    shots_list = []

    t0 = time.time()

    for i in range(n_examples):
        x_full, y, params, shots = generate_one_final_local_example_3q(
            rng=rng,
            topology="chain"
        )

        Xfull.append(x_full)
        Y.append(y)
        params_list.append(params)
        shots_list.append(shots)

        if (i + 1) % print_every == 0 or (i + 1) == n_examples:
            print(
                f"Local full dataset {i+1:4d}/{n_examples} "
                f"| elapsed {time.time()-t0:.1f} s"
            )

    return {
        "Xfull": np.stack(Xfull).astype(np.float32),
        "Y": np.stack(Y).astype(np.float32),
        "params": params_list,
        "shots": np.array(shots_list, dtype=int),
    }


LOCAL_DATA_PATH = os.path.join(FINAL_DIR, "final_local_full_data.pkl")

if os.path.exists(LOCAL_DATA_PATH):
    print("Loading existing final local data...")
    final_local_data = final_load_pickle("final_local_full_data.pkl")
else:
    print("Generating final local train/val/test data...")

    final_local_data = {
        "train": build_final_local_full_dataset_3q(
            FINAL_RUN_CONFIG["N_LOCAL_TRAIN"],
            seed=FINAL_RUN_CONFIG["BASE_SEED"] + 1,
            print_every=25
        ),
        "val": build_final_local_full_dataset_3q(
            FINAL_RUN_CONFIG["N_LOCAL_VAL"],
            seed=FINAL_RUN_CONFIG["BASE_SEED"] + 2,
            print_every=25
        ),
        "test": build_final_local_full_dataset_3q(
            FINAL_RUN_CONFIG["N_LOCAL_TEST"],
            seed=FINAL_RUN_CONFIG["BASE_SEED"] + 3,
            print_every=25
        ),
    }

    final_save_pickle(final_local_data, "final_local_full_data.pkl")

print("Local train:", final_local_data["train"]["Xfull"].shape, final_local_data["train"]["Y"].shape)
print("Local val:", final_local_data["val"]["Xfull"].shape, final_local_data["val"]["Y"].shape)
print("Local test:", final_local_data["test"]["Xfull"].shape, final_local_data["test"]["Y"].shape)

# %% Cell 51
# ============================================================
# CELL 136 — FINAL LOCAL RANDOM-MASK SWEEP
# ============================================================

def build_masked_from_final_full(
    Xfull,
    Y,
    shots,
    K,
    seed=0,
    mask_mode="random"
):
    rng = np.random.default_rng(seed)

    X = []
    masks = []

    for i in range(len(Y)):
        if mask_mode == "complete" or K == FULL_TOMO_DIM:
            obs = np.arange(FULL_TOMO_DIM, dtype=int)
        elif mask_mode == "random":
            obs = random_mask_indices(K=K, rng=rng)
        else:
            raise ValueError("mask_mode must be random or complete.")

        x = make_masked_tomography_input(
            x_full_tomo=Xfull[i],
            observed_indices=obs,
            shots=int(shots[i])
        )

        X.append(x)
        masks.append(obs)

    return {
        "X": np.stack(X).astype(np.float32),
        "Y": Y.astype(np.float32),
        "masks": masks,
    }


def train_eval_final_local_K(K):
    print("\n" + "=" * 100)
    print(f"FINAL LOCAL K SWEEP | K = {K}")
    print("=" * 100)

    train_ds = build_masked_from_final_full(
        Xfull=final_local_data["train"]["Xfull"],
        Y=final_local_data["train"]["Y"],
        shots=final_local_data["train"]["shots"],
        K=K,
        seed=FINAL_RUN_CONFIG["BASE_SEED"] + 100 + K
    )

    val_ds = build_masked_from_final_full(
        Xfull=final_local_data["val"]["Xfull"],
        Y=final_local_data["val"]["Y"],
        shots=final_local_data["val"]["shots"],
        K=K,
        seed=FINAL_RUN_CONFIG["BASE_SEED"] + 200 + K
    )

    test_ds = build_masked_from_final_full(
        Xfull=final_local_data["test"]["Xfull"],
        Y=final_local_data["test"]["Y"],
        shots=final_local_data["test"]["shots"],
        K=K,
        seed=FINAL_RUN_CONFIG["BASE_SEED"] + 300 + K
    )

    # Mean baseline
    Y_mean_train = train_ds["Y"].mean(axis=0)
    Y_pred_mean = np.repeat(Y_mean_train[None, :], len(test_ds["Y"]), axis=0).astype(np.float32)

    # Direct masked tomography baseline
    Y_pred_direct = direct_masked_channels_for_dataset_3q(
        test_ds["X"],
        Y_mean_train
    )

    # Ridge baseline
    X_mean, X_std = compute_standardizer(train_ds["X"])
    X_train_s = apply_standardizer(train_ds["X"], X_mean, X_std)
    X_test_s = apply_standardizer(test_ds["X"], X_mean, X_std)

    ridge = Ridge(alpha=1e-2)
    ridge.fit(X_train_s, train_ds["Y"])
    Y_pred_ridge = ridge.predict(X_test_s).astype(np.float32)

    # Random forest baseline
    rf = RandomForestRegressor(
        n_estimators=FINAL_RUN_CONFIG["LOCAL_RF_ESTIMATORS"],
        min_samples_leaf=2,
        random_state=FINAL_RUN_CONFIG["BASE_SEED"] + 400 + K,
        n_jobs=-1
    )
    rf.fit(train_ds["X"], train_ds["Y"])
    Y_pred_rf = rf.predict(test_ds["X"]).astype(np.float32)

    # ChannelNet
    net_pack = train_final_channelnet(
        X_train=train_ds["X"],
        Y_train=train_ds["Y"],
        X_val=val_ds["X"],
        Y_val=val_ds["Y"],
        n_epochs=FINAL_RUN_CONFIG["LOCAL_EPOCHS"],
        batch_size=FINAL_RUN_CONFIG["BATCH_SIZE"],
        lr=FINAL_RUN_CONFIG["LR"],
        weight_decay=FINAL_RUN_CONFIG["WEIGHT_DECAY"],
        lambda_phys=FINAL_RUN_CONFIG["LAMBDA_PHYS"],
        seed=FINAL_RUN_CONFIG["BASE_SEED"] + 500 + K
    )

    Y_pred_net = predict_final_channelnet(
        net_pack,
        test_ds["X"]
    )

    predictions = {
        "Mean": Y_pred_mean,
        "Direct": Y_pred_direct,
        "Ridge": Y_pred_ridge,
        "RandomForest": Y_pred_rf,
        "ChannelNet": Y_pred_net,
    }

    channel_rows = []

    for method, Y_pred in predictions.items():
        errs = per_example_l2_error(Y_pred, test_ds["Y"])

        channel_rows.append({
            "K": K,
            "Method": method,
            "Channel MSE": mse_np(Y_pred, test_ds["Y"]),
            "Mean L2": float(errs.mean()),
            "Median L2": float(np.median(errs)),
            "Std L2": float(errs.std()),
        })

    result = {
        "K": K,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "ridge": ridge,
        "rf": rf,
        "net_pack": net_pack,
        "predictions": predictions,
        "channel_rows": channel_rows,
    }

    print("\nChannel summary K =", K)
    display(pd.DataFrame(channel_rows))

    return result


LOCAL_SWEEP_PATH = os.path.join(FINAL_DIR, "final_local_sweep_results.pkl")

if os.path.exists(LOCAL_SWEEP_PATH):
    print("Loading existing final local sweep results...")
    final_local_sweep_results = final_load_pickle("final_local_sweep_results.pkl")
else:
    final_local_sweep_results = {}

    for K in FINAL_RUN_CONFIG["LOCAL_K_LIST"]:
        res = train_eval_final_local_K(K)
        final_local_sweep_results[K] = res

        # Save after every K to survive Colab disconnect
        final_save_pickle(final_local_sweep_results, "final_local_sweep_results.pkl")

print("Finished/loaded final local sweep.")


# ------------------------------------------------------------
# Local channel summary plots
# ------------------------------------------------------------

final_local_channel_rows = []
for K, res in final_local_sweep_results.items():
    final_local_channel_rows.extend(res["channel_rows"])

final_local_channel_table = pd.DataFrame(final_local_channel_rows)
final_local_channel_table.to_csv(
    os.path.join(FINAL_DIR, "final_local_channel_table.csv"),
    index=False
)

display(final_local_channel_table)

plt.figure(figsize=(8, 5))
for method in final_local_channel_table["Method"].unique():
    df_m = final_local_channel_table[final_local_channel_table["Method"] == method].sort_values("K")
    plt.plot(df_m["K"], df_m["Mean L2"], marker="o", label=method)

plt.xlabel("Number of observed local tomography settings K")
plt.ylabel("Mean channel-label L2 error")
plt.title("Final scaled local error-process learning")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FINAL_DIR, "fig_final_local_channel_L2_vs_K.png"), dpi=250)
plt.show()

plt.figure(figsize=(8, 5))
for method in final_local_channel_table["Method"].unique():
    df_m = final_local_channel_table[final_local_channel_table["Method"] == method].sort_values("K")
    plt.plot(df_m["K"], df_m["Channel MSE"], marker="o", label=method)

plt.yscale("log")
plt.xlabel("Number of observed local tomography settings K")
plt.ylabel("Channel-label MSE")
plt.title("Final scaled local channel MSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FINAL_DIR, "fig_final_local_channel_MSE_vs_K.png"), dpi=250)
plt.show()


# ------------------------------------------------------------
# Local QAOA evaluation for K=18 and K=54 only
# Keeps runtime reasonable.
# ------------------------------------------------------------

def summarize_qaoa_results_list(results):
    methods = list(results[0].keys())
    rows = []

    noisy_mae = np.array([r["Noisy"]["error"]["MAE"] for r in results]).mean()

    for method in methods:
        mae_arr = np.array([r[method]["error"]["MAE"] for r in results])
        rmse_arr = np.array([r[method]["error"]["RMSE"] for r in results])
        regret_arr = np.array([r[method]["selected"]["regret"] for r in results])
        approx_arr = np.array([r[method]["selected"]["approximation_ratio"] for r in results])
        popt_arr = np.array([r[method]["selected"]["p_opt_selected"] for r in results])
        disp_arr = np.array([r[method]["selected"]["parameter_displacement"] for r in results])

        rows.append({
            "Method": method,
            "Mean MAE": float(mae_arr.mean()),
            "Median MAE": float(np.median(mae_arr)),
            "Mean RMSE": float(rmse_arr.mean()),
            "Mean regret": float(regret_arr.mean()),
            "Mean approximation ratio": float(approx_arr.mean()),
            "Mean P_opt": float(popt_arr.mean()),
            "Mean parameter displacement": float(disp_arr.mean()),
            "Improvement ratio": 1.0 if method == "Noisy" else improvement_ratio(noisy_mae, mae_arr.mean()),
        })

    return pd.DataFrame(rows)


def evaluate_final_local_qaoa_for_K(K, n_devices):
    res_K = final_local_sweep_results[K]
    Y_true = res_K["test_ds"]["Y"]

    pred_methods = {
        "Direct": res_K["predictions"]["Direct"],
        "Ridge": res_K["predictions"]["Ridge"],
        "RandomForest": res_K["predictions"]["RandomForest"],
        "ChannelNet": res_K["predictions"]["ChannelNet"],
    }

    n_eval = min(n_devices, len(Y_true))
    results = []

    for k in range(n_eval):
        r = evaluate_qaoa_device_3q(
            idx_device=k,
            Y_true_all=Y_true,
            prediction_methods=pred_methods,
            qaoa_shots=qaoa_shots_3q,
            seed=FINAL_RUN_CONFIG["BASE_SEED"] + 6000 + 10*K
        )

        results.append(r)

        if (k + 1) % 10 == 0 or (k + 1) == n_eval:
            print(f"K={K} QAOA {k+1}/{n_eval}")

    table = summarize_qaoa_results_list(results)
    table.insert(0, "K", K)

    return results, table


LOCAL_QAOA_PATH = os.path.join(FINAL_DIR, "final_local_qaoa_results.pkl")

if os.path.exists(LOCAL_QAOA_PATH):
    print("Loading existing local QAOA results...")
    local_qaoa_pack = final_load_pickle("final_local_qaoa_results.pkl")
else:
    local_qaoa_pack = {}

    for K in [18, 54]:
        results, table = evaluate_final_local_qaoa_for_K(
            K=K,
            n_devices=FINAL_RUN_CONFIG["N_QAOA_LOCAL_EVAL"]
        )

        local_qaoa_pack[K] = {
            "results": results,
            "table": table
        }

        final_save_pickle(local_qaoa_pack, "final_local_qaoa_results.pkl")

final_local_qaoa_table = pd.concat(
    [pack["table"] for pack in local_qaoa_pack.values()],
    ignore_index=True
)

final_local_qaoa_table.to_csv(
    os.path.join(FINAL_DIR, "final_local_qaoa_table.csv"),
    index=False
)

display(final_local_qaoa_table)

# %% Cell 52
# ============================================================
# CELL 137 — FINAL SCALED PAIR-PROBE EXPERIMENT
# ============================================================

required_pair_names = [
    "build_pair_augmented_dataset_3q",
    "build_pair_probe_matrix_from_dataset",
    "build_local_pairprobe_dataset",
    "PairProbeAwareNet3Q",
    "split_channels_torch_pair",
    "bloch_test_t_pair",
    "split_channel_pair",
    "evaluate_qaoa_pair_device_3q",
    "PAIR_LABEL_DIM",
    "PAIR_PROBE_DIM_3Q",
    "PAIR_EDGES_3Q",
    "PAIR_PAULI_LABELS",
]

missing_pair = [name for name in required_pair_names if name not in globals()]
if missing_pair:
    raise RuntimeError("Missing Step 12/13 pair definitions: " + str(missing_pair))

print("Pair-probe requirements are available.")


PAIR_DATA_PATH = os.path.join(FINAL_DIR, "final_pair_data_and_probes_process_relative.pkl")

if os.path.exists(PAIR_DATA_PATH):
    print("Loading existing final pair data/probes...")
    final_pair_pack = final_load_pickle("final_pair_data_and_probes_process_relative.pkl")

    final_pair_train = final_pair_pack["train"]
    final_pair_val = final_pair_pack["val"]
    final_pair_test = final_pair_pack["test"]

    final_Xpair_train = final_pair_pack["Xpair_train"]
    final_Xpair_val = final_pair_pack["Xpair_val"]
    final_Xpair_test = final_pair_pack["Xpair_test"]

else:
    print("Generating final pair-aware train/val/test datasets...")

    final_pair_train = build_pair_augmented_dataset_3q(
        n_examples=FINAL_RUN_CONFIG["N_PAIR_TRAIN"],
        seed=FINAL_RUN_CONFIG["BASE_SEED"] + 7001,
        K=18,
        mask_mode="random",
        pair_enriched=True,
        print_every=20
    )

    final_pair_val = build_pair_augmented_dataset_3q(
        n_examples=FINAL_RUN_CONFIG["N_PAIR_VAL"],
        seed=FINAL_RUN_CONFIG["BASE_SEED"] + 7002,
        K=18,
        mask_mode="random",
        pair_enriched=True,
        print_every=20
    )

    final_pair_test = build_pair_augmented_dataset_3q(
        n_examples=FINAL_RUN_CONFIG["N_PAIR_TEST"],
        seed=FINAL_RUN_CONFIG["BASE_SEED"] + 7003,
        K=18,
        mask_mode="random",
        pair_enriched=True,
        print_every=20
    )

    print("Generating pair-probe matrices...")

    final_Xpair_train = build_pair_probe_matrix_from_dataset(
        final_pair_train,
        seed=FINAL_RUN_CONFIG["BASE_SEED"] + 7101,
        print_every=20
    )

    final_Xpair_val = build_pair_probe_matrix_from_dataset(
        final_pair_val,
        seed=FINAL_RUN_CONFIG["BASE_SEED"] + 7102,
        print_every=20
    )

    final_Xpair_test = build_pair_probe_matrix_from_dataset(
        final_pair_test,
        seed=FINAL_RUN_CONFIG["BASE_SEED"] + 7103,
        print_every=20
    )

    final_pair_pack = {
        "train": final_pair_train,
        "val": final_pair_val,
        "test": final_pair_test,
        "Xpair_train": final_Xpair_train,
        "Xpair_val": final_Xpair_val,
        "Xpair_test": final_Xpair_test,
    }

    final_save_pickle(final_pair_pack, "final_pair_data_and_probes_process_relative.pkl")

print("Final pair train X:", final_pair_train["X"].shape)
print("Final pair train Y_aug:", final_pair_train["Y_aug"].shape)
print("Final pair probes:", final_Xpair_train.shape)


# ------------------------------------------------------------
# Train pair-probe model for one K_pair
# ------------------------------------------------------------

def train_final_pairprobe_K(K_pair):
    print("\n" + "=" * 100)
    print(f"FINAL PAIR-PROBE TRAINING | K_pair = {K_pair}")
    print("=" * 100)

    ds_train = build_local_pairprobe_dataset(
        pair_dataset=final_pair_train,
        Xpair_probe=final_Xpair_train,
        K_pair=K_pair,
        seed=FINAL_RUN_CONFIG["BASE_SEED"] + 8000 + K_pair
    )

    ds_val = build_local_pairprobe_dataset(
        pair_dataset=final_pair_val,
        Xpair_probe=final_Xpair_val,
        K_pair=K_pair,
        seed=FINAL_RUN_CONFIG["BASE_SEED"] + 8100 + K_pair
    )

    ds_test = build_local_pairprobe_dataset(
        pair_dataset=final_pair_test,
        Xpair_probe=final_Xpair_test,
        K_pair=K_pair,
        seed=FINAL_RUN_CONFIG["BASE_SEED"] + 8200 + K_pair
    )

    X_mean, X_std = compute_standardizer(ds_train["X"])
    Y_mean, Y_std = compute_standardizer(ds_train["Y_aug"])
    Y_ch_mean, Y_ch_std = compute_standardizer(ds_train["Y_channel"])

    X_train_s = apply_standardizer(ds_train["X"], X_mean, X_std)
    X_val_s = apply_standardizer(ds_val["X"], X_mean, X_std)
    X_test_s = apply_standardizer(ds_test["X"], X_mean, X_std)

    Y_train_s = apply_standardizer(ds_train["Y_aug"], Y_mean, Y_std)
    Y_val_s = apply_standardizer(ds_val["Y_aug"], Y_mean, Y_std)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_s, dtype=torch.float32),
            torch.tensor(Y_train_s, dtype=torch.float32)
        ),
        batch_size=FINAL_RUN_CONFIG["BATCH_SIZE"],
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val_s, dtype=torch.float32),
            torch.tensor(Y_val_s, dtype=torch.float32)
        ),
        batch_size=128,
        shuffle=False
    )

    model = PairProbeAwareNet3Q(
        input_dim=ds_train["X"].shape[1],
        channel_dim=LABEL_DIM,
        pair_dim=PAIR_LABEL_DIM
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FINAL_RUN_CONFIG["LR"],
        weight_decay=FINAL_RUN_CONFIG["WEIGHT_DECAY"]
    )

    mse = nn.MSELoss()

    Y_ch_mean_t_local = torch.tensor(Y_ch_mean, dtype=torch.float32).to(device)
    Y_ch_std_t_local = torch.tensor(Y_ch_std, dtype=torch.float32).to(device)

    def physicality_loss_pair_local(y_channel_s):
        y_phys = y_channel_s * Y_ch_std_t_local + Y_ch_mean_t_local
        A, b = split_channels_torch_pair(y_phys)

        r = bloch_test_t_pair[None, None, :, :]
        penalties = []

        for q in range(N_QUBITS):
            A_q = A[:, q, :, :]
            b_q = b[:, q, :]

            out = torch.matmul(r, A_q.transpose(1, 2)[:, None, :, :])
            out = out.squeeze(1) + b_q[:, None, :]

            norms = torch.linalg.norm(out, dim=-1)
            penalties.append((torch.relu(norms - 1.0) ** 2).mean())

        return sum(penalties) / len(penalties)

    history = {
        "train_channel_mse": [],
        "train_pair_mse": [],
        "val_channel_mse": [],
        "val_pair_mse": [],
        "val_phys": [],
    }

    best_score = np.inf
    best_state = None

    for epoch in range(1, FINAL_RUN_CONFIG["PAIR_EPOCHS"] + 1):
        model.train()

        train_ch_total = 0.0
        train_pair_total = 0.0
        n_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)

            pred_ch = pred[:, :LABEL_DIM]
            pred_pair = pred[:, LABEL_DIM:LABEL_DIM+PAIR_LABEL_DIM]

            y_ch = yb[:, :LABEL_DIM]
            y_pair = yb[:, LABEL_DIM:LABEL_DIM+PAIR_LABEL_DIM]

            ch_loss = mse(pred_ch, y_ch)
            pair_loss = mse(pred_pair, y_pair)
            phys_loss = physicality_loss_pair_local(pred_ch)

            loss = (
                ch_loss
                + FINAL_RUN_CONFIG["LAMBDA_PAIR"] * pair_loss
                + FINAL_RUN_CONFIG["LAMBDA_PHYS"] * phys_loss
            )

            loss.backward()
            optimizer.step()

            bs = xb.shape[0]
            train_ch_total += ch_loss.item() * bs
            train_pair_total += pair_loss.item() * bs
            n_total += bs

        model.eval()

        val_ch_total = 0.0
        val_pair_total = 0.0
        val_phys_total = 0.0
        val_n = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)

                pred_ch = pred[:, :LABEL_DIM]
                pred_pair = pred[:, LABEL_DIM:LABEL_DIM+PAIR_LABEL_DIM]

                y_ch = yb[:, :LABEL_DIM]
                y_pair = yb[:, LABEL_DIM:LABEL_DIM+PAIR_LABEL_DIM]

                ch_loss = mse(pred_ch, y_ch)
                pair_loss = mse(pred_pair, y_pair)
                phys_loss = physicality_loss_pair_local(pred_ch)

                bs = xb.shape[0]
                val_ch_total += ch_loss.item() * bs
                val_pair_total += pair_loss.item() * bs
                val_phys_total += phys_loss.item() * bs
                val_n += bs

        val_ch = val_ch_total / val_n
        val_pair = val_pair_total / val_n
        val_phys = val_phys_total / val_n

        history["train_channel_mse"].append(train_ch_total / n_total)
        history["train_pair_mse"].append(train_pair_total / n_total)
        history["val_channel_mse"].append(val_ch)
        history["val_pair_mse"].append(val_pair)
        history["val_phys"].append(val_phys)

        score = val_ch + FINAL_RUN_CONFIG["LAMBDA_PAIR"] * val_pair

        if score < best_score:
            best_score = score
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"val ch {val_ch:.4e} | "
                f"val pair {val_pair:.4e} | "
                f"val phys {val_phys:.4e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_s = model(X_test_t).cpu().numpy()

    pred_aug = invert_standardizer(pred_s, Y_mean, Y_std)
    pred_ch, pred_pair = split_channel_pair(pred_aug)

    return {
        "K_pair": K_pair,
        "model": model,
        "history": history,
        "ds_train": ds_train,
        "ds_val": ds_val,
        "ds_test": ds_test,
        "X_mean": X_mean,
        "X_std": X_std,
        "Y_mean": Y_mean,
        "Y_std": Y_std,
        "Y_ch_mean": Y_ch_mean,
        "Y_ch_std": Y_ch_std,
        "Y_pred_aug": pred_aug.astype(np.float32),
        "Y_pred_channel": pred_ch.astype(np.float32),
        "Y_pred_pair": pred_pair.astype(np.float32),
        "best_score": best_score,
    }


PAIR_RESULTS_PATH = os.path.join(FINAL_DIR, "final_pairprobe_results_process_relative.pkl")

if os.path.exists(PAIR_RESULTS_PATH):
    print("Loading existing final pair-probe results...")
    final_pairprobe_results = final_load_pickle("final_pairprobe_results_process_relative.pkl")
else:
    final_pairprobe_results = {}

    for Kp in FINAL_RUN_CONFIG["PAIR_K_LIST"]:
        res = train_final_pairprobe_K(Kp)
        final_pairprobe_results[Kp] = res

        final_save_pickle(final_pairprobe_results, "final_pairprobe_results_process_relative.pkl")

print("Finished/loaded final pair-probe models.")


# ------------------------------------------------------------
# Pair-probe channel/pair evaluation
# ------------------------------------------------------------

Y_ch_true_final_pair = final_pair_test["Y_channel"]
Y_pair_true_final_pair = final_pair_test["Y_pair"]

final_pairprobe_eval_rows = []

for Kp, res in final_pairprobe_results.items():
    Y_ch_pred = res["Y_pred_channel"]
    Y_pair_pred = res["Y_pred_pair"]

    ch_err = per_example_l2_error(Y_ch_pred, Y_ch_true_final_pair)
    pair_err = per_example_l2_error(Y_pair_pred, Y_pair_true_final_pair)

    final_pairprobe_eval_rows.append({
        "K_pair": Kp,
        "Channel MSE": mse_np(Y_ch_pred, Y_ch_true_final_pair),
        "Mean channel L2": float(ch_err.mean()),
        "Median channel L2": float(np.median(ch_err)),
        "Pair MSE": mse_np(Y_pair_pred, Y_pair_true_final_pair),
        "Mean pair L2": float(pair_err.mean()),
        "Median pair L2": float(np.median(pair_err)),
        "Mean abs pair error": float(np.mean(np.abs(Y_pair_pred - Y_pair_true_final_pair))),
    })

final_pairprobe_eval_table = pd.DataFrame(final_pairprobe_eval_rows).sort_values("K_pair")
final_pairprobe_eval_table.to_csv(
    os.path.join(FINAL_DIR, "final_pairprobe_eval_table_process_relative.csv"),
    index=False
)

display(final_pairprobe_eval_table)

plt.figure(figsize=(7, 4))
plt.plot(
    final_pairprobe_eval_table["K_pair"],
    final_pairprobe_eval_table["Mean channel L2"],
    marker="o",
    label="Local channel L2"
)
plt.xlabel("Number of pair probes")
plt.ylabel("Mean local-channel L2 error")
plt.title("Final scaled: local-channel learning vs pair probes")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FINAL_DIR, "fig_final_pairprobe_process_relative_channel_L2.png"), dpi=250)
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(
    final_pairprobe_eval_table["K_pair"],
    final_pairprobe_eval_table["Mean pair L2"],
    marker="o",
    label="Pair residual L2"
)
plt.xlabel("Number of pair probes")
plt.ylabel("Mean pair-residual L2 error")
plt.title("Final scaled: pairwise-error learning vs pair probes")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FINAL_DIR, "fig_final_pairprobe_process_relative_pair_L2.png"), dpi=250)
plt.show()

# %% Cell 53
# ============================================================
# CELL 138 — FINAL PAIR-PROBE QAOA + SUMMARY
# ============================================================

PAIR_QAOA_PATH = os.path.join(FINAL_DIR, "final_pairprobe_qaoa_results_process_relative.pkl")

if os.path.exists(PAIR_QAOA_PATH):
    print("Loading existing final pair-probe QAOA results...")
    final_pair_qaoa_pack = final_load_pickle("final_pairprobe_qaoa_results_process_relative.pkl")
else:
    final_pair_qaoa_prediction_methods = {}

    for Kp, res in final_pairprobe_results.items():
        final_pair_qaoa_prediction_methods[f"Kpair={Kp} local+pair"] = {
            "Y_channel": res["Y_pred_channel"],
            "Y_pair": res["Y_pred_pair"],
        }

    final_pair_qaoa_prediction_methods["Oracle true local+pair"] = {
        "Y_channel": Y_ch_true_final_pair,
        "Y_pair": Y_pair_true_final_pair,
    }

    n_eval = min(FINAL_RUN_CONFIG["N_QAOA_PAIR_EVAL"], len(Y_ch_true_final_pair))

    final_pair_qaoa_results = []

    for k in range(n_eval):
        res = evaluate_qaoa_pair_device_3q(
            idx_device=k,
            Y_ch_true_all=Y_ch_true_final_pair,
            Y_pair_true_all=Y_pair_true_final_pair,
            prediction_methods=final_pair_qaoa_prediction_methods,
            qaoa_shots=qaoa_shots_3q,
            seed=FINAL_RUN_CONFIG["BASE_SEED"] + 9000
        )

        final_pair_qaoa_results.append(res)

        if (k + 1) % 10 == 0 or (k + 1) == n_eval:
            print(f"Final pair QAOA {k+1}/{n_eval}")

    final_pair_qaoa_table = summarize_qaoa_results_list(final_pair_qaoa_results)

    final_pair_qaoa_pack = {
        "results": final_pair_qaoa_results,
        "table": final_pair_qaoa_table,
    }

    final_save_pickle(final_pair_qaoa_pack, "final_pairprobe_qaoa_results_process_relative.pkl")

final_pair_qaoa_table = final_pair_qaoa_pack["table"]

final_pair_qaoa_table.to_csv(
    os.path.join(FINAL_DIR, "final_pairprobe_qaoa_table_process_relative.csv"),
    index=False
)

display(final_pair_qaoa_table)


# ------------------------------------------------------------
# Extract Kpair-only QAOA table
# ------------------------------------------------------------

pair_qaoa_byK_rows = []

for Kp in FINAL_RUN_CONFIG["PAIR_K_LIST"]:
    method_name = f"Kpair={Kp} local+pair"
    row = final_pair_qaoa_table[final_pair_qaoa_table["Method"] == method_name].iloc[0]

    pair_qaoa_byK_rows.append({
        "K_pair": Kp,
        "Mean MAE": row["Mean MAE"],
        "Median MAE": row["Median MAE"],
        "Mean RMSE": row["Mean RMSE"],
        "Improvement ratio": row["Improvement ratio"],
        "Mean approximation ratio": row["Mean approximation ratio"],
        "Mean P_opt": row["Mean P_opt"],
    })

final_pair_qaoa_byK = pd.DataFrame(pair_qaoa_byK_rows)
final_pair_qaoa_byK.to_csv(
    os.path.join(FINAL_DIR, "final_pair_qaoa_byK_process_relative.csv"),
    index=False
)

display(final_pair_qaoa_byK)


plt.figure(figsize=(7, 4))
plt.plot(
    final_pair_qaoa_byK["K_pair"],
    final_pair_qaoa_byK["Mean MAE"],
    marker="o"
)
plt.xlabel("Number of pair probes")
plt.ylabel("Mean QAOA MAE")
plt.title("Final scaled: QAOA reliability vs pair probes")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FINAL_DIR, "fig_final_pairprobe_process_relative_qaoa_MAE.png"), dpi=250)
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(
    final_pair_qaoa_byK["K_pair"],
    final_pair_qaoa_byK["Improvement ratio"],
    marker="o"
)
plt.xlabel("Number of pair probes")
plt.ylabel("Improvement ratio over noisy QAOA")
plt.title("Final scaled: QAOA improvement vs pair probes")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FINAL_DIR, "fig_final_pairprobe_process_relative_qaoa_improvement.png"), dpi=250)
plt.show()


# ------------------------------------------------------------
# Final automatic interpretation
# ------------------------------------------------------------

print("=" * 100)
print("FINAL SCALED RUN SUMMARY")
print("=" * 100)

print("\n1. Local channel-learning summary:")
display(final_local_channel_table)

print("\n2. Local QAOA summary:")
display(final_local_qaoa_table)

print("\n3. Pair-probe channel/pair learning summary:")
display(final_pairprobe_eval_table)

print("\n4. Pair-probe QAOA summary:")
display(final_pair_qaoa_table)

print("\n5. Pair-probe QAOA by K:")
display(final_pair_qaoa_byK)

# Best local ChannelNet result
df_local_cn = final_local_channel_table[
    final_local_channel_table["Method"] == "ChannelNet"
].sort_values("Mean L2")

best_local_cn = df_local_cn.iloc[0]

print("\nBest local ChannelNet by channel L2:")
print(best_local_cn)

# Pair probe improvement
k0_pair = final_pairprobe_eval_table[final_pairprobe_eval_table["K_pair"] == 0].iloc[0]
kmax = max(FINAL_RUN_CONFIG["PAIR_K_LIST"])
kmax_pair = final_pairprobe_eval_table[final_pairprobe_eval_table["K_pair"] == kmax].iloc[0]

print("\nPair-probe learning improvement:")
print("K_pair=0 pair L2:", k0_pair["Mean pair L2"])
print(f"K_pair={kmax} pair L2:", kmax_pair["Mean pair L2"])
print("K_pair=0 channel L2:", k0_pair["Mean channel L2"])
print(f"K_pair={kmax} channel L2:", kmax_pair["Mean channel L2"])

k0_qaoa = final_pair_qaoa_byK[final_pair_qaoa_byK["K_pair"] == 0].iloc[0]
kmax_qaoa = final_pair_qaoa_byK[final_pair_qaoa_byK["K_pair"] == kmax].iloc[0]

print("\nPair-probe QAOA improvement:")
print("K_pair=0 QAOA MAE:", k0_qaoa["Mean MAE"])
print(f"K_pair={kmax} QAOA MAE:", kmax_qaoa["Mean MAE"])
print("K_pair=0 improvement:", k0_qaoa["Improvement ratio"])
print(f"K_pair={kmax} improvement:", kmax_qaoa["Improvement ratio"])

if kmax_qaoa["Mean MAE"] < k0_qaoa["Mean MAE"]:
    print("\nFINAL RESULT: Pair probes improve downstream QAOA reliability.")
else:
    print("\nFINAL RESULT: Pair probes did not improve QAOA in this scaled run.")

# Save final master pack
final_master_pack = {
    "FINAL_RUN_CONFIG": FINAL_RUN_CONFIG,
    "final_local_channel_table": final_local_channel_table,
    "final_local_qaoa_table": final_local_qaoa_table,
    "final_pairprobe_eval_table": final_pairprobe_eval_table,
    "final_pair_qaoa_table": final_pair_qaoa_table,
    "final_pair_qaoa_byK": final_pair_qaoa_byK,
}

final_save_pickle(final_master_pack, "FINAL_MASTER_SUMMARY.pkl")

print("\nSaved all final scaled results to:")
print(FINAL_DIR)

# %% [markdown]
# ### PRA Step 3: non-oracle target-edge observable mitigation
#
# This block adds a non-oracle replacement for the oracle-assisted bias-correction diagnostic.  Instead of using
# \(C_{\rm ideal}\) inside the correction rule, the target QAOA circuit is measured in the required edge Pauli bases, the learned local-plus-edge response matrix is inverted with ridge regularization, and the corrected MaxCut cost is reconstructed from the corrected \(ZZ\) edge observables.  The ideal landscape is used only afterwards to report simulation metrics.

# %% Cell 55
# ============================================================
# PRA STEP 3 — Non-oracle target-edge observable mitigation
# ============================================================
# This patch replaces the oracle-assisted landscape-bias correction
#     C_mit = C_noisy - (C_model - C_ideal)
# by a non-oracle edge-observable estimator for the MaxCut cost.
#
# The new protocol uses only:
#   1. target-circuit measurements of edge Pauli observables,
#   2. the learned local-plus-edge response model,
#   3. a regularized inverse of the learned edge response.
#
# C_ideal is used only after correction to evaluate accuracy in simulation;
# it is not used to construct C_mit.

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Requirements
# ------------------------------------------------------------
required_for_pra_step3 = [
    "PAIR_EDGES_3Q",
    "PAIR_AXIS_PAIRS",
    "PAIR_PAULI_LABELS",
    "N_QAOA",
    "rho_ideal_grid_3q",
    "gamma_grid_3q",
    "beta_grid_3q",
    "affine_channel_to_ptm",
    "unflatten_local_channels",
    "split_pair_process_residual_blocks",
    "rho_to_pauli_coeffs_3q",
    "pauli_string_labels_3q",
    "landscape_error_metrics_3q",
    "selected_point_metrics",
    "Popt_ideal_3q",
    "C_ideal_3q",
]

missing_step3 = [name for name in required_for_pra_step3 if name not in globals()]
if missing_step3:
    raise RuntimeError(
        "Missing requirements for PRA Step 3: " + str(missing_step3) +
        "\nRun the QAOA cells and PRA Step 2 cells first."
    )

EDGE_1Q_LABELS_16 = ["I", "X", "Y", "Z"]
EDGE_LABEL_PAIRS_16 = [(a, b) for a in EDGE_1Q_LABELS_16 for b in EDGE_1Q_LABELS_16]
EDGE_PAIR_INDEX_16 = {pair: k for k, pair in enumerate(EDGE_LABEL_PAIRS_16)}
EDGE_ZZ_INDEX_16 = EDGE_PAIR_INDEX_16[("Z", "Z")]
EDGE_NONTRIVIAL_PAIR_INDICES_16 = np.array(
    [EDGE_PAIR_INDEX_16[pair] for pair in PAIR_AXIS_PAIRS],
    dtype=int
)

print("PRA Step 3: non-oracle edge-observable estimator")
print("16-component edge basis:", EDGE_LABEL_PAIRS_16)
print("ZZ index:", EDGE_ZZ_INDEX_16)


# ------------------------------------------------------------
# Edge coefficient extraction
# ------------------------------------------------------------
def full_pauli_index_for_edge_label_pair(edge, label_i, label_j):
    """
    Index in the 3-qubit Pauli coefficient vector corresponding to
    label_i on edge[0], label_j on edge[1], and identity elsewhere.
    """
    labels = ["I"] * N_QAOA
    qi, qj = edge
    labels[qi] = label_i
    labels[qj] = label_j
    return pauli_string_labels_3q.index("".join(labels))


EDGE_COEFF_INDICES_16_BY_EDGE = np.array([
    [full_pauli_index_for_edge_label_pair(edge, li, lj) for (li, lj) in EDGE_LABEL_PAIRS_16]
    for edge in PAIR_EDGES_3Q
], dtype=int)


def edge_coeff_vector_16_from_rho_3q(rho, edge_index):
    """
    Return the 16 Pauli coefficients of the two-qubit reduced operator
    on a hardware edge, ordered as II, IX, IY, IZ, XI, ..., ZZ.

    These coefficients are expectation values Tr[rho P_a^i P_b^j].
    """
    coeffs_full = rho_to_pauli_coeffs_3q(rho)
    return coeffs_full[EDGE_COEFF_INDICES_16_BY_EDGE[edge_index]].astype(np.float64)


# ------------------------------------------------------------
# Learned edge-response matrices
# ------------------------------------------------------------
def edge_response_matrix_16_from_local_pair_3q(
    y_channel,
    edge_index,
    y_pair_residual=None,
    eta_pair=1.0
):
    """
    Build a 16 x 16 edge response matrix.

    The base response is the tensor product of the two learned local affine
    Pauli-transfer matrices. If a process-relative pair residual is supplied,
    its 9 x 9 block is inserted into the two-body Pauli-pair sector.

    This gives an edge-level model
        c_noisy_edge ≈ R_edge c_ideal_edge.
    """
    edge = PAIR_EDGES_3Q[edge_index]
    qi, qj = edge

    A_list, b_list = unflatten_local_channels(y_channel, N=N_QAOA)
    R_i = affine_channel_to_ptm(A_list[qi], b_list[qi])
    R_j = affine_channel_to_ptm(A_list[qj], b_list[qj])

    R_edge = np.kron(R_i, R_j).astype(np.float64)

    if y_pair_residual is not None:
        y_pair_residual = np.asarray(y_pair_residual, dtype=np.float64)
        if y_pair_residual.size > 0:
            blocks = split_pair_process_residual_blocks(y_pair_residual)
            block = blocks[edge_index]
            idx = EDGE_NONTRIVIAL_PAIR_INDICES_16
            R_edge[np.ix_(idx, idx)] += float(eta_pair) * block

    return R_edge


# ------------------------------------------------------------
# Finite-shot target-circuit edge measurements
# ------------------------------------------------------------
def sample_edge_coeff_vector_16(c_exact, shots=2048, rng=None):
    """
    Simulate finite-shot estimates of all nontrivial two-qubit Pauli
    coefficients on an edge. Each Pauli coefficient is measured in its own
    basis as a binary ±1 observable. The identity coefficient is fixed to 1.
    """
    if rng is None:
        rng = np.random.default_rng()

    c_exact = np.asarray(c_exact, dtype=np.float64)
    c_hat = np.zeros_like(c_exact, dtype=np.float64)
    c_hat[0] = 1.0

    for k in range(1, len(c_exact)):
        m = float(np.clip(c_exact[k], -1.0, 1.0))
        p_plus = 0.5 * (1.0 + m)
        n_plus = rng.binomial(int(shots), p_plus)
        c_hat[k] = (2.0 * n_plus - int(shots)) / int(shots)

    return c_hat


def generate_target_edge_measurements_3q(
    y_channel_true,
    y_pair_true,
    rho_ideal_grid,
    shots=2048,
    seed=0,
    eta_pair_true=1.0
):
    """
    Generate simulated target-circuit measurements for the non-oracle
    estimator.

    This is only the data-generation side of the numerical benchmark. It uses
    the hidden true response to simulate what hardware would output. The
    correction rule below does not use C_ideal or ideal target observables.

    Returns
    -------
    edge_meas : array, shape (n_gamma, n_beta, n_edges, 16)
        Finite-shot measured edge Pauli coefficients.
    edge_exact : array, same shape
        Exact noisy edge coefficients before shot sampling.
    """
    rng = np.random.default_rng(seed)

    n_gamma = len(gamma_grid_3q)
    n_beta = len(beta_grid_3q)
    n_edges = len(PAIR_EDGES_3Q)

    edge_meas = np.zeros((n_gamma, n_beta, n_edges, 16), dtype=np.float64)
    edge_exact = np.zeros_like(edge_meas)

    R_true_edges = [
        edge_response_matrix_16_from_local_pair_3q(
            y_channel=y_channel_true,
            edge_index=e_idx,
            y_pair_residual=y_pair_true,
            eta_pair=eta_pair_true
        )
        for e_idx in range(n_edges)
    ]

    for ig in range(n_gamma):
        for ib in range(n_beta):
            rho_ideal = rho_ideal_grid[ig][ib]

            for e_idx in range(n_edges):
                c_ideal_edge = edge_coeff_vector_16_from_rho_3q(rho_ideal, e_idx)
                c_noisy_edge = R_true_edges[e_idx] @ c_ideal_edge
                c_noisy_edge[0] = 1.0
                c_noisy_edge[1:] = np.clip(c_noisy_edge[1:], -1.0, 1.0)

                edge_exact[ig, ib, e_idx] = c_noisy_edge
                edge_meas[ig, ib, e_idx] = sample_edge_coeff_vector_16(
                    c_noisy_edge,
                    shots=shots,
                    rng=rng
                )

    return edge_meas, edge_exact


def cost_landscape_from_edge_coeff_measurements_3q(edge_coeffs):
    """
    Compute MaxCut cost from measured or corrected edge ZZ coefficients.
    """
    edge_coeffs = np.asarray(edge_coeffs, dtype=np.float64)
    n_gamma, n_beta, n_edges, _ = edge_coeffs.shape

    C = np.zeros((n_gamma, n_beta), dtype=np.float64)
    for e_idx in range(n_edges):
        zz = edge_coeffs[:, :, e_idx, EDGE_ZZ_INDEX_16]
        C += 0.5 * (1.0 - zz)

    return np.clip(C, 0.0, len(PAIR_EDGES_3Q))


# ------------------------------------------------------------
# Regularized inverse edge-observable estimator
# ------------------------------------------------------------
def regularized_invert_edge_response_16(c_noisy, R_edge, lam=1e-3, clip=True):
    """
    Estimate ideal edge coefficients from measured noisy edge coefficients:

        c_noisy ≈ R_edge c_ideal.

    The identity coefficient is constrained to c_ideal[0] = 1, and the
    remaining 15 coefficients are obtained by ridge-regularized least squares.
    """
    c_noisy = np.asarray(c_noisy, dtype=np.float64)
    R_edge = np.asarray(R_edge, dtype=np.float64)

    # Enforce the physical identity coefficient explicitly.
    b = c_noisy - R_edge[:, 0]
    A = R_edge[:, 1:]

    lhs = A.T @ A + float(lam) * np.eye(A.shape[1])
    rhs = A.T @ b

    try:
        x = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    c_hat = np.empty(16, dtype=np.float64)
    c_hat[0] = 1.0
    c_hat[1:] = x

    if clip:
        c_hat[1:] = np.clip(c_hat[1:], -1.0, 1.0)

    return c_hat


def corrected_landscape_from_edge_observable_estimator_3q(
    edge_measurements,
    y_channel_model,
    y_pair_model=None,
    lam=1e-3,
    eta_pair_model=1.0,
    clip=True
):
    """
    Non-oracle mitigation rule.

    Inputs are target-circuit edge measurements and a learned response model.
    The ideal QAOA landscape is not used. For each edge and grid point, the
    learned edge response is inverted to estimate the corrected ZZ observable.
    """
    edge_measurements = np.asarray(edge_measurements, dtype=np.float64)
    n_gamma, n_beta, n_edges, _ = edge_measurements.shape

    R_model_edges = [
        edge_response_matrix_16_from_local_pair_3q(
            y_channel=y_channel_model,
            edge_index=e_idx,
            y_pair_residual=y_pair_model,
            eta_pair=eta_pair_model
        )
        for e_idx in range(n_edges)
    ]

    corrected_edges = np.zeros_like(edge_measurements)

    for ig in range(n_gamma):
        for ib in range(n_beta):
            for e_idx in range(n_edges):
                corrected_edges[ig, ib, e_idx] = regularized_invert_edge_response_16(
                    c_noisy=edge_measurements[ig, ib, e_idx],
                    R_edge=R_model_edges[e_idx],
                    lam=lam,
                    clip=clip
                )

    C_corr = cost_landscape_from_edge_coeff_measurements_3q(corrected_edges)
    return C_corr, corrected_edges


# ------------------------------------------------------------
# Evaluation helper for pair-aware models
# ------------------------------------------------------------
def evaluate_qaoa_pair_device_nonoracle_3q(
    idx_device,
    Y_ch_true_all,
    Y_pair_true_all,
    prediction_methods,
    qaoa_shots=2048,
    seed=0,
    lam=1e-3,
    eta_pair_model=1.0
):
    """
    Evaluate the non-oracle target-edge observable estimator.

    Unlike the oracle-assisted Eq. (23) diagnostic, this function does not use
    C_ideal to build the corrected landscape. C_ideal is used only afterwards
    to report simulation metrics.
    """
    y_ch_true = Y_ch_true_all[idx_device]
    y_pair_true = Y_pair_true_all[idx_device]

    edge_meas, edge_exact = generate_target_edge_measurements_3q(
        y_channel_true=y_ch_true,
        y_pair_true=y_pair_true,
        rho_ideal_grid=rho_ideal_grid_3q,
        shots=qaoa_shots,
        seed=seed + idx_device,
        eta_pair_true=1.0
    )

    C_noisy_edge = cost_landscape_from_edge_coeff_measurements_3q(edge_meas)

    results = {
        "Noisy edge measured": {
            "C": C_noisy_edge,
            "edge_measurements": edge_meas,
            "error": landscape_error_metrics_3q(C_noisy_edge, C_ideal_3q),
            "selected": selected_point_metrics(C_noisy_edge, C_ideal_3q, Popt_ideal_3q),
            "uses_oracle_correction": False,
        }
    }

    for method_name, pred_pack in prediction_methods.items():
        y_ch_model = pred_pack["Y_channel"][idx_device]

        if pred_pack.get("Y_pair", None) is not None:
            y_pair_model = pred_pack["Y_pair"][idx_device]
        else:
            y_pair_model = None

        C_corr, corrected_edges = corrected_landscape_from_edge_observable_estimator_3q(
            edge_measurements=edge_meas,
            y_channel_model=y_ch_model,
            y_pair_model=y_pair_model,
            lam=lam,
            eta_pair_model=eta_pair_model,
            clip=True
        )

        results[method_name] = {
            "C": C_corr,
            "corrected_edges": corrected_edges,
            "error": landscape_error_metrics_3q(C_corr, C_ideal_3q),
            "selected": selected_point_metrics(C_corr, C_ideal_3q, Popt_ideal_3q),
            "uses_oracle_correction": False,
        }

    return results


def summarize_qaoa_results_list_nonoracle_3q(results_list):
    """Summarize a list returned by evaluate_qaoa_pair_device_nonoracle_3q."""
    methods = list(results_list[0].keys())
    noisy_mae = np.mean([r["Noisy edge measured"]["error"]["MAE"] for r in results_list])

    rows = []
    for method in methods:
        mae_arr = np.array([r[method]["error"]["MAE"] for r in results_list], dtype=np.float64)
        rmse_arr = np.array([r[method]["error"]["RMSE"] for r in results_list], dtype=np.float64)
        regret_arr = np.array([r[method]["selected"]["regret"] for r in results_list], dtype=np.float64)
        approx_arr = np.array([r[method]["selected"]["approximation_ratio"] for r in results_list], dtype=np.float64)
        popt_arr = np.array([r[method]["selected"]["p_opt_selected"] for r in results_list], dtype=np.float64)
        disp_arr = np.array([r[method]["selected"].get("parameter_displacement", np.nan) for r in results_list], dtype=np.float64)

        rows.append({
            "Method": method,
            "Mean MAE": float(mae_arr.mean()),
            "Std MAE": float(mae_arr.std(ddof=1)) if len(mae_arr) > 1 else 0.0,
            "Mean RMSE": float(rmse_arr.mean()),
            "Mean regret": float(regret_arr.mean()),
            "Mean approximation ratio": float(approx_arr.mean()),
            "Mean P_opt": float(popt_arr.mean()),
            "Mean parameter displacement": float(np.nanmean(disp_arr)),
            "Improvement ratio": float(noisy_mae / (mae_arr.mean() + 1e-12)),
            "Oracle correction used": False,
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Optional run block
# ------------------------------------------------------------
# Example usage after Step 2 pair-probe models are trained:
#
# nonoracle_prediction_methods_3q = {
#     "Kpair=0 local only": {
#         "Y_channel": pairprobe_results_3q[0]["Y_pred_channel"],
#         "Y_pair": None,
#     },
#     "Kpair=18 local+pair": {
#         "Y_channel": pairprobe_results_3q[18]["Y_pred_channel"],
#         "Y_pair": pairprobe_results_3q[18]["Y_pred_pair"],
#     },
#     "Oracle true local+pair response": {
#         "Y_channel": Y_ch_true_pp,
#         "Y_pair": Y_pair_true_pp,
#     },
# }
#
# N_NONORACLE_EVAL_3Q = min(10, len(Y_ch_true_pp))
# results_qaoa_nonoracle_3q = []
# for k in range(N_NONORACLE_EVAL_3Q):
#     res = evaluate_qaoa_pair_device_nonoracle_3q(
#         idx_device=k,
#         Y_ch_true_all=Y_ch_true_pp,
#         Y_pair_true_all=Y_pair_true_pp,
#         prediction_methods=nonoracle_prediction_methods_3q,
#         qaoa_shots=2048,
#         seed=91000,
#         lam=1e-3,
#         eta_pair_model=1.0,
#     )
#     results_qaoa_nonoracle_3q.append(res)
#
# nonoracle_qaoa_table_3q = summarize_qaoa_results_list_nonoracle_3q(
#     results_qaoa_nonoracle_3q
# )
# display(nonoracle_qaoa_table_3q)

# %% Cell 56
import os, glob

FINAL_DIR = "outputs/three_qubit_qaoa_FINAL_scaled_run"

for f in sorted(glob.glob(os.path.join(FINAL_DIR, "*"))):
    print(os.path.basename(f))

# %% [markdown]
# ### PRA Step 4: confidence intervals and repeated-seed reporting
#
# This block adds the statistical reporting layer requested for the PRA revision.  It builds confidence intervals over hidden-device samples from existing per-device QAOA results, repeats finite-shot QAOA evaluations over several sampling seeds, and provides a controlled neural-network-seed repeat for ChannelNet.  These cells do not change the simulator or models; they turn the results into mean and 95% confidence-interval tables.

# %% Cell 58
# ============================================================
# PRA STEP 4 — Confidence intervals and repeated-seed reporting
# ============================================================
# Purpose:
#   Report uncertainty over:
#       1. hidden-device samples,
#       2. finite-shot sampling seeds,
#       3. neural-network training seeds.
#
# This block does not change the physics model. It adds a reproducible
# evaluation layer that turns per-device/per-seed results into mean ± 95% CI
# tables suitable for PRA-style reporting.
# ============================================================

import os
import math
import numpy as np
import pandas as pd

try:
    from scipy.stats import t as student_t
except Exception:
    student_t = None

# ------------------------------------------------------------
# Generic confidence-interval helpers
# ------------------------------------------------------------

def mean_ci_t(values, confidence=0.95):
    """Return mean, standard deviation, standard error, and Student-t CI."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)

    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "sem": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "ci_half_width": np.nan,
        }

    mean = float(np.mean(x))

    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "std": 0.0,
            "sem": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "ci_half_width": np.nan,
        }

    std = float(np.std(x, ddof=1))
    sem = float(std / np.sqrt(n))

    if student_t is not None:
        q = float(student_t.ppf((1.0 + confidence) / 2.0, df=n - 1))
    else:
        # conservative normal approximation fallback
        q = 1.96

    half = float(q * sem)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "ci_half_width": half,
    }


def format_mean_ci(mean, lo, hi, digits=4):
    """Format a mean and confidence interval for tables."""
    if not np.isfinite(mean):
        return "nan"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{mean:.{digits}g}"
    return f"{mean:.{digits}g} [{lo:.{digits}g}, {hi:.{digits}g}]"


def qaoa_results_to_long(
    results,
    experiment="qaoa",
    K=None,
    K_pair=None,
    shot_seed=None,
    nn_seed=None,
    split="test",
):
    """Convert a list of per-device QAOA result dictionaries to long format."""
    rows = []

    for device_index, r in enumerate(results):
        for method, pack in r.items():
            err = pack.get("error", {})
            sel = pack.get("selected", {})

            rows.append({
                "experiment": experiment,
                "split": split,
                "device_index": device_index,
                "shot_seed": shot_seed,
                "nn_seed": nn_seed,
                "K": K,
                "K_pair": K_pair,
                "Method": method,
                "MAE": err.get("MAE", np.nan),
                "RMSE": err.get("RMSE", np.nan),
                "regret": sel.get("regret", np.nan),
                "approximation_ratio": sel.get("approximation_ratio", np.nan),
                "P_opt": sel.get("p_opt_selected", np.nan),
                "parameter_displacement": sel.get("parameter_displacement", np.nan),
                "uses_oracle_correction": pack.get("uses_oracle_correction", np.nan),
            })

    return pd.DataFrame(rows)


def summarize_long_qaoa_ci(
    df_long,
    group_cols=("experiment", "K", "K_pair", "Method"),
    confidence=0.95,
):
    """Summarize long-format QAOA rows with mean and 95% CI."""
    metric_cols = [
        "MAE",
        "RMSE",
        "regret",
        "approximation_ratio",
        "P_opt",
        "parameter_displacement",
    ]

    group_cols = [c for c in group_cols if c in df_long.columns]
    rows = []

    for keys, g in df_long.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))

        for metric in metric_cols:
            if metric not in g.columns:
                continue
            ci = mean_ci_t(g[metric].values, confidence=confidence)
            row = dict(base)
            row.update({
                "metric": metric,
                "n": ci["n"],
                "mean": ci["mean"],
                "std": ci["std"],
                "sem": ci["sem"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "ci_half_width": ci["ci_half_width"],
                "mean_95CI": format_mean_ci(ci["mean"], ci["ci_low"], ci["ci_high"]),
            })
            rows.append(row)

    return pd.DataFrame(rows)


def add_improvement_ratio_ci(df_long, baseline_method="Noisy", metric="MAE"):
    """
    Add mean-level improvement ratio relative to a baseline method.
    This reports ratio of means, not a bootstrap CI for the ratio.
    """
    df = df_long.copy()
    grouping = [c for c in ["experiment", "K", "K_pair", "shot_seed", "nn_seed"] if c in df.columns]

    out = []
    for keys, g in df.groupby(grouping, dropna=False) if grouping else [((), df)]:
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(grouping, keys))

        b = g[g["Method"] == baseline_method]
        if len(b) == 0:
            # Common non-oracle baseline name
            b = g[g["Method"] == "Noisy edge measured"]
        if len(b) == 0:
            baseline_mean = np.nan
        else:
            baseline_mean = float(np.nanmean(b[metric]))

        for method, gm in g.groupby("Method"):
            m = float(np.nanmean(gm[metric]))
            row = dict(base)
            row.update({
                "Method": method,
                f"Mean {metric}": m,
                "Baseline mean": baseline_mean,
                "Improvement ratio": baseline_mean / (m + 1e-12) if np.isfinite(baseline_mean) else np.nan,
            })
            out.append(row)

    return pd.DataFrame(out)


# ------------------------------------------------------------
# Hidden-device confidence intervals from existing QAOA packs
# ------------------------------------------------------------

def build_hidden_device_ci_tables_from_existing_results(save=True):
    """
    Build CI tables from already-computed per-device QAOA result lists.
    This captures uncertainty over hidden-device samples in the test/eval set.
    """
    long_parts = []

    if "local_qaoa_pack" in globals():
        for K, pack in local_qaoa_pack.items():
            long_parts.append(qaoa_results_to_long(
                pack["results"],
                experiment="local_qaoa_oracle_diagnostic",
                K=K,
                split="test",
            ))

    if "final_pair_qaoa_pack" in globals():
        long_parts.append(qaoa_results_to_long(
            final_pair_qaoa_pack["results"],
            experiment="pair_qaoa_oracle_diagnostic",
            split="test",
        ))

    if "results_qaoa_nonoracle_3q" in globals():
        long_parts.append(qaoa_results_to_long(
            results_qaoa_nonoracle_3q,
            experiment="pair_qaoa_nonoracle_edge_estimator",
            split="test",
        ))

    if len(long_parts) == 0:
        raise RuntimeError(
            "No existing QAOA result lists found. Run the local/pair QAOA evaluation cells first."
        )

    df_long = pd.concat(long_parts, ignore_index=True)
    df_ci = summarize_long_qaoa_ci(df_long)
    df_improvement = add_improvement_ratio_ci(df_long)

    if save and "FINAL_DIR" in globals():
        df_long.to_csv(os.path.join(FINAL_DIR, "pra_step4_hidden_device_qaoa_long.csv"), index=False)
        df_ci.to_csv(os.path.join(FINAL_DIR, "pra_step4_hidden_device_qaoa_ci.csv"), index=False)
        df_improvement.to_csv(os.path.join(FINAL_DIR, "pra_step4_hidden_device_improvement_ratios.csv"), index=False)
        print("Saved Step 4 hidden-device CI tables to", FINAL_DIR)

    return df_long, df_ci, df_improvement


# ------------------------------------------------------------
# Finite-shot seed repeated QAOA evaluation
# ------------------------------------------------------------

def run_local_qaoa_finite_shot_seed_repeats(
    K_list=(18, 54),
    shot_seeds=(1001, 1002, 1003, 1004, 1005),
    n_devices=None,
    save=True,
):
    """Repeat local QAOA evaluation for several finite-shot seeds."""
    if "final_local_sweep_results" not in globals():
        raise RuntimeError("final_local_sweep_results not found. Run the final local sweep first.")

    long_parts = []

    for K in K_list:
        res_K = final_local_sweep_results[K]
        Y_true = res_K["test_ds"]["Y"]
        pred_methods = {
            "Direct": res_K["predictions"]["Direct"],
            "Ridge": res_K["predictions"]["Ridge"],
            "RandomForest": res_K["predictions"]["RandomForest"],
            "ChannelNet": res_K["predictions"]["ChannelNet"],
        }

        n_eval = min(n_devices or len(Y_true), len(Y_true))

        for shot_seed in shot_seeds:
            results = []
            for k in range(n_eval):
                r = evaluate_qaoa_device_3q(
                    idx_device=k,
                    Y_true_all=Y_true,
                    prediction_methods=pred_methods,
                    qaoa_shots=qaoa_shots_3q,
                    seed=int(shot_seed),
                )
                results.append(r)

            long_parts.append(qaoa_results_to_long(
                results,
                experiment="local_qaoa_finite_shot_repeats",
                K=K,
                shot_seed=int(shot_seed),
                split="test",
            ))
            print(f"Finished local finite-shot repeat K={K}, shot_seed={shot_seed}")

    df_long = pd.concat(long_parts, ignore_index=True)
    df_ci = summarize_long_qaoa_ci(df_long, group_cols=("experiment", "K", "Method"))

    if save and "FINAL_DIR" in globals():
        df_long.to_csv(os.path.join(FINAL_DIR, "pra_step4_local_finite_shot_repeats_long.csv"), index=False)
        df_ci.to_csv(os.path.join(FINAL_DIR, "pra_step4_local_finite_shot_repeats_ci.csv"), index=False)
        print("Saved local finite-shot repeat CI tables to", FINAL_DIR)

    return df_long, df_ci


def run_pair_nonoracle_finite_shot_seed_repeats(
    prediction_methods,
    shot_seeds=(2001, 2002, 2003, 2004, 2005),
    n_devices=20,
    lam=1e-3,
    eta_pair_model=1.0,
    save=True,
):
    """Repeat non-oracle edge-observable QAOA evaluation for several shot seeds."""
    required = [
        "evaluate_qaoa_pair_device_nonoracle_3q",
        "Y_ch_true_final_pair",
        "Y_pair_true_final_pair",
    ]
    missing = [name for name in required if name not in globals()]
    if missing:
        raise RuntimeError("Missing non-oracle pair requirements: " + str(missing))

    long_parts = []
    n_eval = min(n_devices, len(Y_ch_true_final_pair))

    for shot_seed in shot_seeds:
        results = []
        for k in range(n_eval):
            r = evaluate_qaoa_pair_device_nonoracle_3q(
                idx_device=k,
                Y_ch_true_all=Y_ch_true_final_pair,
                Y_pair_true_all=Y_pair_true_final_pair,
                prediction_methods=prediction_methods,
                qaoa_shots=qaoa_shots_3q,
                seed=int(shot_seed),
                lam=lam,
                eta_pair_model=eta_pair_model,
            )
            results.append(r)

        long_parts.append(qaoa_results_to_long(
            results,
            experiment="pair_qaoa_nonoracle_finite_shot_repeats",
            shot_seed=int(shot_seed),
            split="test",
        ))
        print(f"Finished pair non-oracle finite-shot repeat shot_seed={shot_seed}")

    df_long = pd.concat(long_parts, ignore_index=True)
    df_ci = summarize_long_qaoa_ci(df_long, group_cols=("experiment", "Method"))

    if save and "FINAL_DIR" in globals():
        df_long.to_csv(os.path.join(FINAL_DIR, "pra_step4_pair_nonoracle_finite_shot_repeats_long.csv"), index=False)
        df_ci.to_csv(os.path.join(FINAL_DIR, "pra_step4_pair_nonoracle_finite_shot_repeats_ci.csv"), index=False)
        print("Saved pair non-oracle finite-shot repeat CI tables to", FINAL_DIR)

    return df_long, df_ci


# ------------------------------------------------------------
# Neural-network seed repeated local ChannelNet training
# ------------------------------------------------------------

def train_local_channelnet_for_nn_seed(
    K=18,
    nn_seed=0,
    n_epochs=None,
):
    """Retrain only ChannelNet for a fixed K and a different NN seed."""
    if "final_local_data" not in globals():
        raise RuntimeError("final_local_data not found. Run the final local data cell first.")

    train_ds = build_masked_from_final_full(
        Xfull=final_local_data["train"]["Xfull"],
        Y=final_local_data["train"]["Y"],
        shots=final_local_data["train"]["shots"],
        K=K,
        seed=FINAL_RUN_CONFIG["BASE_SEED"] + 100 + K,
    )
    val_ds = build_masked_from_final_full(
        Xfull=final_local_data["val"]["Xfull"],
        Y=final_local_data["val"]["Y"],
        shots=final_local_data["val"]["shots"],
        K=K,
        seed=FINAL_RUN_CONFIG["BASE_SEED"] + 200 + K,
    )
    test_ds = build_masked_from_final_full(
        Xfull=final_local_data["test"]["Xfull"],
        Y=final_local_data["test"]["Y"],
        shots=final_local_data["test"]["shots"],
        K=K,
        seed=FINAL_RUN_CONFIG["BASE_SEED"] + 300 + K,
    )

    pack = train_final_channelnet(
        X_train=train_ds["X"],
        Y_train=train_ds["Y"],
        X_val=val_ds["X"],
        Y_val=val_ds["Y"],
        n_epochs=n_epochs or FINAL_RUN_CONFIG["LOCAL_EPOCHS"],
        batch_size=FINAL_RUN_CONFIG["BATCH_SIZE"],
        lr=FINAL_RUN_CONFIG["LR"],
        weight_decay=FINAL_RUN_CONFIG["WEIGHT_DECAY"],
        lambda_phys=FINAL_RUN_CONFIG["LAMBDA_PHYS"],
        seed=int(nn_seed),
    )

    Y_pred = predict_final_channelnet(pack, test_ds["X"])

    channel_mse = float(np.mean((Y_pred - test_ds["Y"]) ** 2))
    channel_l2 = np.linalg.norm(Y_pred - test_ds["Y"], axis=1)

    return {
        "K": K,
        "nn_seed": int(nn_seed),
        "pack": pack,
        "test_ds": test_ds,
        "Y_pred": Y_pred,
        "channel_mse": channel_mse,
        "channel_l2_mean": float(np.mean(channel_l2)),
        "channel_l2_std": float(np.std(channel_l2, ddof=1)) if len(channel_l2) > 1 else 0.0,
    }


def run_local_channelnet_nn_seed_repeats(
    K_list=(18, 54),
    nn_seeds=(301, 302, 303),
    n_qaoa_devices=20,
    n_epochs=None,
    qaoa_seed=99001,
    save=True,
):
    """Retrain ChannelNet for different NN seeds and evaluate channel + QAOA metrics."""
    all_rows = []
    qaoa_long_parts = []
    trained_packs = {}

    for K in K_list:
        for nn_seed in nn_seeds:
            print("=" * 100)
            print(f"Training local ChannelNet NN seed repeat: K={K}, nn_seed={nn_seed}")
            rep = train_local_channelnet_for_nn_seed(K=K, nn_seed=nn_seed, n_epochs=n_epochs)
            trained_packs[(K, int(nn_seed))] = rep

            all_rows.append({
                "experiment": "local_channelnet_nn_seed_repeats",
                "K": K,
                "nn_seed": int(nn_seed),
                "channel_mse": rep["channel_mse"],
                "channel_l2_mean": rep["channel_l2_mean"],
                "channel_l2_std": rep["channel_l2_std"],
                "best_val_mse": float(rep["pack"]["best_val_mse"]),
            })

            # QAOA evaluation for this NN seed
            Y_true = rep["test_ds"]["Y"]
            pred_methods = {"ChannelNet": rep["Y_pred"]}
            n_eval = min(n_qaoa_devices, len(Y_true))
            qaoa_results = []
            for k in range(n_eval):
                r = evaluate_qaoa_device_3q(
                    idx_device=k,
                    Y_true_all=Y_true,
                    prediction_methods=pred_methods,
                    qaoa_shots=qaoa_shots_3q,
                    seed=int(qaoa_seed),
                )
                qaoa_results.append(r)

            qaoa_long_parts.append(qaoa_results_to_long(
                qaoa_results,
                experiment="local_channelnet_nn_seed_qaoa",
                K=K,
                nn_seed=int(nn_seed),
                split="test",
            ))

    df_channel = pd.DataFrame(all_rows)
    df_channel_ci = []
    for K, g in df_channel.groupby("K"):
        for metric in ["channel_mse", "channel_l2_mean", "best_val_mse"]:
            ci = mean_ci_t(g[metric].values)
            df_channel_ci.append({
                "experiment": "local_channelnet_nn_seed_repeats",
                "K": K,
                "metric": metric,
                **ci,
                "mean_95CI": format_mean_ci(ci["mean"], ci["ci_low"], ci["ci_high"]),
            })
    df_channel_ci = pd.DataFrame(df_channel_ci)

    df_qaoa_long = pd.concat(qaoa_long_parts, ignore_index=True)
    df_qaoa_ci = summarize_long_qaoa_ci(df_qaoa_long, group_cols=("experiment", "K", "Method"))

    if save and "FINAL_DIR" in globals():
        df_channel.to_csv(os.path.join(FINAL_DIR, "pra_step4_local_channelnet_nn_seed_channel_long.csv"), index=False)
        df_channel_ci.to_csv(os.path.join(FINAL_DIR, "pra_step4_local_channelnet_nn_seed_channel_ci.csv"), index=False)
        df_qaoa_long.to_csv(os.path.join(FINAL_DIR, "pra_step4_local_channelnet_nn_seed_qaoa_long.csv"), index=False)
        df_qaoa_ci.to_csv(os.path.join(FINAL_DIR, "pra_step4_local_channelnet_nn_seed_qaoa_ci.csv"), index=False)
        final_save_pickle(trained_packs, "pra_step4_local_channelnet_nn_seed_trained_packs.pkl")
        print("Saved local ChannelNet NN-seed repeat tables to", FINAL_DIR)

    return df_channel, df_channel_ci, df_qaoa_long, df_qaoa_ci, trained_packs


# ------------------------------------------------------------
# Suggested lightweight run commands
# ------------------------------------------------------------
# 1. Hidden-device CIs from existing results:
#
# df_hidden_long, df_hidden_ci, df_hidden_imp = build_hidden_device_ci_tables_from_existing_results()
# display(df_hidden_ci)
#
# 2. Finite-shot CIs for final local results:
#
# df_shot_long, df_shot_ci = run_local_qaoa_finite_shot_seed_repeats(
#     K_list=(18, 54),
#     shot_seeds=(1001, 1002, 1003, 1004, 1005),
#     n_devices=50,
# )
# display(df_shot_ci)
#
# 3. Neural-network seed CIs for ChannelNet:
#
# df_nn_ch, df_nn_ch_ci, df_nn_qaoa_long, df_nn_qaoa_ci, nn_packs = run_local_channelnet_nn_seed_repeats(
#     K_list=(18, 54),
#     nn_seeds=(301, 302, 303, 304, 305),
#     n_qaoa_devices=50,
# )
# display(df_nn_ch_ci)
# display(df_nn_qaoa_ci)
#
# For a quick smoke test before the full run, use n_devices=5, n_qaoa_devices=5,
# and n_epochs=20.

# %% [markdown]
# ### PRA Step 5: operational ablation study
#
# This block adds the requested ablation layer: **local-only non-oracle correction**, **local-plus-pair correction with shrinkage \(\eta\)** selected on validation devices, and a **direct QAOA-aware loss** option for the pair-probe learner.  The goal is to test whether pair probes are operationally useful for the target QAOA/MaxCut observable, not only identifiable as pair residual labels.

# %% Cell 60
# ============================================================
# PRA STEP 5 — Ablation study for operational pair-probe usefulness
# ============================================================
# Required PRA ablation:
#   1. local-only correction,
#   2. pair-residual correction with shrinkage eta,
#   3. direct QAOA-aware loss.
#
# This block is cumulative with Steps 1--4. It assumes that the Step 2
# process-relative pair residuals and the Step 3 non-oracle edge-observable
# estimator are already defined.
# ============================================================

import os
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ------------------------------------------------------------
# Requirements
# ------------------------------------------------------------
required_for_pra_step5 = [
    "PAIR_EDGES_3Q",
    "PAIR_AXIS_PAIRS",
    "PAIR_LABEL_DIM",
    "PAIR_AUG_DIM",
    "LABEL_DIM",
    "N_QUBITS",
    "N_QAOA",
    "PairProbeAwareNet3Q",
    "split_channels_torch_pair",
    "bloch_test_t_pair",
    "build_local_pairprobe_dataset",
    "compute_standardizer",
    "apply_standardizer",
    "invert_standardizer",
    "split_channel_pair",
    "edge_coeff_vector_16_from_rho_3q",
    "rho_ideal_grid_3q",
    "EDGE_NONTRIVIAL_PAIR_INDICES_16",
    "EDGE_ZZ_INDEX_16",
    "generate_target_edge_measurements_3q",
    "corrected_landscape_from_edge_observable_estimator_3q",
    "cost_landscape_from_edge_coeff_measurements_3q",
    "landscape_error_metrics_3q",
    "selected_point_metrics",
    "Popt_ideal_3q",
    "C_ideal_3q",
]

missing_step5 = [name for name in required_for_pra_step5 if name not in globals()]
if missing_step5:
    raise RuntimeError(
        "Missing requirements for PRA Step 5: " + str(missing_step5) +
        "\nRun the simulator, PRA Step 2, and PRA Step 3 cells first."
    )

print("PRA Step 5 requirements are available.")


# ------------------------------------------------------------
# A. Helper: predict a trained pair-probe model on val/test/train split
# ------------------------------------------------------------
def predict_pairprobe_result_on_split(pair_result, split="test"):
    """
    Return unstandardized predicted [channel, pair] labels for a saved
    pair-probe training result on one of its stored datasets.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be 'train', 'val', or 'test'.")

    ds_key = f"ds_{split}"
    if ds_key not in pair_result:
        raise KeyError(f"pair_result does not contain {ds_key}.")

    ds = pair_result[ds_key]
    model = pair_result["model"]
    X_s = apply_standardizer(ds["X"], pair_result["X_mean"], pair_result["X_std"])

    model.eval()
    with torch.no_grad():
        pred_s = model(torch.tensor(X_s, dtype=torch.float32).to(device)).cpu().numpy()

    pred_aug = invert_standardizer(pred_s, pair_result["Y_mean"], pair_result["Y_std"])
    pred_ch, pred_pair = split_channel_pair(pred_aug)

    return {
        "Y_pred_aug": pred_aug.astype(np.float32),
        "Y_pred_channel": pred_ch.astype(np.float32),
        "Y_pred_pair": pred_pair.astype(np.float32),
        "Y_true_channel": ds["Y_channel"].astype(np.float32),
        "Y_true_pair": ds["Y_pair"].astype(np.float32),
        "dataset": ds,
    }


# ------------------------------------------------------------
# B. Direct QAOA-aware loss: differentiable proxy
# ------------------------------------------------------------
def build_qaoa_aware_edge_coeffs_tensor(
    stride_gamma=5,
    stride_beta=5,
    max_points=64,
    seed=0,
):
    """
    Select a compact set of QAOA grid points and return ideal edge coefficient
    vectors c_ideal_edge for the 16-element edge Pauli basis.

    Shape returned: (n_probe_points, n_edges, 16)
    """
    candidates = []
    for ig in range(0, len(rho_ideal_grid_3q), int(stride_gamma)):
        for ib in range(0, len(rho_ideal_grid_3q[ig]), int(stride_beta)):
            candidates.append((ig, ib))

    rng = np.random.default_rng(seed)
    if len(candidates) > max_points:
        idx = rng.choice(len(candidates), size=int(max_points), replace=False)
        selected = [candidates[k] for k in sorted(idx)]
    else:
        selected = candidates

    coeffs = []
    for ig, ib in selected:
        rho = rho_ideal_grid_3q[ig][ib]
        edge_coeffs = []
        for e_idx in range(len(PAIR_EDGES_3Q)):
            edge_coeffs.append(edge_coeff_vector_16_from_rho_3q(rho, e_idx))
        coeffs.append(edge_coeffs)

    coeffs = np.asarray(coeffs, dtype=np.float32)
    return torch.tensor(coeffs, dtype=torch.float32).to(device), selected


def torch_local_ptm_from_channel_parts(A, b):
    """
    Convert batched affine Bloch maps r_out = A r_in + b into 4x4 Pauli
    transfer matrices in the basis [I, X, Y, Z].

    A shape: (B, N, 3, 3), b shape: (B, N, 3)
    returns R shape: (B, N, 4, 4)
    """
    B = A.shape[0]
    N = A.shape[1]
    R = torch.zeros((B, N, 4, 4), dtype=A.dtype, device=A.device)
    R[:, :, 0, 0] = 1.0
    R[:, :, 1:, 0] = b
    R[:, :, 1:, 1:] = A
    return R


def torch_batch_kron_4x4(Ri, Rj):
    """Batched Kronecker product for two batches of 4x4 matrices."""
    return torch.einsum("bij,bkl->bikjl", Ri, Rj).reshape(Ri.shape[0], 16, 16)


def torch_pair_process_blocks(y_pair):
    """Reshape flattened process-relative pair residuals to (B, n_edges, 9, 9)."""
    return y_pair.reshape(y_pair.shape[0], len(PAIR_EDGES_3Q), 9, 9)


def torch_edge_response_matrices_from_local_pair(y_channel, y_pair=None, eta_pair=1.0):
    """
    Batched differentiable version of edge_response_matrix_16_from_local_pair_3q.

    Returns shape (B, n_edges, 16, 16).
    """
    A, b = split_channels_torch_pair(y_channel)
    R_local = torch_local_ptm_from_channel_parts(A, b)

    blocks = None
    if y_pair is not None:
        blocks = torch_pair_process_blocks(y_pair)

    idx = torch.tensor(EDGE_NONTRIVIAL_PAIR_INDICES_16, dtype=torch.long, device=y_channel.device)
    R_edges = []

    for e_idx, (qi, qj) in enumerate(PAIR_EDGES_3Q):
        R_edge = torch_batch_kron_4x4(R_local[:, qi], R_local[:, qj])

        if blocks is not None:
            R_patch = R_edge[:, idx[:, None], idx[None, :]] + float(eta_pair) * blocks[:, e_idx]
            R_edge = R_edge.clone()
            R_edge[:, idx[:, None], idx[None, :]] = R_patch

        R_edges.append(R_edge)

    return torch.stack(R_edges, dim=1)


def qaoa_proxy_costs_from_augmented_labels_torch(y_aug_phys, edge_coeffs_t, eta_pair=1.0):
    """
    Compute differentiable model-predicted noisy QAOA costs from augmented
    local-plus-pair labels. This is used only as a training proxy.

    y_aug_phys: unstandardized labels, shape (B, LABEL_DIM + PAIR_LABEL_DIM)
    edge_coeffs_t: shape (P, n_edges, 16)
    returns costs: shape (B, P)
    """
    y_channel = y_aug_phys[:, :LABEL_DIM]
    y_pair = y_aug_phys[:, LABEL_DIM:LABEL_DIM + PAIR_LABEL_DIM]

    R_edges = torch_edge_response_matrices_from_local_pair(
        y_channel=y_channel,
        y_pair=y_pair,
        eta_pair=eta_pair,
    )

    zz_rows = R_edges[:, :, EDGE_ZZ_INDEX_16, :]  # (B, E, 16)
    zz_pred = torch.einsum("bei,pei->bpe", zz_rows, edge_coeffs_t)
    costs = 0.5 * (1.0 - zz_pred).sum(dim=-1)
    return costs


def qaoa_aware_loss_from_augmented_outputs(pred_aug_phys, target_aug_phys, edge_coeffs_t):
    """
    Direct QAOA-aware loss. It penalizes error in the predicted noisy MaxCut
    cost on representative QAOA probe states, not only error in channel/pair labels.
    """
    C_pred = qaoa_proxy_costs_from_augmented_labels_torch(pred_aug_phys, edge_coeffs_t)
    with torch.no_grad():
        C_target = qaoa_proxy_costs_from_augmented_labels_torch(target_aug_phys, edge_coeffs_t)
    return torch.mean((C_pred - C_target) ** 2)


# ------------------------------------------------------------
# C. Train a pair-probe model with direct QAOA-aware loss
# ------------------------------------------------------------
def train_final_pairprobe_K_qaoa_aware(
    K_pair,
    lambda_qaoa=0.2,
    qaoa_stride_gamma=5,
    qaoa_stride_beta=5,
    qaoa_max_points=64,
    seed=None,
    n_epochs=None,
):
    """
    Train the same PairProbeAwareNet3Q architecture, but add a direct
    QAOA-aware proxy loss to the usual channel + pair + physicality losses.

    The proxy loss compares predicted and reference noisy MaxCut costs on a
    compact set of QAOA probe states. This tests whether pair probes improve
    the target observable, not only residual-label MSE.
    """
    if seed is None:
        seed = int(FINAL_RUN_CONFIG.get("BASE_SEED", 12345)) + 9100 + int(K_pair)
    if n_epochs is None:
        n_epochs = int(FINAL_RUN_CONFIG.get("PAIR_EPOCHS", 300))

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("\n" + "=" * 100)
    print(f"PRA STEP 5 | QAOA-aware pair-probe training | K_pair = {K_pair}")
    print("=" * 100)

    ds_train = build_local_pairprobe_dataset(
        pair_dataset=final_pair_train,
        Xpair_probe=final_Xpair_train,
        K_pair=K_pair,
        seed=int(FINAL_RUN_CONFIG.get("BASE_SEED", 12345)) + 8000 + int(K_pair),
    )
    ds_val = build_local_pairprobe_dataset(
        pair_dataset=final_pair_val,
        Xpair_probe=final_Xpair_val,
        K_pair=K_pair,
        seed=int(FINAL_RUN_CONFIG.get("BASE_SEED", 12345)) + 8100 + int(K_pair),
    )
    ds_test = build_local_pairprobe_dataset(
        pair_dataset=final_pair_test,
        Xpair_probe=final_Xpair_test,
        K_pair=K_pair,
        seed=int(FINAL_RUN_CONFIG.get("BASE_SEED", 12345)) + 8200 + int(K_pair),
    )

    X_mean, X_std = compute_standardizer(ds_train["X"])
    Y_mean, Y_std = compute_standardizer(ds_train["Y_aug"])
    Y_ch_mean, Y_ch_std = compute_standardizer(ds_train["Y_channel"])

    X_train_s = apply_standardizer(ds_train["X"], X_mean, X_std)
    X_val_s = apply_standardizer(ds_val["X"], X_mean, X_std)
    X_test_s = apply_standardizer(ds_test["X"], X_mean, X_std)

    Y_train_s = apply_standardizer(ds_train["Y_aug"], Y_mean, Y_std)
    Y_val_s = apply_standardizer(ds_val["Y_aug"], Y_mean, Y_std)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_s, dtype=torch.float32),
            torch.tensor(Y_train_s, dtype=torch.float32),
        ),
        batch_size=int(FINAL_RUN_CONFIG.get("BATCH_SIZE", 32)),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val_s, dtype=torch.float32),
            torch.tensor(Y_val_s, dtype=torch.float32),
        ),
        batch_size=128,
        shuffle=False,
    )

    model = PairProbeAwareNet3Q(
        input_dim=ds_train["X"].shape[1],
        channel_dim=LABEL_DIM,
        pair_dim=PAIR_LABEL_DIM,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(FINAL_RUN_CONFIG.get("LR", 8e-4)),
        weight_decay=float(FINAL_RUN_CONFIG.get("WEIGHT_DECAY", 1e-5)),
    )
    mse = nn.MSELoss()

    Y_mean_t = torch.tensor(Y_mean, dtype=torch.float32).to(device)
    Y_std_t = torch.tensor(Y_std, dtype=torch.float32).to(device)
    Y_ch_mean_t = torch.tensor(Y_ch_mean, dtype=torch.float32).to(device)
    Y_ch_std_t = torch.tensor(Y_ch_std, dtype=torch.float32).to(device)

    qaoa_edge_coeffs_t, qaoa_probe_indices = build_qaoa_aware_edge_coeffs_tensor(
        stride_gamma=qaoa_stride_gamma,
        stride_beta=qaoa_stride_beta,
        max_points=qaoa_max_points,
        seed=seed + 17,
    )
    print("QAOA-aware probe points:", len(qaoa_probe_indices))

    def physicality_loss_pair_local(y_channel_s):
        y_phys = y_channel_s * Y_ch_std_t + Y_ch_mean_t
        A, b = split_channels_torch_pair(y_phys)
        r = bloch_test_t_pair[None, None, :, :]
        penalties = []
        for q in range(N_QUBITS):
            A_q = A[:, q, :, :]
            b_q = b[:, q, :]
            out = torch.matmul(r, A_q.transpose(1, 2)[:, None, :, :])
            out = out.squeeze(1) + b_q[:, None, :]
            norms = torch.linalg.norm(out, dim=-1)
            penalties.append((torch.relu(norms - 1.0) ** 2).mean())
        return sum(penalties) / len(penalties)

    history = {
        "train_channel_mse": [],
        "train_pair_mse": [],
        "train_qaoa_loss": [],
        "val_channel_mse": [],
        "val_pair_mse": [],
        "val_qaoa_loss": [],
        "val_phys": [],
    }

    best_score = np.inf
    best_state = None

    lambda_pair = float(FINAL_RUN_CONFIG.get("LAMBDA_PAIR", 1.0))
    lambda_phys = float(FINAL_RUN_CONFIG.get("LAMBDA_PHYS", 0.03))
    lambda_qaoa = float(lambda_qaoa)

    for epoch in range(1, int(n_epochs) + 1):
        model.train()
        tr_ch = tr_pair = tr_qaoa = 0.0
        n_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)

            pred_ch = pred[:, :LABEL_DIM]
            pred_pair = pred[:, LABEL_DIM:LABEL_DIM + PAIR_LABEL_DIM]
            y_ch = yb[:, :LABEL_DIM]
            y_pair = yb[:, LABEL_DIM:LABEL_DIM + PAIR_LABEL_DIM]

            ch_loss = mse(pred_ch, y_ch)
            pair_loss = mse(pred_pair, y_pair)
            phys_loss = physicality_loss_pair_local(pred_ch)

            pred_phys = pred * Y_std_t + Y_mean_t
            y_phys = yb * Y_std_t + Y_mean_t
            qaoa_loss = qaoa_aware_loss_from_augmented_outputs(
                pred_aug_phys=pred_phys,
                target_aug_phys=y_phys,
                edge_coeffs_t=qaoa_edge_coeffs_t,
            )

            loss = (
                ch_loss
                + lambda_pair * pair_loss
                + lambda_phys * phys_loss
                + lambda_qaoa * qaoa_loss
            )
            loss.backward()
            optimizer.step()

            bs = xb.shape[0]
            tr_ch += ch_loss.item() * bs
            tr_pair += pair_loss.item() * bs
            tr_qaoa += qaoa_loss.item() * bs
            n_total += bs

        model.eval()
        val_ch = val_pair = val_phys = val_qaoa = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)

                pred_ch = pred[:, :LABEL_DIM]
                pred_pair = pred[:, LABEL_DIM:LABEL_DIM + PAIR_LABEL_DIM]
                y_ch = yb[:, :LABEL_DIM]
                y_pair = yb[:, LABEL_DIM:LABEL_DIM + PAIR_LABEL_DIM]

                ch_loss = mse(pred_ch, y_ch)
                pair_loss = mse(pred_pair, y_pair)
                phys_loss = physicality_loss_pair_local(pred_ch)
                qaoa_loss = qaoa_aware_loss_from_augmented_outputs(
                    pred_aug_phys=pred * Y_std_t + Y_mean_t,
                    target_aug_phys=yb * Y_std_t + Y_mean_t,
                    edge_coeffs_t=qaoa_edge_coeffs_t,
                )

                bs = xb.shape[0]
                val_ch += ch_loss.item() * bs
                val_pair += pair_loss.item() * bs
                val_phys += phys_loss.item() * bs
                val_qaoa += qaoa_loss.item() * bs
                val_n += bs

        val_ch /= val_n
        val_pair /= val_n
        val_phys /= val_n
        val_qaoa /= val_n

        history["train_channel_mse"].append(tr_ch / n_total)
        history["train_pair_mse"].append(tr_pair / n_total)
        history["train_qaoa_loss"].append(tr_qaoa / n_total)
        history["val_channel_mse"].append(val_ch)
        history["val_pair_mse"].append(val_pair)
        history["val_qaoa_loss"].append(val_qaoa)
        history["val_phys"].append(val_phys)

        score = val_ch + lambda_pair * val_pair + lambda_qaoa * val_qaoa
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"Epoch {epoch:4d} | val ch {val_ch:.4e} | "
                f"val pair {val_pair:.4e} | val QAOA {val_qaoa:.4e} | "
                f"val phys {val_phys:.4e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_s = model(X_test_t).cpu().numpy()

    pred_aug = invert_standardizer(pred_s, Y_mean, Y_std)
    pred_ch, pred_pair = split_channel_pair(pred_aug)

    return {
        "K_pair": int(K_pair),
        "model": model,
        "history": history,
        "ds_train": ds_train,
        "ds_val": ds_val,
        "ds_test": ds_test,
        "X_mean": X_mean,
        "X_std": X_std,
        "Y_mean": Y_mean,
        "Y_std": Y_std,
        "Y_ch_mean": Y_ch_mean,
        "Y_ch_std": Y_ch_std,
        "Y_pred_aug": pred_aug.astype(np.float32),
        "Y_pred_channel": pred_ch.astype(np.float32),
        "Y_pred_pair": pred_pair.astype(np.float32),
        "best_score": float(best_score),
        "lambda_qaoa": lambda_qaoa,
        "qaoa_probe_indices": qaoa_probe_indices,
        "training_variant": "direct_qaoa_aware_loss",
    }


# ------------------------------------------------------------
# D. Non-oracle ablation evaluator with per-method eta and lambda
# ------------------------------------------------------------
def evaluate_qaoa_pair_device_nonoracle_ablation_3q(
    idx_device,
    Y_ch_true_all,
    Y_pair_true_all,
    prediction_methods,
    qaoa_shots=2048,
    seed=0,
    default_lam=1e-3,
):
    """
    Evaluate several non-oracle correction variants on the same finite-shot
    target-edge measurements for one hidden device.

    prediction_methods format:
        {
          "method name": {
              "Y_channel": array[n_devices, LABEL_DIM],
              "Y_pair": array[n_devices, PAIR_LABEL_DIM] or None,
              "eta": float,
              "lam": optional float,
          }
        }
    """
    y_ch_true = Y_ch_true_all[idx_device]
    y_pair_true = Y_pair_true_all[idx_device]

    edge_meas, edge_exact = generate_target_edge_measurements_3q(
        y_channel_true=y_ch_true,
        y_pair_true=y_pair_true,
        rho_ideal_grid=rho_ideal_grid_3q,
        shots=qaoa_shots,
        seed=seed + idx_device,
        eta_pair_true=1.0,
    )

    C_noisy_edge = cost_landscape_from_edge_coeff_measurements_3q(edge_meas)

    results = {
        "Noisy edge measured": {
            "C": C_noisy_edge,
            "edge_measurements": edge_meas,
            "error": landscape_error_metrics_3q(C_noisy_edge, C_ideal_3q),
            "selected": selected_point_metrics(C_noisy_edge, C_ideal_3q, Popt_ideal_3q),
            "uses_oracle_correction": False,
            "eta_pair_model": np.nan,
        }
    }

    for method_name, pack in prediction_methods.items():
        y_ch_model = pack["Y_channel"][idx_device]
        y_pair_all = pack.get("Y_pair", None)
        y_pair_model = None if y_pair_all is None else y_pair_all[idx_device]
        eta = float(pack.get("eta", 1.0))
        lam = float(pack.get("lam", default_lam))

        C_corr, corrected_edges = corrected_landscape_from_edge_observable_estimator_3q(
            edge_measurements=edge_meas,
            y_channel_model=y_ch_model,
            y_pair_model=y_pair_model,
            lam=lam,
            eta_pair_model=eta,
            clip=True,
        )

        results[method_name] = {
            "C": C_corr,
            "corrected_edges": corrected_edges,
            "error": landscape_error_metrics_3q(C_corr, C_ideal_3q),
            "selected": selected_point_metrics(C_corr, C_ideal_3q, Popt_ideal_3q),
            "uses_oracle_correction": False,
            "eta_pair_model": eta,
            "lam": lam,
        }

    return results


def summarize_ablation_results_list_3q(results_list):
    """Summarize ablation results over hidden devices."""
    methods = list(results_list[0].keys())
    noisy_mae = np.mean([r["Noisy edge measured"]["error"]["MAE"] for r in results_list])

    rows = []
    for method in methods:
        mae = np.array([r[method]["error"]["MAE"] for r in results_list], dtype=np.float64)
        rmse = np.array([r[method]["error"]["RMSE"] for r in results_list], dtype=np.float64)
        regret = np.array([r[method]["selected"]["regret"] for r in results_list], dtype=np.float64)
        popt = np.array([r[method]["selected"]["p_opt_selected"] for r in results_list], dtype=np.float64)
        disp = np.array([r[method]["selected"].get("parameter_displacement", np.nan) for r in results_list], dtype=np.float64)
        etas = np.array([r[method].get("eta_pair_model", np.nan) for r in results_list], dtype=np.float64)

        rows.append({
            "Method": method,
            "n_devices": int(len(mae)),
            "Mean MAE": float(np.mean(mae)),
            "Std MAE": float(np.std(mae, ddof=1)) if len(mae) > 1 else 0.0,
            "Mean RMSE": float(np.mean(rmse)),
            "Mean regret": float(np.mean(regret)),
            "Mean P_opt": float(np.mean(popt)),
            "Mean parameter displacement": float(np.nanmean(disp)),
            "Mean eta": float(np.nanmean(etas)) if np.isfinite(etas).any() else np.nan,
            "Improvement ratio": float(noisy_mae / (np.mean(mae) + 1e-12)),
            "Oracle correction used": False,
        })

    return pd.DataFrame(rows)


def run_nonoracle_ablation_3q(
    Y_ch_true,
    Y_pair_true,
    prediction_methods,
    n_devices=20,
    qaoa_shots=2048,
    seed=99000,
    default_lam=1e-3,
):
    """Run non-oracle ablation over a subset of hidden devices."""
    n_eval = min(int(n_devices), len(Y_ch_true))
    results = []
    for idx in range(n_eval):
        results.append(
            evaluate_qaoa_pair_device_nonoracle_ablation_3q(
                idx_device=idx,
                Y_ch_true_all=Y_ch_true,
                Y_pair_true_all=Y_pair_true,
                prediction_methods=prediction_methods,
                qaoa_shots=qaoa_shots,
                seed=seed,
                default_lam=default_lam,
            )
        )
    table = summarize_ablation_results_list_3q(results)
    return results, table


# ------------------------------------------------------------
# E. Select eta on validation devices and evaluate on test devices
# ------------------------------------------------------------
def select_eta_on_validation_3q(
    Y_ch_true_val,
    Y_pair_true_val,
    Y_pred_channel_val,
    Y_pred_pair_val,
    eta_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
    n_val_devices=10,
    qaoa_shots=2048,
    seed=88000,
    lam=1e-3,
):
    """
    Choose shrinkage eta by minimizing validation MAE of the non-oracle
    edge-observable estimator.
    """
    rows = []
    tables = {}

    for eta in eta_grid:
        methods = {
            f"pair residual eta={eta:.2f}": {
                "Y_channel": Y_pred_channel_val,
                "Y_pair": Y_pred_pair_val,
                "eta": float(eta),
                "lam": lam,
            }
        }
        _, table = run_nonoracle_ablation_3q(
            Y_ch_true=Y_ch_true_val,
            Y_pair_true=Y_pair_true_val,
            prediction_methods=methods,
            n_devices=n_val_devices,
            qaoa_shots=qaoa_shots,
            seed=seed + int(round(float(eta) * 1000)),
            default_lam=lam,
        )
        tables[float(eta)] = table
        row = table[table["Method"].str.startswith("pair residual")].iloc[0].to_dict()
        row["eta_candidate"] = float(eta)
        rows.append(row)

    eta_table = pd.DataFrame(rows).sort_values("Mean MAE", ascending=True).reset_index(drop=True)
    eta_star = float(eta_table.loc[0, "eta_candidate"])

    print("Selected eta by validation MAE:", eta_star)
    return eta_star, eta_table, tables


def run_step5_ablation_for_pair_result(
    pair_result,
    qaoa_aware_result=None,
    eta_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
    n_val_devices=10,
    n_test_devices=20,
    qaoa_shots=2048,
    lam=1e-3,
    seed=99000,
    include_oracle_upper_bound=True,
):
    """
    Full Step 5 workflow for a trained pair-probe result.

    Returns validation eta table and test ablation table comparing:
        - local-only non-oracle correction,
        - local-plus-pair correction with validation-selected eta,
        - optional QAOA-aware-loss model,
        - optional true-response upper bound.
    """
    val_pred = predict_pairprobe_result_on_split(pair_result, split="val")
    test_pred = predict_pairprobe_result_on_split(pair_result, split="test")

    eta_star, eta_table, eta_tables_by_value = select_eta_on_validation_3q(
        Y_ch_true_val=val_pred["Y_true_channel"],
        Y_pair_true_val=val_pred["Y_true_pair"],
        Y_pred_channel_val=val_pred["Y_pred_channel"],
        Y_pred_pair_val=val_pred["Y_pred_pair"],
        eta_grid=eta_grid,
        n_val_devices=n_val_devices,
        qaoa_shots=qaoa_shots,
        seed=seed + 1000,
        lam=lam,
    )

    methods = {
        "local-only correction": {
            "Y_channel": test_pred["Y_pred_channel"],
            "Y_pair": None,
            "eta": 0.0,
            "lam": lam,
        },
        f"local+pair correction eta={eta_star:.2f}": {
            "Y_channel": test_pred["Y_pred_channel"],
            "Y_pair": test_pred["Y_pred_pair"],
            "eta": eta_star,
            "lam": lam,
        },
    }

    qaoa_aware_test_pred = None
    if qaoa_aware_result is not None:
        qaoa_aware_test_pred = predict_pairprobe_result_on_split(qaoa_aware_result, split="test")
        methods[f"direct QAOA-aware loss eta={eta_star:.2f}"] = {
            "Y_channel": qaoa_aware_test_pred["Y_pred_channel"],
            "Y_pair": qaoa_aware_test_pred["Y_pred_pair"],
            "eta": eta_star,
            "lam": lam,
        }

    if include_oracle_upper_bound:
        methods["true response upper bound"] = {
            "Y_channel": test_pred["Y_true_channel"],
            "Y_pair": test_pred["Y_true_pair"],
            "eta": 1.0,
            "lam": lam,
        }

    test_results, test_table = run_nonoracle_ablation_3q(
        Y_ch_true=test_pred["Y_true_channel"],
        Y_pair_true=test_pred["Y_true_pair"],
        prediction_methods=methods,
        n_devices=n_test_devices,
        qaoa_shots=qaoa_shots,
        seed=seed + 2000,
        default_lam=lam,
    )

    return {
        "eta_star": eta_star,
        "eta_table": eta_table,
        "eta_tables_by_value": eta_tables_by_value,
        "test_results": test_results,
        "test_table": test_table,
        "test_prediction_standard": test_pred,
        "test_prediction_qaoa_aware": qaoa_aware_test_pred,
    }


def save_step5_ablation_outputs(step5_output, prefix="pra_step5_ablation"):
    """Save Step 5 ablation tables to FINAL_DIR if available."""
    out_dir = globals().get("FINAL_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)

    eta_path = os.path.join(out_dir, f"{prefix}_eta_validation.csv")
    test_path = os.path.join(out_dir, f"{prefix}_test_table.csv")

    step5_output["eta_table"].to_csv(eta_path, index=False)
    step5_output["test_table"].to_csv(test_path, index=False)

    print("Saved:", eta_path)
    print("Saved:", test_path)
    return eta_path, test_path


# ------------------------------------------------------------
# F. Example run block
# ------------------------------------------------------------
# Recommended first run:
#
# K_PAIR_STEP5 = 18
#
# # 1) Use the already trained standard pair-probe model.
# pair_result_step5 = final_pairprobe_results[K_PAIR_STEP5]
#
# # 2) Train the QAOA-aware-loss variant. This is the expensive new part.
# qaoa_aware_pair_result_step5 = train_final_pairprobe_K_qaoa_aware(
#     K_pair=K_PAIR_STEP5,
#     lambda_qaoa=0.2,
#     qaoa_stride_gamma=5,
#     qaoa_stride_beta=5,
#     qaoa_max_points=64,
#     n_epochs=FINAL_RUN_CONFIG["PAIR_EPOCHS"],
# )
#
# # 3) Run the validation eta selection and final test ablation.
# step5_ablation_output = run_step5_ablation_for_pair_result(
#     pair_result=pair_result_step5,
#     qaoa_aware_result=qaoa_aware_pair_result_step5,
#     eta_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
#     n_val_devices=10,
#     n_test_devices=20,
#     qaoa_shots=2048,
#     lam=1e-3,
#     seed=99000,
# )
#
# display(step5_ablation_output["eta_table"])
# display(step5_ablation_output["test_table"])
# save_step5_ablation_outputs(step5_ablation_output, prefix=f"pra_step5_Kpair{K_PAIR_STEP5}")

print("PRA Step 5 ablation code is ready. Run the example block after final_pairprobe_results are available.")

# %% Cell 61
# ============================================================
# FINAL PRA DASHBOARD — run all final revised outputs
# ============================================================
# This cell assumes the previous cells have been run from a clean runtime.
# It produces the final non-oracle ablation, confidence-interval tables,
# and a zipped output package in FINAL_DIR.

import os, json, pickle, zipfile, glob, time
from IPython.display import display

print("="*100)
print("FINAL PRA CONSISTENCY CHECK")
print("="*100)
print("Device:", device)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Hard checks for all required revisions.
assert "leakage_eval_table_3q" in globals(), "Step 1 leakage results are missing."
assert PAIR_LABEL_DIM == len(PAIR_EDGES_3Q) * 9 * 9, f"Step 2 broken: PAIR_LABEL_DIM={PAIR_LABEL_DIM}, expected {len(PAIR_EDGES_3Q)*81}."
assert globals().get("PAIR_RESIDUAL_KIND", None) == "process_relative_edge_response_9x9_per_edge", "Step 2 residual kind is not process-relative."
assert "corrected_landscape_from_edge_observable_estimator_3q" in globals(), "Step 3 non-oracle edge estimator missing."
assert "summarize_long_qaoa_ci" in globals(), "Step 4 confidence-interval functions missing."
assert "run_step5_ablation_for_pair_result" in globals(), "Step 5 ablation functions missing."
assert "final_pairprobe_results" in globals(), "Final pair-probe models/results missing. Run final pair-probe cells first."
assert 18 in final_pairprobe_results, "K_pair=18 model missing from final_pairprobe_results."

revision_check = pd.DataFrame([
    {"Revision": 1, "Requirement": "Leakage observable", "Status": "included", "Implementation": "per-transmon mean/max leakage labels"},
    {"Revision": 2, "Requirement": "Process-relative pair residual", "Status": "included", "Implementation": "2 edges × 9 outputs × 9 inputs = 162 labels"},
    {"Revision": 3, "Requirement": "Non-oracle Eq. replacement", "Status": "included", "Implementation": "target-edge observable estimator with ridge inverse"},
    {"Revision": 4, "Requirement": "Confidence intervals", "Status": "included", "Implementation": "hidden devices, finite-shot seeds, NN seeds"},
    {"Revision": 5, "Requirement": "Operational ablation", "Status": "included", "Implementation": "local-only, pair shrinkage eta, QAOA-aware loss"},
    {"Revision": 6, "Requirement": "Deployability framing", "Status": "included", "Implementation": "leakage-explicit diagnostics + non-oracle target-circuit protocol"},
])
revision_check_path = os.path.join(FINAL_DIR, "FINAL_revision_checklist.csv")
revision_check.to_csv(revision_check_path, index=False)
display(revision_check)

# ------------------------------------------------------------
# Step 4A: hidden-device confidence intervals from existing results
# ------------------------------------------------------------
print("\nBuilding hidden-device confidence intervals from existing final local QAOA results...")
df_hidden_long, df_hidden_ci, df_hidden_imp = build_hidden_device_ci_tables_from_existing_results()
display(df_hidden_ci.head(20))
display(df_hidden_imp.head(20))

# ------------------------------------------------------------
# Step 4B: finite-shot seed repeats for the non-oracle pair estimator
# ------------------------------------------------------------
# Keep these modest by default so the final cell completes in Colab.
# Increase N_STEP4_SHOT_DEVICES and SHOT_SEEDS_FINAL for stronger final statistics.
N_STEP4_SHOT_DEVICES = min(20, len(final_pair_test["Y_channel"]))
SHOT_SEEDS_FINAL = (101, 202, 303, 404, 505)
print(f"\nRunning non-oracle pair finite-shot seed repeats over {N_STEP4_SHOT_DEVICES} devices and seeds {SHOT_SEEDS_FINAL}...")
pair_prediction_methods_step4 = {
    "Kpair=18 local+pair non-oracle": {
        "Y_channel": final_pairprobe_results[18]["Y_pred_channel"],
        "Y_pair": final_pairprobe_results[18]["Y_pred_pair"],
    }
}
df_pair_shot_long, df_pair_shot_ci = run_pair_nonoracle_finite_shot_seed_repeats(
    prediction_methods=pair_prediction_methods_step4,
    shot_seeds=SHOT_SEEDS_FINAL,
    n_devices=N_STEP4_SHOT_DEVICES,
    lam=1e-3,
    eta_pair_model=1.0,
)
display(df_pair_shot_ci)

# ------------------------------------------------------------
# Step 4C: neural-network seed repeats for local ChannelNet
# ------------------------------------------------------------
# This is expensive because it retrains ChannelNet. It is enabled by default for PRA completeness.
# Reduce NN_REPEAT_EPOCHS for a quick test; use FINAL_RUN_CONFIG["LOCAL_EPOCHS"] for the final paper run.
NN_REPEAT_K_LIST = (18, 54)
NN_REPEAT_SEEDS = (111, 222, 333)
NN_REPEAT_EPOCHS = FINAL_RUN_CONFIG["LOCAL_EPOCHS"]
print(f"\nRunning NN-seed repeats for K={NN_REPEAT_K_LIST}, seeds={NN_REPEAT_SEEDS}, epochs={NN_REPEAT_EPOCHS}...")
nn_packs = run_local_channelnet_nn_seed_repeats(
    K_list=NN_REPEAT_K_LIST,
    nn_seeds=NN_REPEAT_SEEDS,
    n_epochs=NN_REPEAT_EPOCHS,
    n_qaoa_devices=min(20, FINAL_RUN_CONFIG["N_QAOA_LOCAL_EVAL"]),
)
df_nn_channel_long, df_nn_channel_ci, df_nn_qaoa_long, df_nn_qaoa_ci, trained_nn_packs = nn_packs
display(df_nn_channel_ci)
display(df_nn_qaoa_ci)

# ------------------------------------------------------------
# Step 5: final operational ablation with QAOA-aware loss
# ------------------------------------------------------------
K_PAIR_STEP5 = 18
qaoa_aware_path = os.path.join(FINAL_DIR, f"pra_step5_qaoa_aware_pair_result_Kpair{K_PAIR_STEP5}.pkl")
step5_output_path = os.path.join(FINAL_DIR, f"pra_step5_ablation_output_Kpair{K_PAIR_STEP5}.pkl")

if os.path.exists(qaoa_aware_path):
    print("\nLoading cached QAOA-aware pair-probe model:", qaoa_aware_path)
    with open(qaoa_aware_path, "rb") as f:
        qaoa_aware_pair_result_step5 = pickle.load(f)
else:
    print("\nTraining QAOA-aware pair-probe model for Step 5 ablation...")
    qaoa_aware_pair_result_step5 = train_final_pairprobe_K_qaoa_aware(
        K_pair=K_PAIR_STEP5,
        lambda_qaoa=0.2,
        qaoa_stride_gamma=5,
        qaoa_stride_beta=5,
        qaoa_max_points=64,
        n_epochs=FINAL_RUN_CONFIG["PAIR_EPOCHS"],
    )
    with open(qaoa_aware_path, "wb") as f:
        pickle.dump(qaoa_aware_pair_result_step5, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved:", qaoa_aware_path)

if os.path.exists(step5_output_path):
    print("Loading cached Step 5 ablation output:", step5_output_path)
    with open(step5_output_path, "rb") as f:
        step5_ablation_output = pickle.load(f)
else:
    print("Running Step 5 validation eta selection and test ablation...")
    step5_ablation_output = run_step5_ablation_for_pair_result(
        pair_result=final_pairprobe_results[K_PAIR_STEP5],
        qaoa_aware_result=qaoa_aware_pair_result_step5,
        eta_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
        n_val_devices=min(20, len(final_pair_val["Y_channel"])),
        n_test_devices=min(30, len(final_pair_test["Y_channel"])),
        qaoa_shots=2048,
        lam=1e-3,
        seed=99000,
    )
    with open(step5_output_path, "wb") as f:
        pickle.dump(step5_ablation_output, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved:", step5_output_path)

eta_csv, step5_csv = save_step5_ablation_outputs(step5_ablation_output, prefix=f"pra_step5_Kpair{K_PAIR_STEP5}")
print("Selected eta:", step5_ablation_output["eta_star"])
display(step5_ablation_output["eta_table"])
display(step5_ablation_output["test_table"])

# ------------------------------------------------------------
# Final master tables
# ------------------------------------------------------------
final_tables = {
    "revision_checklist": revision_check,
    "leakage_eval_table": leakage_eval_table_3q,
    "final_local_channel_table": final_local_channel_table,
    "final_local_qaoa_table_oracle_diagnostic": final_local_qaoa_table,
    "final_pairprobe_eval_table": final_pairprobe_eval_table,
    "final_pair_qaoa_table_oracle_diagnostic": final_pair_qaoa_table,
    "hidden_device_ci": df_hidden_ci,
    "hidden_device_improvement_ci": df_hidden_imp,
    "pair_nonoracle_shot_ci": df_pair_shot_ci,
    "nn_seed_channel_ci": df_nn_channel_ci,
    "nn_seed_qaoa_ci": df_nn_qaoa_ci,
    "step5_eta_validation": step5_ablation_output["eta_table"],
    "step5_test_ablation_nonoracle": step5_ablation_output["test_table"],
}

for name, df in final_tables.items():
    if isinstance(df, pd.DataFrame):
        path = os.path.join(FINAL_DIR, f"FINAL_{name}.csv")
        df.to_csv(path, index=False)
        print("Saved:", path)

summary_json = {
    "final_dir": FINAL_DIR,
    "device": str(device),
    "cuda_available": bool(torch.cuda.is_available()),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "pair_label_dim": int(PAIR_LABEL_DIM),
    "pair_residual_kind": str(PAIR_RESIDUAL_KIND),
    "selected_eta_step5": float(step5_ablation_output["eta_star"]),
    "run_config": FINAL_RUN_CONFIG,
}
summary_json_path = os.path.join(FINAL_DIR, "FINAL_run_summary.json")
with open(summary_json_path, "w") as f:
    json.dump(summary_json, f, indent=2)
print("Saved:", summary_json_path)

# Zip final output directory for download.
zip_path = os.path.join(FINAL_DIR, "FINAL_PRA_revised_outputs.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(FINAL_DIR):
        for fn in files:
            full = os.path.join(root, fn)
            if full == zip_path:
                continue
            arc = os.path.relpath(full, FINAL_DIR)
            zf.write(full, arc)
print("\nFinal output zip:", zip_path)
print("FINAL PRA RUN COMPLETE.")
