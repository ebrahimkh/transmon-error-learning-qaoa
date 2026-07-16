# %% [markdown]
# # Two-qubit proof-of-concept code
#
# This note
#If you use this code in academic work, please cite the associated manuscript:

#Ebrahim Khaleghian and Özgür E. Müstecaplıoğlu,
#"Physics-Informed Learning of Effective Error Processes from Limited Noisy
#Transmon Measurements for Robust QAOA Reliability", 2026.
# %% Cell 2
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

def dagger(A):
    return A.conj().T

def kron(*ops):
    """Kronecker product of multiple operators/vectors."""
    out = np.array([[1.0 + 0.0j]])
    for op in ops:
        out = np.kron(out, op)
    return out

def basis(dim, idx):
    """Column basis vector |idx> in dimension dim."""
    v = np.zeros((dim, 1), dtype=complex)
    v[idx, 0] = 1.0
    return v

def fro_norm(A):
    return np.linalg.norm(A, ord="fro")

# %% Cell 3
# Single-transmon/qutrit dimension
d = 3
I3 = np.eye(d, dtype=complex)

# Qutrit annihilation operator
# a|0> = 0, a|1> = |0>, a|2> = sqrt(2)|1>
a = np.zeros((d, d), dtype=complex)
for n in range(1, d):
    a[n - 1, n] = np.sqrt(n)

adag = dagger(a)
n_op = adag @ a

# Two-transmon operators
a1 = kron(a, I3)
a2 = kron(I3, a)

adag1 = dagger(a1)
adag2 = dagger(a2)

n1 = adag1 @ a1
n2 = adag2 @ a2

I9 = np.eye(d * d, dtype=complex)

print("Single transmon dimension:", d)
print("Two-transmon Hilbert-space dimension:", I9.shape[0])
print("Single-qutrit annihilation operator a:")
print(a)

# %% Cell 4
def two_qutrit_basis(i, j):
    """Basis |i,j> for two qutrits."""
    return kron(basis(d, i), basis(d, j))

# Computational basis states inside 9D space
comp_states = {
    "00": two_qutrit_basis(0, 0),
    "01": two_qutrit_basis(0, 1),
    "10": two_qutrit_basis(1, 0),
    "11": two_qutrit_basis(1, 1),
}

# Projector onto computational subspace
P_comp = sum(v @ dagger(v) for v in comp_states.values())

def op_from_terms(terms):
    """
    Build operator from terms:
    (ket_label, bra_label, coefficient)
    """
    O = np.zeros((d * d, d * d), dtype=complex)
    for ket_label, bra_label, coeff in terms:
        O += coeff * comp_states[ket_label] @ dagger(comp_states[bra_label])
    return O

# Pauli operators on qubit 1, embedded in two-qutrit space
X1 = op_from_terms([
    ("10", "00", 1), ("00", "10", 1),
    ("11", "01", 1), ("01", "11", 1)
])

Y1 = op_from_terms([
    ("10", "00", 1j), ("00", "10", -1j),
    ("11", "01", 1j), ("01", "11", -1j)
])

Z1 = op_from_terms([
    ("00", "00", 1), ("01", "01", 1),
    ("10", "10", -1), ("11", "11", -1)
])

# Pauli operators on qubit 2, embedded in two-qutrit space
X2 = op_from_terms([
    ("01", "00", 1), ("00", "01", 1),
    ("11", "10", 1), ("10", "11", 1)
])

Y2 = op_from_terms([
    ("01", "00", 1j), ("00", "01", -1j),
    ("11", "10", 1j), ("10", "11", -1j)
])

Z2 = op_from_terms([
    ("00", "00", 1), ("10", "10", 1),
    ("01", "01", -1), ("11", "11", -1)
])

# %% Cell 5
checks = {
    "Tr(P_comp), should be 4": np.trace(P_comp).real,
    "P_comp Hermiticity error": fro_norm(P_comp - dagger(P_comp)),
    "P_comp idempotency error": fro_norm(P_comp @ P_comp - P_comp),

    "X1^2 - P_comp": fro_norm(X1 @ X1 - P_comp),
    "Y1^2 - P_comp": fro_norm(Y1 @ Y1 - P_comp),
    "Z1^2 - P_comp": fro_norm(Z1 @ Z1 - P_comp),

    "X2^2 - P_comp": fro_norm(X2 @ X2 - P_comp),
    "Y2^2 - P_comp": fro_norm(Y2 @ Y2 - P_comp),
    "Z2^2 - P_comp": fro_norm(Z2 @ Z2 - P_comp),

    "[X1,Y1] - 2iZ1": fro_norm(X1 @ Y1 - Y1 @ X1 - 2j * Z1),
    "[X2,Y2] - 2iZ2": fro_norm(X2 @ Y2 - Y2 @ X2 - 2j * Z2),
}

for name, value in checks.items():
    print(f"{name:35s}: {value}")

# %% Cell 6
from scipy.linalg import expm

# %% Cell 7
def two_pi(x):
    return 2 * np.pi * x

# Hidden device parameters
# These are known to the simulator, but later the ML model will NOT see them.
device_params = {
    # Detunings in rad/ns
    "Delta1": two_pi(0.000),   # qubit 1 close to rotating frame
    "Delta2": two_pi(0.150),   # qubit 2 spectator detuned from qubit 1 drive

    # Anharmonicities in rad/ns
    # transmon |1> -> |2> transition is shifted by -alpha
    "alpha1": two_pi(0.220),
    "alpha2": two_pi(0.230),

    # Coupling strength in rad/ns
    "g": two_pi(0.003),

    # Small effective ZZ-like shift in rad/ns
    "zeta": two_pi(0.0008),

    # Hidden pulse calibration imperfections
    "amp_scale_1": 1.00,
    "amp_scale_2": 1.00,
    "phase_error_1": 0.00,
    "phase_error_2": 0.00,
}

# %% Cell 8
def build_H0(params):
    Delta1 = params["Delta1"]
    Delta2 = params["Delta2"]
    alpha1 = params["alpha1"]
    alpha2 = params["alpha2"]
    g = params["g"]
    zeta = params["zeta"]

    H1 = Delta1 * n1 - 0.5 * alpha1 * (n1 @ (n1 - I9))
    H2 = Delta2 * n2 - 0.5 * alpha2 * (n2 @ (n2 - I9))

    H_coupling = g * (adag1 @ a2 + a1 @ adag2)
    H_zz = zeta * (n1 @ n2)

    H0 = H1 + H2 + H_coupling + H_zz
    return H0

H0 = build_H0(device_params)

print("H0 Hermiticity error:", fro_norm(H0 - dagger(H0)))
print("H0 shape:", H0.shape)

# %% Cell 9
def control_hamiltonian(params, qubit=1, Omega=0.0, phase=0.0):
    """
    Square microwave drive on qubit 1 or 2.

    Omega: nominal drive amplitude in rad/ns
    phase: drive phase
    """
    if qubit == 1:
        aq = a1
        adagq = adag1
        amp_scale = params["amp_scale_1"]
        phase_error = params["phase_error_1"]
    elif qubit == 2:
        aq = a2
        adagq = adag2
        amp_scale = params["amp_scale_2"]
        phase_error = params["phase_error_2"]
    else:
        raise ValueError("qubit must be 1 or 2")

    effective_phase = phase + phase_error
    effective_Omega = amp_scale * Omega

    Hx = 0.5 * effective_Omega * np.cos(effective_phase) * (aq + adagq)
    Hy = 0.5 * effective_Omega * np.sin(effective_phase) * (1j * (adagq - aq))

    return Hx + Hy

# %% Cell 10
def evolve_state_square_pulse(psi0, params, qubit=1, theta=np.pi, phase=0.0, T=40.0, n_steps=200):
    """
    Evolve under a square pulse with area theta.

    For an ideal two-level system:
        theta = Omega * T

    theta = pi gives an X_pi pulse.
    theta = pi/2 gives an X_pi/2 pulse.
    """
    H0 = build_H0(params)

    Omega = theta / T
    Hc = control_hamiltonian(params, qubit=qubit, Omega=Omega, phase=phase)

    dt = T / n_steps
    U_step = expm(-1j * (H0 + Hc) * dt)

    psi = psi0.copy()
    states = [psi.copy()]

    for _ in range(n_steps):
        psi = U_step @ psi
        psi = psi / np.linalg.norm(psi)
        states.append(psi.copy())

    times = np.linspace(0, T, n_steps + 1)
    return times, states

def population(psi, label):
    v = comp_states[label]
    return np.abs((dagger(v) @ psi)[0, 0])**2

def leakage_population(psi):
    """
    Population outside the computational subspace.
    """
    p_comp = np.real((dagger(psi) @ P_comp @ psi)[0, 0])
    return max(0.0, 1.0 - p_comp)

def populations_over_time(states):
    pops = {
        "00": [],
        "01": [],
        "10": [],
        "11": [],
        "leak": [],
    }

    for psi in states:
        for label in ["00", "01", "10", "11"]:
            pops[label].append(population(psi, label))
        pops["leak"].append(leakage_population(psi))

    return pops

# %% Cell 11
psi00 = comp_states["00"]

times_idle, states_idle = evolve_state_square_pulse(
    psi0=psi00,
    params=device_params,
    qubit=1,
    theta=0.0,
    phase=0.0,
    T=40.0,
    n_steps=200
)

pops_idle = populations_over_time(states_idle)

print("Final idle populations:")
for key in pops_idle:
    print(f"{key}: {pops_idle[key][-1]:.6f}")

# %% Cell 12
times_xpi, states_xpi = evolve_state_square_pulse(
    psi0=psi00,
    params=device_params,
    qubit=1,
    theta=np.pi,
    phase=0.0,
    T=40.0,
    n_steps=200
)

pops_xpi = populations_over_time(states_xpi)

print("Final X_pi populations:")
for key in pops_xpi:
    print(f"{key}: {pops_xpi[key][-1]:.6f}")

# %% Cell 13
plt.figure(figsize=(8, 5))

for key in ["00", "10", "01", "11", "leak"]:
    plt.plot(times_xpi, pops_xpi[key], label=key)

plt.xlabel("Time (ns)")
plt.ylabel("Population")
plt.title("Closed-system pulse test: X_pi drive on transmon 1")
plt.legend()
plt.grid(True)
plt.show()

# %% Cell 14
times_xhalf, states_xhalf = evolve_state_square_pulse(
    psi0=psi00,
    params=device_params,
    qubit=1,
    theta=np.pi/2,
    phase=0.0,
    T=40.0,
    n_steps=200
)

pops_xhalf = populations_over_time(states_xhalf)

print("Final X_pi/2 populations:")
for key in pops_xhalf:
    print(f"{key}: {pops_xhalf[key][-1]:.6f}")

# %% Cell 15
def ket_to_rho(psi):
    """Convert state vector |psi> to density matrix rho."""
    return psi @ dagger(psi)

def trace_expectation(rho, O):
    """Compute real expectation Tr(rho O)."""
    val = np.trace(rho @ O)
    return float(np.real_if_close(val))

def normalize_ket(psi):
    return psi / np.linalg.norm(psi)

def bloch_vector_from_rho_qubit(rho, qubit=1):
    """
    Return local Bloch vector (x,y,z) for qubit 1 or qubit 2
    using embedded Pauli operators.
    """
    if qubit == 1:
        return np.array([
            trace_expectation(rho, X1),
            trace_expectation(rho, Y1),
            trace_expectation(rho, Z1)
        ])
    elif qubit == 2:
        return np.array([
            trace_expectation(rho, X2),
            trace_expectation(rho, Y2),
            trace_expectation(rho, Z2)
        ])
    else:
        raise ValueError("qubit must be 1 or 2")

# %% Cell 16
# Single-transmon/qutrit states using only |0>, |1> computational levels
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

def two_transmon_product_state(q1_state, q2_state):
    """Two-transmon product state |q1_state> ⊗ |q2_state>."""
    return kron(q1_state, q2_state)

# %% Cell 17
def prepare_local_tomography_state(target_qubit, local_label, spectator_label="0"):
    """
    Prepare a two-transmon state where one qubit is in local_label
    and the other qubit is in spectator_label.
    """
    local = local_state_dict[local_label]
    spectator = local_state_dict[spectator_label]

    if target_qubit == 1:
        return two_transmon_product_state(local, spectator)
    elif target_qubit == 2:
        return two_transmon_product_state(spectator, local)
    else:
        raise ValueError("target_qubit must be 1 or 2")

# %% Cell 18
measurement_ops = {
    1: {
        "X": X1,
        "Y": Y1,
        "Z": Z1,
    },
    2: {
        "X": X2,
        "Y": Y2,
        "Z": Z2,
    }
}

tomography_input_labels = ["0", "1", "+", "-", "+i", "-i"]
measurement_labels = ["X", "Y", "Z"]

# %% Cell 19
expected_bloch = {
    "0":  np.array([0.0, 0.0, 1.0]),
    "1":  np.array([0.0, 0.0, -1.0]),
    "+":  np.array([1.0, 0.0, 0.0]),
    "-":  np.array([-1.0, 0.0, 0.0]),
    "+i": np.array([0.0, 1.0, 0.0]),
    "-i": np.array([0.0, -1.0, 0.0]),
}

for target_qubit in [1, 2]:
    print(f"\nLocal tomography states for qubit {target_qubit}")
    print("-" * 50)

    for label in tomography_input_labels:
        psi = prepare_local_tomography_state(
            target_qubit=target_qubit,
            local_label=label,
            spectator_label="0"
        )
        rho = ket_to_rho(psi)
        r = bloch_vector_from_rho_qubit(rho, qubit=target_qubit)

        err = np.linalg.norm(r - expected_bloch[label])

        print(
            f"state {label:>2s}: "
            f"Bloch = [{r[0]: .3f}, {r[1]: .3f}, {r[2]: .3f}], "
            f"error = {err:.2e}"
        )

# %% Cell 20
def sample_hidden_device(rng=None):
    """
    Sample one hidden two-transmon device.

    Units:
    - frequencies: rad/ns
    - T1, Tphi: ns

    These parameters define the hidden data generator.
    The learning model will not receive them.
    """
    if rng is None:
        rng = np.random.default_rng()

    params = {}

    # Detuning drift around rotating frame
    params["Delta1"] = two_pi(rng.normal(0.000, 0.004))
    params["Delta2"] = two_pi(rng.normal(0.150, 0.006))

    # Anharmonicities
    params["alpha1"] = two_pi(rng.normal(0.220, 0.008))
    params["alpha2"] = two_pi(rng.normal(0.230, 0.008))

    # Coupling and small ZZ shift
    params["g"] = two_pi(rng.normal(0.0030, 0.0005))
    params["zeta"] = two_pi(rng.normal(0.0008, 0.0002))

    # Pulse calibration imperfections
    params["amp_scale_1"] = rng.normal(1.00, 0.025)
    params["amp_scale_2"] = rng.normal(1.00, 0.025)

    params["phase_error_1"] = rng.normal(0.0, 0.025)
    params["phase_error_2"] = rng.normal(0.0, 0.025)

    # Decoherence parameters, used in later steps
    params["T1_1"] = rng.uniform(25000.0, 60000.0)
    params["T1_2"] = rng.uniform(25000.0, 60000.0)

    params["Tphi_1"] = rng.uniform(20000.0, 50000.0)
    params["Tphi_2"] = rng.uniform(20000.0, 50000.0)

    # Readout assignment errors
    # r01 = P(measure 1 | true 0)
    # r10 = P(measure 0 | true 1)
    params["r01_1"] = rng.uniform(0.005, 0.040)
    params["r10_1"] = rng.uniform(0.005, 0.050)

    params["r01_2"] = rng.uniform(0.005, 0.040)
    params["r10_2"] = rng.uniform(0.005, 0.050)

    return params

# %% Cell 21
rng = np.random.default_rng(42)

for k in range(3):
    p = sample_hidden_device(rng)
    print(f"\nHidden device {k+1}")
    print("-" * 40)
    print(f"Delta1 / 2pi = {p['Delta1']/(2*np.pi): .5f} GHz")
    print(f"Delta2 / 2pi = {p['Delta2']/(2*np.pi): .5f} GHz")
    print(f"alpha1 / 2pi = {p['alpha1']/(2*np.pi): .5f} GHz")
    print(f"alpha2 / 2pi = {p['alpha2']/(2*np.pi): .5f} GHz")
    print(f"g      / 2pi = {p['g']/(2*np.pi): .5f} GHz")
    print(f"zeta   / 2pi = {p['zeta']/(2*np.pi): .5f} GHz")
    print(f"amp_scale_1  = {p['amp_scale_1']: .4f}")
    print(f"phase_err_1  = {p['phase_error_1']: .4f} rad")
    print(f"T1_1         = {p['T1_1']: .1f} ns")
    print(f"Tphi_1       = {p['Tphi_1']: .1f} ns")
    print(f"readout q1   = r01 {p['r01_1']: .3f}, r10 {p['r10_1']: .3f}")

# %% Cell 22
rng = np.random.default_rng(123)

max_herm_error = 0.0

for _ in range(20):
    p = sample_hidden_device(rng)
    H0_random = build_H0(p)
    err = fro_norm(H0_random - dagger(H0_random))
    max_herm_error = max(max_herm_error, err)

print("Maximum H0 Hermiticity error over 20 random devices:", max_herm_error)

# %% Cell 23
def vec(rho):
    """
    Column-stacking vectorization.
    """
    return rho.reshape((-1, 1), order="F")

def unvec(v, dim):
    """
    Inverse of column-stacking vectorization.
    """
    return v.reshape((dim, dim), order="F")

def clean_density_matrix(rho):
    """
    Numerical cleanup: enforce Hermiticity and trace normalization.
    """
    rho = 0.5 * (rho + dagger(rho))
    tr = np.trace(rho)
    if abs(tr) > 1e-12:
        rho = rho / tr
    return rho

# %% Cell 24
def collapse_operators(params):
    """
    Build collapse operators for two-transmon open-system evolution.
    Units: T1 and Tphi are in ns, so rates are 1/ns.
    """
    cops = []

    # T1 relaxation
    gamma1_1 = 1.0 / params["T1_1"]
    gamma1_2 = 1.0 / params["T1_2"]

    cops.append(np.sqrt(gamma1_1) * a1)
    cops.append(np.sqrt(gamma1_2) * a2)

    # Pure dephasing
    gammaphi_1 = 1.0 / params["Tphi_1"]
    gammaphi_2 = 1.0 / params["Tphi_2"]

    cops.append(np.sqrt(gammaphi_1) * n1)
    cops.append(np.sqrt(gammaphi_2) * n2)

    return cops

# %% Cell 25
def liouvillian(H, cops):
    """
    Build Liouvillian superoperator L such that:
        d vec(rho) / dt = L vec(rho)
    """
    dim = H.shape[0]
    I = np.eye(dim, dtype=complex)

    # Hamiltonian part
    L = -1j * (np.kron(I, H) - np.kron(H.T, I))

    # Dissipators
    for C in cops:
        CdC = dagger(C) @ C

        # C rho C^\dagger
        L += np.kron(C.conj(), C)

        # -1/2 C^\dagger C rho
        L += -0.5 * np.kron(I, CdC)

        # -1/2 rho C^\dagger C
        L += -0.5 * np.kron(CdC.T, I)

    return L

# %% Cell 26
def evolve_rho_square_pulse(
    rho0,
    params,
    qubit=1,
    theta=np.pi,
    phase=0.0,
    T=40.0,
    n_steps=1,
    include_decoherence=True
):
    """
    Evolve density matrix under a square pulse.

    For this step, H is time-independent during the square pulse,
    so one matrix exponential is enough. n_steps is kept for later flexibility.
    """
    H0 = build_H0(params)
    Omega = theta / T
    Hc = control_hamiltonian(params, qubit=qubit, Omega=Omega, phase=phase)
    H = H0 + Hc

    if include_decoherence:
        cops = collapse_operators(params)
        L = liouvillian(H, cops)
        propagator = expm(L * T)

        rho_vec_final = propagator @ vec(rho0)
        rho_final = unvec(rho_vec_final, rho0.shape[0])
        rho_final = clean_density_matrix(rho_final)

    else:
        U = expm(-1j * H * T)
        rho_final = U @ rho0 @ dagger(U)
        rho_final = clean_density_matrix(rho_final)

    return rho_final

# %% Cell 27
def density_leakage_population(rho):
    """
    Population outside computational subspace.
    """
    p_comp = np.real(np.trace(P_comp @ rho))
    return max(0.0, 1.0 - p_comp)

def computational_populations_rho(rho):
    pops = {}
    for label, ket in comp_states.items():
        P = ket @ dagger(ket)
        pops[label] = float(np.real_if_close(np.trace(P @ rho)))
    pops["leak"] = density_leakage_population(rho)
    return pops

# %% Cell 28
rng = np.random.default_rng(2026)
test_params = sample_hidden_device(rng)

rho00 = ket_to_rho(comp_states["00"])

rho_xpi_noisy = evolve_rho_square_pulse(
    rho0=rho00,
    params=test_params,
    qubit=1,
    theta=np.pi,
    phase=0.0,
    T=40.0,
    include_decoherence=True
)

pops = computational_populations_rho(rho_xpi_noisy)

print("Trace:", np.trace(rho_xpi_noisy))
print("Hermiticity error:", fro_norm(rho_xpi_noisy - dagger(rho_xpi_noisy)))
print("\nNoisy X_pi final populations:")
for k, v in pops.items():
    print(f"{k}: {v:.6f}")

# %% Cell 29
def ideal_measurement_probabilities(rho, qubit=1, basis_label="Z"):
    """
    Return ideal probabilities for measuring +1 and -1
    for X, Y, or Z on the selected qubit.

    Leakage reduces the magnitude of the embedded Pauli expectation.
    """
    O = measurement_ops[qubit][basis_label]
    m = trace_expectation(rho, O)

    # Numerical clipping
    m = float(np.clip(m, -1.0, 1.0))

    p_plus = 0.5 * (1.0 + m)
    p_minus = 0.5 * (1.0 - m)

    return p_plus, p_minus, m

# %% Cell 30
def apply_readout_error(p_plus, p_minus, params, qubit=1):
    """
    Apply binary readout assignment error.

    + outcome is treated as 0-like.
    - outcome is treated as 1-like.
    """
    if qubit == 1:
        r01 = params["r01_1"]
        r10 = params["r10_1"]
    elif qubit == 2:
        r01 = params["r01_2"]
        r10 = params["r10_2"]
    else:
        raise ValueError("qubit must be 1 or 2")

    p_plus_meas = (1.0 - r01) * p_plus + r10 * p_minus
    p_minus_meas = r01 * p_plus + (1.0 - r10) * p_minus

    # Numerical cleanup
    p_plus_meas = float(np.clip(p_plus_meas, 0.0, 1.0))
    p_minus_meas = float(np.clip(p_minus_meas, 0.0, 1.0))

    s = p_plus_meas + p_minus_meas
    p_plus_meas /= s
    p_minus_meas /= s

    return p_plus_meas, p_minus_meas

# %% Cell 31
def sample_binary_measurement(
    rho,
    params,
    qubit=1,
    basis_label="Z",
    shots=1024,
    rng=None
):
    """
    Measure local X/Y/Z observable with readout error and finite shots.
    Return estimated expectation value and counts.
    """
    if rng is None:
        rng = np.random.default_rng()

    p_plus, p_minus, m_ideal = ideal_measurement_probabilities(
        rho, qubit=qubit, basis_label=basis_label
    )

    p_plus_meas, p_minus_meas = apply_readout_error(
        p_plus, p_minus, params, qubit=qubit
    )

    counts = rng.multinomial(shots, [p_plus_meas, p_minus_meas])
    n_plus, n_minus = counts

    m_hat = (n_plus - n_minus) / shots

    return {
        "m_hat": float(m_hat),
        "m_ideal_no_readout": float(m_ideal),
        "p_plus_meas": float(p_plus_meas),
        "p_minus_meas": float(p_minus_meas),
        "n_plus": int(n_plus),
        "n_minus": int(n_minus),
    }

# %% Cell 32
rng = np.random.default_rng(1)

for basis_label in ["X", "Y", "Z"]:
    result = sample_binary_measurement(
        rho_xpi_noisy,
        test_params,
        qubit=1,
        basis_label=basis_label,
        shots=1024,
        rng=rng
    )

    print(f"\nMeasurement basis {basis_label} on qubit 1")
    print(f"ideal expectation before readout error: {result['m_ideal_no_readout']:.4f}")
    print(f"measured probs: p+={result['p_plus_meas']:.4f}, p-={result['p_minus_meas']:.4f}")
    print(f"counts: n+={result['n_plus']}, n-={result['n_minus']}")
    print(f"finite-shot m_hat: {result['m_hat']:.4f}")

# %% Cell 33
def generate_local_tomography_vector(
    params,
    shots=1024,
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    rng=None,
    include_decoherence=True
):
    """
    Generate finite-shot local tomography data.

    For each target qubit:
      - prepare local states on target
      - spectator starts in |0>
      - apply noisy pulse to target
      - measure X,Y,Z on target

    Returns:
      feature vector of length 36
    """
    if rng is None:
        rng = np.random.default_rng()

    features = []

    for target_qubit in [1, 2]:
        for state_label in tomography_input_labels:
            psi0 = prepare_local_tomography_state(
                target_qubit=target_qubit,
                local_label=state_label,
                spectator_label="0"
            )
            rho0 = ket_to_rho(psi0)

            rho_out = evolve_rho_square_pulse(
                rho0=rho0,
                params=params,
                qubit=target_qubit,
                theta=pulse_theta,
                phase=pulse_phase,
                T=pulse_T,
                include_decoherence=include_decoherence
            )

            for basis_label in measurement_labels:
                meas = sample_binary_measurement(
                    rho_out,
                    params,
                    qubit=target_qubit,
                    basis_label=basis_label,
                    shots=shots,
                    rng=rng
                )
                features.append(meas["m_hat"])

    return np.array(features, dtype=np.float32)

# %% Cell 34
rng = np.random.default_rng(11)
test_params = sample_hidden_device(rng)

x_tomo = generate_local_tomography_vector(
    params=test_params,
    shots=1024,
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    rng=rng,
    include_decoherence=True
)

print("Tomography vector shape:", x_tomo.shape)
print("First 10 entries:")
print(x_tomo[:10])
print("Minimum value:", x_tomo.min())
print("Maximum value:", x_tomo.max())

# %% Cell 35
plt.figure(figsize=(10, 4))
plt.plot(x_tomo, marker="o")
plt.axhline(1.0, linestyle="--")
plt.axhline(-1.0, linestyle="--")
plt.xlabel("Tomography feature index")
plt.ylabel("Finite-shot expectation")
plt.title("One finite-shot local tomography vector")
plt.grid(True)
plt.show()

# %% Cell 36
def exact_measured_expectation_with_readout(
    rho,
    params,
    qubit=1,
    basis_label="Z"
):
    """
    Exact measured expectation value after readout error,
    but before finite-shot sampling.

    Returns:
        m_meas = p_plus_meas - p_minus_meas
    """
    p_plus, p_minus, m_pre_readout = ideal_measurement_probabilities(
        rho,
        qubit=qubit,
        basis_label=basis_label
    )

    p_plus_meas, p_minus_meas = apply_readout_error(
        p_plus,
        p_minus,
        params,
        qubit=qubit
    )

    m_meas = p_plus_meas - p_minus_meas

    return float(m_meas)

# %% Cell 37
def exact_output_bloch_for_local_input(
    params,
    target_qubit=1,
    local_label="0",
    spectator_label="0",
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    include_decoherence=True,
    include_readout=True
):
    """
    Exact local output Bloch vector after noisy pulse.

    If include_readout=True:
        output includes readout assignment error.
    If include_readout=False:
        output is the quantum expectation before readout error.
    """
    psi0 = prepare_local_tomography_state(
        target_qubit=target_qubit,
        local_label=local_label,
        spectator_label=spectator_label
    )
    rho0 = ket_to_rho(psi0)

    rho_out = evolve_rho_square_pulse(
        rho0=rho0,
        params=params,
        qubit=target_qubit,
        theta=pulse_theta,
        phase=pulse_phase,
        T=pulse_T,
        include_decoherence=include_decoherence
    )

    r_out = []

    for basis_label in measurement_labels:
        if include_readout:
            m = exact_measured_expectation_with_readout(
                rho_out,
                params,
                qubit=target_qubit,
                basis_label=basis_label
            )
        else:
            _, _, m = ideal_measurement_probabilities(
                rho_out,
                qubit=target_qubit,
                basis_label=basis_label
            )

        r_out.append(m)

    return np.array(r_out, dtype=np.float64)

# %% Cell 38
def fit_affine_bloch_map(r_in_list, r_out_list):
    """
    Fit r_out = A r_in + b.

    r_in_list:  shape (N, 3)
    r_out_list: shape (N, 3)

    Returns:
        A: shape (3, 3)
        b: shape (3,)
    """
    R_in = np.asarray(r_in_list, dtype=np.float64)
    R_out = np.asarray(r_out_list, dtype=np.float64)

    # Design matrix: [x, y, z, 1]
    X_design = np.hstack([R_in, np.ones((R_in.shape[0], 1))])

    # Solve X_design @ W = R_out
    # W shape: (4, 3)
    W, residuals, rank, s = np.linalg.lstsq(X_design, R_out, rcond=None)

    A = W[:3, :].T
    b = W[3, :]

    return A, b

# %% Cell 39
def reference_local_affine_channel(
    params,
    target_qubit=1,
    spectator_label="0",
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    include_decoherence=True,
    include_readout=True
):
    """
    Compute reference effective affine Bloch map for one target qubit.

    Returns:
        A, b
    """
    r_in_list = []
    r_out_list = []

    for local_label in tomography_input_labels:
        r_in = expected_bloch[local_label]

        r_out = exact_output_bloch_for_local_input(
            params=params,
            target_qubit=target_qubit,
            local_label=local_label,
            spectator_label=spectator_label,
            pulse_theta=pulse_theta,
            pulse_phase=pulse_phase,
            pulse_T=pulse_T,
            include_decoherence=include_decoherence,
            include_readout=include_readout
        )

        r_in_list.append(r_in)
        r_out_list.append(r_out)

    A, b = fit_affine_bloch_map(r_in_list, r_out_list)

    return A, b

# %% Cell 40
def flatten_two_local_channels(A1, b1, A2, b2):
    """
    Flatten two local affine channels into one vector of length 24.
    """
    return np.concatenate([
        A1.reshape(-1),
        b1.reshape(-1),
        A2.reshape(-1),
        b2.reshape(-1)
    ]).astype(np.float32)

def unflatten_two_local_channels(y):
    """
    Convert length-24 vector back into:
        A1, b1, A2, b2
    """
    y = np.asarray(y)

    A1 = y[0:9].reshape(3, 3)
    b1 = y[9:12]

    A2 = y[12:21].reshape(3, 3)
    b2 = y[21:24]

    return A1, b1, A2, b2

# %% Cell 41
def reference_two_local_channels_label(
    params,
    spectator_label="0",
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    include_decoherence=True,
    include_readout=True
):
    """
    Compute full length-24 reference label:
        [A1, b1, A2, b2]
    """
    A1_ref, b1_ref = reference_local_affine_channel(
        params=params,
        target_qubit=1,
        spectator_label=spectator_label,
        pulse_theta=pulse_theta,
        pulse_phase=pulse_phase,
        pulse_T=pulse_T,
        include_decoherence=include_decoherence,
        include_readout=include_readout
    )

    A2_ref, b2_ref = reference_local_affine_channel(
        params=params,
        target_qubit=2,
        spectator_label=spectator_label,
        pulse_theta=pulse_theta,
        pulse_phase=pulse_phase,
        pulse_T=pulse_T,
        include_decoherence=include_decoherence,
        include_readout=include_readout
    )

    y = flatten_two_local_channels(A1_ref, b1_ref, A2_ref, b2_ref)

    return y

# %% Cell 42
rng = np.random.default_rng(1234)
test_params = sample_hidden_device(rng)

y_ref = reference_two_local_channels_label(
    params=test_params,
    spectator_label="0",
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    include_decoherence=True,
    include_readout=True
)

print("Reference label shape:", y_ref.shape)
print("First 10 entries:")
print(y_ref[:10])

A1_ref, b1_ref, A2_ref, b2_ref = unflatten_two_local_channels(y_ref)

print("\nA1 reference:")
print(A1_ref)

print("\nb1 reference:")
print(b1_ref)

print("\nA2 reference:")
print(A2_ref)

print("\nb2 reference:")
print(b2_ref)

# %% Cell 43
def check_affine_map_fit(
    params,
    target_qubit=1,
    spectator_label="0",
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    include_decoherence=True,
    include_readout=True
):
    A, b = reference_local_affine_channel(
        params=params,
        target_qubit=target_qubit,
        spectator_label=spectator_label,
        pulse_theta=pulse_theta,
        pulse_phase=pulse_phase,
        pulse_T=pulse_T,
        include_decoherence=include_decoherence,
        include_readout=include_readout
    )

    print(f"\nAffine map check for qubit {target_qubit}")
    print("-" * 70)

    max_err = 0.0

    for local_label in tomography_input_labels:
        r_in = expected_bloch[local_label]

        r_exact = exact_output_bloch_for_local_input(
            params=params,
            target_qubit=target_qubit,
            local_label=local_label,
            spectator_label=spectator_label,
            pulse_theta=pulse_theta,
            pulse_phase=pulse_phase,
            pulse_T=pulse_T,
            include_decoherence=include_decoherence,
            include_readout=include_readout
        )

        r_pred = A @ r_in + b
        err = np.linalg.norm(r_pred - r_exact)
        max_err = max(max_err, err)

        print(
            f"input {local_label:>2s} | "
            f"exact [{r_exact[0]: .3f}, {r_exact[1]: .3f}, {r_exact[2]: .3f}] | "
            f"affine [{r_pred[0]: .3f}, {r_pred[1]: .3f}, {r_pred[2]: .3f}] | "
            f"err {err:.2e}"
        )

    print("Maximum fit error:", max_err)
    return max_err

# %% Cell 44
err_q1 = check_affine_map_fit(test_params, target_qubit=1)
err_q2 = check_affine_map_fit(test_params, target_qubit=2)

# %% Cell 45
rng = np.random.default_rng(2027)
test_params = sample_hidden_device(rng)

x_tomo_128 = generate_local_tomography_vector(
    params=test_params,
    shots=128,
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    rng=rng,
    include_decoherence=True
)

x_tomo_4096 = generate_local_tomography_vector(
    params=test_params,
    shots=4096,
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    rng=rng,
    include_decoherence=True
)

y_ref = reference_two_local_channels_label(
    params=test_params,
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    include_decoherence=True,
    include_readout=True
)

print("Finite-shot tomography vector, 128 shots:")
print(x_tomo_128[:12])

print("\nFinite-shot tomography vector, 4096 shots:")
print(x_tomo_4096[:12])

print("\nReference channel label:")
print(y_ref[:12])

# %% Cell 46
plt.figure(figsize=(10, 4))

plt.plot(x_tomo_128, marker="o", label="128 shots")
plt.plot(x_tomo_4096, marker="s", label="4096 shots", alpha=0.8)

plt.axhline(1.0, linestyle="--", linewidth=1)
plt.axhline(-1.0, linestyle="--", linewidth=1)

plt.xlabel("Tomography feature index")
plt.ylabel("Measured expectation")
plt.title("Finite-shot local tomography input: shot-noise comparison")
plt.legend()
plt.grid(True)
plt.show()

# %% Cell 47
def generate_one_supervised_example(
    rng,
    shots=1024,
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    include_decoherence=True,
    include_readout=True
):
    """
    Generate one supervised example.

    Hidden device:
        randomized transmon parameters.

    Input x:
        finite-shot local tomography vector, shape (36,)

    Label y:
        exact effective local affine channels, shape (24,)
    """
    params = sample_hidden_device(rng)

    x = generate_local_tomography_vector(
        params=params,
        shots=shots,
        pulse_theta=pulse_theta,
        pulse_phase=pulse_phase,
        pulse_T=pulse_T,
        rng=rng,
        include_decoherence=include_decoherence
    )

    y = reference_two_local_channels_label(
        params=params,
        spectator_label="0",
        pulse_theta=pulse_theta,
        pulse_phase=pulse_phase,
        pulse_T=pulse_T,
        include_decoherence=include_decoherence,
        include_readout=include_readout
    )

    return x.astype(np.float32), y.astype(np.float32), params

# %% Cell 48
import time

def build_dataset(
    n_examples,
    seed=0,
    shots=1024,
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    include_decoherence=True,
    include_readout=True,
    print_every=10
):
    """
    Build synthetic dataset:
        X shape: (n_examples, 36)
        Y shape: (n_examples, 24)

    Also returns hidden_params_list only for analysis/debug.
    The neural network will not use hidden_params_list.
    """
    rng = np.random.default_rng(seed)

    X = []
    Y = []
    hidden_params_list = []

    t0 = time.time()

    for i in range(n_examples):
        x, y, params = generate_one_supervised_example(
            rng=rng,
            shots=shots,
            pulse_theta=pulse_theta,
            pulse_phase=pulse_phase,
            pulse_T=pulse_T,
            include_decoherence=include_decoherence,
            include_readout=include_readout
        )

        X.append(x)
        Y.append(y)
        hidden_params_list.append(params)

        if (i + 1) % print_every == 0 or (i + 1) == n_examples:
            elapsed = time.time() - t0
            print(f"Generated {i+1:4d}/{n_examples} examples | elapsed {elapsed:.1f} s")

    X = np.stack(X).astype(np.float32)
    Y = np.stack(Y).astype(np.float32)

    return X, Y, hidden_params_list

# %% Cell 49
N_train = 60
N_val = 20
N_test = 20

shots_dataset = 1024

X_train, Y_train, params_train = build_dataset(
    n_examples=N_train,
    seed=100,
    shots=shots_dataset,
    print_every=10
)

X_val, Y_val, params_val = build_dataset(
    n_examples=N_val,
    seed=200,
    shots=shots_dataset,
    print_every=10
)

X_test, Y_test, params_test = build_dataset(
    n_examples=N_test,
    seed=300,
    shots=shots_dataset,
    print_every=10
)

# %% Cell 50
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

print("X_val shape:", X_val.shape)
print("Y_val shape:", Y_val.shape)

print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

print("\nX range:")
print("min:", X_train.min(), "max:", X_train.max())

print("\nY range:")
print("min:", Y_train.min(), "max:", Y_train.max())

# %% Cell 51
idx = 0

print("One input tomography vector x:")
print(X_train[idx])

print("\nOne target channel label y:")
print(Y_train[idx])

A1, b1, A2, b2 = unflatten_two_local_channels(Y_train[idx])

print("\nA1:")
print(A1)

print("\nb1:")
print(b1)

print("\nA2:")
print(A2)

print("\nb2:")
print(b2)

# %% Cell 52
plt.figure(figsize=(8, 4))
plt.hist(Y_train.flatten(), bins=40)
plt.xlabel("Target channel parameter value")
plt.ylabel("Count")
plt.title("Distribution of reference local-channel parameters")
plt.grid(True)
plt.show()

# %% Cell 53
feature_std = X_train.std(axis=0)
label_std = Y_train.std(axis=0)

print("Mean feature std:", feature_std.mean())
print("Max feature std:", feature_std.max())
print("Min feature std:", feature_std.min())

print("\nMean label std:", label_std.mean())
print("Max label std:", label_std.max())
print("Min label std:", label_std.min())

# %% Cell 54
i, j = 0, 1

plt.figure(figsize=(10, 4))
plt.plot(X_train[i], marker="o", label=f"example {i}")
plt.plot(X_train[j], marker="s", label=f"example {j}", alpha=0.8)
plt.xlabel("Tomography feature index")
plt.ylabel("Finite-shot expectation")
plt.title("Two different finite-shot tomography inputs")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(Y_train[i], marker="o", label=f"example {i}")
plt.plot(Y_train[j], marker="s", label=f"example {j}", alpha=0.8)
plt.xlabel("Channel-label index")
plt.ylabel("Reference channel parameter")
plt.title("Two different reference local-channel labels")
plt.legend()
plt.grid(True)
plt.show()

# %% Cell 55
def compute_standardizer(X, eps=1e-8):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return mean.astype(np.float32), std.astype(np.float32)

def apply_standardizer(X, mean, std):
    return ((X - mean) / std).astype(np.float32)

def invert_standardizer(X_std, mean, std):
    return (X_std * std + mean).astype(np.float32)

# %% Cell 56
X_mean, X_std = compute_standardizer(X_train)
Y_mean, Y_std = compute_standardizer(Y_train)

X_train_s = apply_standardizer(X_train, X_mean, X_std)
X_val_s = apply_standardizer(X_val, X_mean, X_std)
X_test_s = apply_standardizer(X_test, X_mean, X_std)

Y_train_s = apply_standardizer(Y_train, Y_mean, Y_std)
Y_val_s = apply_standardizer(Y_val, Y_mean, Y_std)
Y_test_s = apply_standardizer(Y_test, Y_mean, Y_std)

print("Standardized X_train mean:", X_train_s.mean())
print("Standardized X_train std:", X_train_s.std())

print("Standardized Y_train mean:", Y_train_s.mean())
print("Standardized Y_train std:", Y_train_s.std())

# %% Cell 57
np.savez(
    "channelnet_debug_dataset.npz",
    X_train=X_train,
    Y_train=Y_train,
    X_val=X_val,
    Y_val=Y_val,
    X_test=X_test,
    Y_test=Y_test,
    X_mean=X_mean,
    X_std=X_std,
    Y_mean=Y_mean,
    Y_std=Y_std,
    shots_dataset=shots_dataset
)

print("Saved dataset to channelnet_debug_dataset.npz")

# %% Cell 58
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %% Cell 59
X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train_s, dtype=torch.float32)

X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
Y_val_t = torch.tensor(Y_val_s, dtype=torch.float32)

X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test_s, dtype=torch.float32)

train_dataset = TensorDataset(X_train_t, Y_train_t)
val_dataset = TensorDataset(X_val_t, Y_val_t)
test_dataset = TensorDataset(X_test_t, Y_test_t)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Training tensor shape:", X_train_t.shape, Y_train_t.shape)
print("Validation tensor shape:", X_val_t.shape, Y_val_t.shape)
print("Test tensor shape:", X_test_t.shape, Y_test_t.shape)

# %% Cell 60
class ChannelNet(nn.Module):
    """
    MLP mapping finite-shot local tomography vector to
    two local affine Bloch channels:
        [A1, b1, A2, b2]
    """
    def __init__(self, input_dim=36, output_dim=24):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


model = ChannelNet(input_dim=36, output_dim=24).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print("Number of trainable parameters:", n_params)

# %% Cell 61
# Store Y standardizer as torch tensors
Y_mean_t = torch.tensor(Y_mean, dtype=torch.float32).to(device)
Y_std_t = torch.tensor(Y_std, dtype=torch.float32).to(device)

def unstandardize_y_torch(y_s):
    """
    Convert standardized network output back to physical channel parameters.
    """
    return y_s * Y_std_t + Y_mean_t


def split_channels_torch(y_phys):
    """
    y_phys: shape (batch, 24)

    Returns:
        A1: (batch, 3, 3)
        b1: (batch, 3)
        A2: (batch, 3, 3)
        b2: (batch, 3)
    """
    A1 = y_phys[:, 0:9].reshape(-1, 3, 3)
    b1 = y_phys[:, 9:12]

    A2 = y_phys[:, 12:21].reshape(-1, 3, 3)
    b2 = y_phys[:, 21:24]

    return A1, b1, A2, b2


# Test Bloch vectors: axes and a few diagonals
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

# Normalize diagonal vectors to lie on Bloch sphere
for i in range(6, len(bloch_test_np)):
    bloch_test_np[i] /= np.linalg.norm(bloch_test_np[i])

bloch_test = torch.tensor(bloch_test_np, dtype=torch.float32).to(device)


def physicality_loss_from_standardized_output(y_pred_s):
    """
    Penalize predicted local channels that map test Bloch vectors outside the Bloch ball.
    """
    y_phys = unstandardize_y_torch(y_pred_s)
    A1, b1, A2, b2 = split_channels_torch(y_phys)

    # bloch_test shape: (K, 3)
    # output shape should be: (batch, K, 3)
    r = bloch_test.unsqueeze(0)  # (1, K, 3)

    out1 = torch.matmul(r, A1.transpose(1, 2)) + b1.unsqueeze(1)
    out2 = torch.matmul(r, A2.transpose(1, 2)) + b2.unsqueeze(1)

    norm1 = torch.linalg.norm(out1, dim=-1)
    norm2 = torch.linalg.norm(out2, dim=-1)

    penalty1 = torch.relu(norm1 - 1.0) ** 2
    penalty2 = torch.relu(norm2 - 1.0) ** 2

    return penalty1.mean() + penalty2.mean()

# %% Cell 62
mse_loss = nn.MSELoss()

def evaluate_model(model, loader, lambda_phys=0.0):
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_phys = 0.0
    n_total = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)

            mse = mse_loss(pred, yb)
            phys = physicality_loss_from_standardized_output(pred)
            loss = mse + lambda_phys * phys

            batch_size = xb.shape[0]
            total_loss += loss.item() * batch_size
            total_mse += mse.item() * batch_size
            total_phys += phys.item() * batch_size
            n_total += batch_size

    return {
        "loss": total_loss / n_total,
        "mse": total_mse / n_total,
        "phys": total_phys / n_total,
    }


def train_channelnet(
    model,
    train_loader,
    val_loader,
    n_epochs=300,
    lr=1e-3,
    weight_decay=1e-5,
    lambda_phys=0.01
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    history = {
        "train_loss": [],
        "train_mse": [],
        "train_phys": [],
        "val_loss": [],
        "val_mse": [],
        "val_phys": [],
    }

    best_val = np.inf
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()

        total_loss = 0.0
        total_mse = 0.0
        total_phys = 0.0
        n_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)

            mse = mse_loss(pred, yb)
            phys = physicality_loss_from_standardized_output(pred)
            loss = mse + lambda_phys * phys

            loss.backward()
            optimizer.step()

            batch_size = xb.shape[0]
            total_loss += loss.item() * batch_size
            total_mse += mse.item() * batch_size
            total_phys += phys.item() * batch_size
            n_total += batch_size

        train_metrics = {
            "loss": total_loss / n_total,
            "mse": total_mse / n_total,
            "phys": total_phys / n_total,
        }

        val_metrics = evaluate_model(model, val_loader, lambda_phys=lambda_phys)

        history["train_loss"].append(train_metrics["loss"])
        history["train_mse"].append(train_metrics["mse"])
        history["train_phys"].append(train_metrics["phys"])

        history["val_loss"].append(val_metrics["loss"])
        history["val_mse"].append(val_metrics["mse"])
        history["val_phys"].append(val_metrics["phys"])

        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"train mse {train_metrics['mse']:.4e} | "
                f"val mse {val_metrics['mse']:.4e} | "
                f"val phys {val_metrics['phys']:.4e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history

# %% Cell 63
history = train_channelnet(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    n_epochs=300,
    lr=1e-3,
    weight_decay=1e-5,
    lambda_phys=0.01
)

# %% Cell 64
plt.figure(figsize=(8, 5))

plt.plot(history["train_mse"], label="train MSE")
plt.plot(history["val_mse"], label="validation MSE")

plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE on standardized channel labels")
plt.title("ChannelNet training curve")
plt.legend()
plt.grid(True)
plt.show()

# %% Cell 65
train_metrics = evaluate_model(model, train_loader, lambda_phys=0.01)
val_metrics = evaluate_model(model, val_loader, lambda_phys=0.01)
test_metrics = evaluate_model(model, test_loader, lambda_phys=0.01)

print("Train metrics:", train_metrics)
print("Validation metrics:", val_metrics)
print("Test metrics:", test_metrics)

# %% Cell 66
def mse_np(a, b):
    return np.mean((a - b) ** 2)

# Baseline in physical label space
Y_mean_baseline = Y_train.mean(axis=0, keepdims=True)

Y_train_baseline = np.repeat(Y_mean_baseline, len(Y_train), axis=0)
Y_val_baseline = np.repeat(Y_mean_baseline, len(Y_val), axis=0)
Y_test_baseline = np.repeat(Y_mean_baseline, len(Y_test), axis=0)

baseline_train_mse = mse_np(Y_train_baseline, Y_train)
baseline_val_mse = mse_np(Y_val_baseline, Y_val)
baseline_test_mse = mse_np(Y_test_baseline, Y_test)

print("Mean-channel baseline MSE in physical channel space:")
print("Train:", baseline_train_mse)
print("Val:  ", baseline_val_mse)
print("Test: ", baseline_test_mse)

# %% Cell 67
def predict_physical_y(model, X_s_np):
    """
    Predict physical, unstandardized channel labels.
    """
    model.eval()

    X_t = torch.tensor(X_s_np, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_s = model(X_t).cpu().numpy()

    pred_phys = invert_standardizer(pred_s, Y_mean, Y_std)

    return pred_phys


Y_train_pred = predict_physical_y(model, X_train_s)
Y_val_pred = predict_physical_y(model, X_val_s)
Y_test_pred = predict_physical_y(model, X_test_s)

channelnet_train_mse = mse_np(Y_train_pred, Y_train)
channelnet_val_mse = mse_np(Y_val_pred, Y_val)
channelnet_test_mse = mse_np(Y_test_pred, Y_test)

print("ChannelNet MSE in physical channel space:")
print("Train:", channelnet_train_mse)
print("Val:  ", channelnet_val_mse)
print("Test: ", channelnet_test_mse)

# %% Cell 68
print("Comparison: physical channel-label MSE")
print("-" * 50)
print(f"{'Split':<12s} {'Mean baseline':>18s} {'ChannelNet':>18s}")
print("-" * 50)
print(f"{'Train':<12s} {baseline_train_mse:18.6e} {channelnet_train_mse:18.6e}")
print(f"{'Validation':<12s} {baseline_val_mse:18.6e} {channelnet_val_mse:18.6e}")
print(f"{'Test':<12s} {baseline_test_mse:18.6e} {channelnet_test_mse:18.6e}")

# %% Cell 69
plt.figure(figsize=(5, 5))

plt.scatter(Y_test.flatten(), Y_test_pred.flatten(), alpha=0.6)

min_val = min(Y_test.min(), Y_test_pred.min())
max_val = max(Y_test.max(), Y_test_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("True channel parameter")
plt.ylabel("Predicted channel parameter")
plt.title("ChannelNet: predicted vs true local-channel parameters")
plt.grid(True)
plt.show()

# %% Cell 70
idx = 0

y_true = Y_test[idx]
y_pred = Y_test_pred[idx]

A1_true, b1_true, A2_true, b2_true = unflatten_two_local_channels(y_true)
A1_pred, b1_pred, A2_pred, b2_pred = unflatten_two_local_channels(y_pred)

print("Test example:", idx)

print("\nA1 true:")
print(A1_true)

print("\nA1 pred:")
print(A1_pred)

print("\nb1 true:")
print(b1_true)

print("\nb1 pred:")
print(b1_pred)

print("\nA2 true:")
print(A2_true)

print("\nA2 pred:")
print(A2_pred)

print("\nb2 true:")
print(b2_true)

print("\nb2 pred:")
print(b2_pred)

print("\nExample MSE:", np.mean((y_true - y_pred) ** 2))

# %% Cell 71
def physicality_violation_np(y_phys):
    """
    Return average and maximum Bloch-ball violation for predicted channels.
    """
    violations = []

    for yi in y_phys:
        A1, b1, A2, b2 = unflatten_two_local_channels(yi)

        for A, b in [(A1, b1), (A2, b2)]:
            for r in bloch_test_np:
                rout = A @ r + b
                violation = max(0.0, np.linalg.norm(rout) - 1.0)
                violations.append(violation)

    violations = np.array(violations)

    return {
        "mean_violation": violations.mean(),
        "max_violation": violations.max(),
        "fraction_violating": np.mean(violations > 1e-6),
    }


phys_true = physicality_violation_np(Y_test)
phys_pred = physicality_violation_np(Y_test_pred)

print("True channel physicality check:", phys_true)
print("Predicted channel physicality check:", phys_pred)

# %% Cell 72
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "X_mean": X_mean,
        "X_std": X_std,
        "Y_mean": Y_mean,
        "Y_std": Y_std,
        "history": history,
        "shots_dataset": shots_dataset,
    },
    "channelnet_debug_model.pt"
)

print("Saved trained model to channelnet_debug_model.pt")

# %% Cell 73
def direct_channel_from_tomography_vector(x_tomo):
    """
    Directly fit local affine Bloch channels from one finite-shot tomography vector.

    Input:
        x_tomo: shape (36,)

    Output:
        y_direct: shape (24,) = [A1, b1, A2, b2]
    """
    x_tomo = np.asarray(x_tomo, dtype=np.float64)

    channels = []

    offset = 0

    for target_qubit in [1, 2]:
        r_in_list = []
        r_out_list = []

        for state_label in tomography_input_labels:
            r_in = expected_bloch[state_label]

            # Feature order for this state: X, Y, Z
            r_out = x_tomo[offset:offset + 3]
            offset += 3

            r_in_list.append(r_in)
            r_out_list.append(r_out)

        A, b = fit_affine_bloch_map(r_in_list, r_out_list)
        channels.append((A, b))

    A1, b1 = channels[0]
    A2, b2 = channels[1]

    y_direct = flatten_two_local_channels(A1, b1, A2, b2)

    return y_direct.astype(np.float32)

# %% Cell 74
idx = 0

y_direct_example = direct_channel_from_tomography_vector(X_test[idx])
y_true_example = Y_test[idx]
y_pred_example = Y_test_pred[idx]

print("Shapes:")
print("y_true:", y_true_example.shape)
print("y_direct:", y_direct_example.shape)
print("y_ChannelNet:", y_pred_example.shape)

print("\nExample physical-space MSE:")
print("Direct noisy tomography:", np.mean((y_direct_example - y_true_example) ** 2))
print("ChannelNet:", np.mean((y_pred_example - y_true_example) ** 2))

# %% Cell 75
def direct_channels_for_dataset(X):
    """
    Apply direct noisy tomography reconstruction to every example.
    """
    Y_direct = []

    for i in range(len(X)):
        y_direct = direct_channel_from_tomography_vector(X[i])
        Y_direct.append(y_direct)

    return np.stack(Y_direct).astype(np.float32)


Y_train_direct = direct_channels_for_dataset(X_train)
Y_val_direct = direct_channels_for_dataset(X_val)
Y_test_direct = direct_channels_for_dataset(X_test)

print("Y_train_direct shape:", Y_train_direct.shape)
print("Y_val_direct shape:", Y_val_direct.shape)
print("Y_test_direct shape:", Y_test_direct.shape)

# %% Cell 76
direct_train_mse = mse_np(Y_train_direct, Y_train)
direct_val_mse = mse_np(Y_val_direct, Y_val)
direct_test_mse = mse_np(Y_test_direct, Y_test)

channelnet_train_mse = mse_np(Y_train_pred, Y_train)
channelnet_val_mse = mse_np(Y_val_pred, Y_val)
channelnet_test_mse = mse_np(Y_test_pred, Y_test)

baseline_train_mse = mse_np(Y_train_baseline, Y_train)
baseline_val_mse = mse_np(Y_val_baseline, Y_val)
baseline_test_mse = mse_np(Y_test_baseline, Y_test)

print("Physical-space channel-label MSE")
print("-" * 75)
print(f"{'Split':<12s} {'Mean baseline':>18s} {'Direct tomography':>20s} {'ChannelNet':>18s}")
print("-" * 75)
print(f"{'Train':<12s} {baseline_train_mse:18.6e} {direct_train_mse:20.6e} {channelnet_train_mse:18.6e}")
print(f"{'Validation':<12s} {baseline_val_mse:18.6e} {direct_val_mse:20.6e} {channelnet_val_mse:18.6e}")
print(f"{'Test':<12s} {baseline_test_mse:18.6e} {direct_test_mse:20.6e} {channelnet_test_mse:18.6e}")

# %% Cell 77
def per_example_l2_error(Y_est, Y_true):
    return np.linalg.norm(Y_est - Y_true, axis=1)

err_direct_test = per_example_l2_error(Y_test_direct, Y_test)
err_channelnet_test = per_example_l2_error(Y_test_pred, Y_test)
err_mean_test = per_example_l2_error(Y_test_baseline, Y_test)

print("Test per-example L2 error:")
print("Mean baseline:")
print("  mean:", err_mean_test.mean(), "median:", np.median(err_mean_test))

print("Direct tomography:")
print("  mean:", err_direct_test.mean(), "median:", np.median(err_direct_test))

print("ChannelNet:")
print("  mean:", err_channelnet_test.mean(), "median:", np.median(err_channelnet_test))

# %% Cell 78
methods = ["Mean\nbaseline", "Direct\nnoisy tomo", "ChannelNet"]
mean_errors = [
    err_mean_test.mean(),
    err_direct_test.mean(),
    err_channelnet_test.mean()
]

median_errors = [
    np.median(err_mean_test),
    np.median(err_direct_test),
    np.median(err_channelnet_test)
]

xpos = np.arange(len(methods))

plt.figure(figsize=(7, 4))
plt.bar(xpos - 0.18, mean_errors, width=0.36, label="Mean L2 error")
plt.bar(xpos + 0.18, median_errors, width=0.36, label="Median L2 error")

plt.xticks(xpos, methods)
plt.ylabel("Channel-label L2 error")
plt.title("Effective channel learning: baseline comparison")
plt.legend()
plt.grid(axis="y")
plt.show()

# %% Cell 79
plt.figure(figsize=(5, 5))

plt.scatter(Y_test.flatten(), Y_test_direct.flatten(), alpha=0.6)

min_val = min(Y_test.min(), Y_test_direct.min())
max_val = max(Y_test.max(), Y_test_direct.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("True channel parameter")
plt.ylabel("Direct tomography estimate")
plt.title("Direct noisy tomography: estimated vs true")
plt.grid(True)
plt.show()

# %% Cell 80
plt.figure(figsize=(5, 5))

plt.scatter(Y_test.flatten(), Y_test_pred.flatten(), alpha=0.6)

min_val = min(Y_test.min(), Y_test_pred.min())
max_val = max(Y_test.max(), Y_test_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("True channel parameter")
plt.ylabel("ChannelNet prediction")
plt.title("ChannelNet: predicted vs true")
plt.grid(True)
plt.show()

# %% Cell 81
idx = int(np.argmin(err_channelnet_test - err_direct_test))

print("Example where ChannelNet improves most relative to direct tomography")
print("Index:", idx)
print("Direct error:", err_direct_test[idx])
print("ChannelNet error:", err_channelnet_test[idx])

A1_true, b1_true, A2_true, b2_true = unflatten_two_local_channels(Y_test[idx])
A1_dir, b1_dir, A2_dir, b2_dir = unflatten_two_local_channels(Y_test_direct[idx])
A1_pred, b1_pred, A2_pred, b2_pred = unflatten_two_local_channels(Y_test_pred[idx])

print("\nA1 true:")
print(A1_true)

print("\nA1 direct tomography:")
print(A1_dir)

print("\nA1 ChannelNet:")
print(A1_pred)

print("\nb1 true:")
print(b1_true)

print("\nb1 direct tomography:")
print(b1_dir)

print("\nb1 ChannelNet:")
print(b1_pred)

# %% Cell 82
idx = int(np.argmax(err_channelnet_test - err_direct_test))

print("Example where direct tomography beats ChannelNet most")
print("Index:", idx)
print("Direct error:", err_direct_test[idx])
print("ChannelNet error:", err_channelnet_test[idx])

A1_true, b1_true, A2_true, b2_true = unflatten_two_local_channels(Y_test[idx])
A1_dir, b1_dir, A2_dir, b2_dir = unflatten_two_local_channels(Y_test_direct[idx])
A1_pred, b1_pred, A2_pred, b2_pred = unflatten_two_local_channels(Y_test_pred[idx])

print("\nA1 true:")
print(A1_true)

print("\nA1 direct tomography:")
print(A1_dir)

print("\nA1 ChannelNet:")
print(A1_pred)

print("\nb1 true:")
print(b1_true)

print("\nb1 direct tomography:")
print(b1_dir)

print("\nb1 ChannelNet:")
print(b1_pred)

# %% Cell 83
phys_direct = physicality_violation_np(Y_test_direct)
phys_channelnet = physicality_violation_np(Y_test_pred)
phys_true = physicality_violation_np(Y_test)

print("Physicality violation comparison on test set")
print("-" * 60)
print("True reference channels:  ", phys_true)
print("Direct noisy tomography:  ", phys_direct)
print("ChannelNet predictions:  ", phys_channelnet)

# %% Cell 84
labels = ["True", "Direct\nnoisy tomo", "ChannelNet"]
mean_violations = [
    phys_true["mean_violation"],
    phys_direct["mean_violation"],
    phys_channelnet["mean_violation"]
]
max_violations = [
    phys_true["max_violation"],
    phys_direct["max_violation"],
    phys_channelnet["max_violation"]
]

xpos = np.arange(len(labels))

plt.figure(figsize=(7, 4))
plt.bar(xpos - 0.18, mean_violations, width=0.36, label="Mean violation")
plt.bar(xpos + 0.18, max_violations, width=0.36, label="Max violation")

plt.xticks(xpos, labels)
plt.ylabel("Bloch-ball violation")
plt.title("Physicality of learned effective channels")
plt.legend()
plt.grid(axis="y")
plt.show()

# %% Cell 85
shot_choices = np.array([128, 512, 1024, 4096], dtype=int)

def append_shot_feature(x_tomo, shots):
    """
    Append log10(shots) as one extra feature.
    Input:
        x_tomo: shape (36,)
    Output:
        x_aug: shape (37,)
    """
    shot_feature = np.array([np.log10(shots)], dtype=np.float32)
    return np.concatenate([x_tomo.astype(np.float32), shot_feature])


def generate_one_mixedshot_example(
    rng,
    shot_choices=shot_choices,
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    include_decoherence=True,
    include_readout=True
):
    """
    Generate one supervised example with randomly selected shot number.

    Hidden device:
        randomized transmon parameters.

    Input x:
        finite-shot local tomography vector + log10(shots), shape (37,)

    Label y:
        exact effective local affine channels, shape (24,)

    Also returns:
        params, shots
    """
    shots = int(rng.choice(shot_choices))
    params = sample_hidden_device(rng)

    x_tomo = generate_local_tomography_vector(
        params=params,
        shots=shots,
        pulse_theta=pulse_theta,
        pulse_phase=pulse_phase,
        pulse_T=pulse_T,
        rng=rng,
        include_decoherence=include_decoherence
    )

    x = append_shot_feature(x_tomo, shots)

    y = reference_two_local_channels_label(
        params=params,
        spectator_label="0",
        pulse_theta=pulse_theta,
        pulse_phase=pulse_phase,
        pulse_T=pulse_T,
        include_decoherence=include_decoherence,
        include_readout=include_readout
    )

    return x.astype(np.float32), y.astype(np.float32), params, shots

# %% Cell 86
def build_mixedshot_dataset(
    n_examples,
    seed=0,
    shot_choices=shot_choices,
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    include_decoherence=True,
    include_readout=True,
    print_every=25
):
    """
    Build mixed-shot synthetic dataset.

    X shape: (n_examples, 37)
    Y shape: (n_examples, 24)
    shots_array shape: (n_examples,)
    """
    rng = np.random.default_rng(seed)

    X = []
    Y = []
    hidden_params_list = []
    shots_list = []

    t0 = time.time()

    for i in range(n_examples):
        x, y, params, shots = generate_one_mixedshot_example(
            rng=rng,
            shot_choices=shot_choices,
            pulse_theta=pulse_theta,
            pulse_phase=pulse_phase,
            pulse_T=pulse_T,
            include_decoherence=include_decoherence,
            include_readout=include_readout
        )

        X.append(x)
        Y.append(y)
        hidden_params_list.append(params)
        shots_list.append(shots)

        if (i + 1) % print_every == 0 or (i + 1) == n_examples:
            elapsed = time.time() - t0
            print(f"Generated {i+1:4d}/{n_examples} examples | elapsed {elapsed:.1f} s")

    X = np.stack(X).astype(np.float32)
    Y = np.stack(Y).astype(np.float32)
    shots_array = np.array(shots_list, dtype=int)

    return X, Y, hidden_params_list, shots_array

# %% Cell 87
# Recommended final-medium size
N_train_final = 300
N_val_final = 80
N_test_final = 80

# If Colab is too slow, use these instead:
# N_train_final = 150
# N_val_final = 40
# N_test_final = 40

print("Final dataset sizes:")
print("Train:", N_train_final)
print("Val:  ", N_val_final)
print("Test: ", N_test_final)

# %% Cell 88
X_train_final, Y_train_final, params_train_final, shots_train_final = build_mixedshot_dataset(
    n_examples=N_train_final,
    seed=1000,
    print_every=25
)

X_val_final, Y_val_final, params_val_final, shots_val_final = build_mixedshot_dataset(
    n_examples=N_val_final,
    seed=2000,
    print_every=20
)

X_test_final, Y_test_final, params_test_final, shots_test_final = build_mixedshot_dataset(
    n_examples=N_test_final,
    seed=3000,
    print_every=20
)

# %% Cell 89
print("X_train_final shape:", X_train_final.shape)
print("Y_train_final shape:", Y_train_final.shape)

print("X_val_final shape:", X_val_final.shape)
print("Y_val_final shape:", Y_val_final.shape)

print("X_test_final shape:", X_test_final.shape)
print("Y_test_final shape:", Y_test_final.shape)

print("\nTomography feature range, excluding shot feature:")
print("min:", X_train_final[:, :36].min())
print("max:", X_train_final[:, :36].max())

print("\nShot feature range:")
print("min:", X_train_final[:, 36].min())
print("max:", X_train_final[:, 36].max())

print("\nShot counts in training set:")
unique, counts = np.unique(shots_train_final, return_counts=True)
for s, c in zip(unique, counts):
    print(f"{s:4d} shots: {c} examples")

# %% Cell 90
np.savez(
    "channelnet_mixedshot_raw_dataset.npz",
    X_train_final=X_train_final,
    Y_train_final=Y_train_final,
    X_val_final=X_val_final,
    Y_val_final=Y_val_final,
    X_test_final=X_test_final,
    Y_test_final=Y_test_final,
    shots_train_final=shots_train_final,
    shots_val_final=shots_val_final,
    shots_test_final=shots_test_final
)

print("Saved raw mixed-shot dataset.")

# %% Cell 91
X_mean_final, X_std_final = compute_standardizer(X_train_final)
Y_mean_final, Y_std_final = compute_standardizer(Y_train_final)

X_train_final_s = apply_standardizer(X_train_final, X_mean_final, X_std_final)
X_val_final_s = apply_standardizer(X_val_final, X_mean_final, X_std_final)
X_test_final_s = apply_standardizer(X_test_final, X_mean_final, X_std_final)

Y_train_final_s = apply_standardizer(Y_train_final, Y_mean_final, Y_std_final)
Y_val_final_s = apply_standardizer(Y_val_final, Y_mean_final, Y_std_final)
Y_test_final_s = apply_standardizer(Y_test_final, Y_mean_final, Y_std_final)

print("Final standardized X mean:", X_train_final_s.mean())
print("Final standardized X std: ", X_train_final_s.std())

print("Final standardized Y mean:", Y_train_final_s.mean())
print("Final standardized Y std: ", Y_train_final_s.std())

# %% Cell 92
np.savez(
    "channelnet_mixedshot_standardized_dataset.npz",
    X_train=X_train_final,
    Y_train=Y_train_final,
    X_val=X_val_final,
    Y_val=Y_val_final,
    X_test=X_test_final,
    Y_test=Y_test_final,
    X_train_s=X_train_final_s,
    Y_train_s=Y_train_final_s,
    X_val_s=X_val_final_s,
    Y_val_s=Y_val_final_s,
    X_test_s=X_test_final_s,
    Y_test_s=Y_test_final_s,
    X_mean=X_mean_final,
    X_std=X_std_final,
    Y_mean=Y_mean_final,
    Y_std=Y_std_final,
    shots_train=shots_train_final,
    shots_val=shots_val_final,
    shots_test=shots_test_final
)

print("Saved standardized mixed-shot dataset.")

# %% Cell 93
X_train_final_t = torch.tensor(X_train_final_s, dtype=torch.float32)
Y_train_final_t = torch.tensor(Y_train_final_s, dtype=torch.float32)

X_val_final_t = torch.tensor(X_val_final_s, dtype=torch.float32)
Y_val_final_t = torch.tensor(Y_val_final_s, dtype=torch.float32)

X_test_final_t = torch.tensor(X_test_final_s, dtype=torch.float32)
Y_test_final_t = torch.tensor(Y_test_final_s, dtype=torch.float32)

train_final_dataset = TensorDataset(X_train_final_t, Y_train_final_t)
val_final_dataset = TensorDataset(X_val_final_t, Y_val_final_t)
test_final_dataset = TensorDataset(X_test_final_t, Y_test_final_t)

train_final_loader = DataLoader(train_final_dataset, batch_size=32, shuffle=True)
val_final_loader = DataLoader(val_final_dataset, batch_size=128, shuffle=False)
test_final_loader = DataLoader(test_final_dataset, batch_size=128, shuffle=False)

print("Final train tensors:", X_train_final_t.shape, Y_train_final_t.shape)
print("Final val tensors:  ", X_val_final_t.shape, Y_val_final_t.shape)
print("Final test tensors: ", X_test_final_t.shape, Y_test_final_t.shape)

# %% Cell 94
model_final = ChannelNet(input_dim=37, output_dim=24).to(device)

n_params_final = sum(p.numel() for p in model_final.parameters() if p.requires_grad)
print(model_final)
print("Number of trainable parameters:", n_params_final)

# %% Cell 95
Y_mean_t = torch.tensor(Y_mean_final, dtype=torch.float32).to(device)
Y_std_t = torch.tensor(Y_std_final, dtype=torch.float32).to(device)

print("Updated Y standardizer tensors for final model.")

# %% Cell 96
unstandardize_y_torch
physicality_loss_from_standardized_output

# %% Cell 97
history_final = train_channelnet(
    model=model_final,
    train_loader=train_final_loader,
    val_loader=val_final_loader,
    n_epochs=400,
    lr=8e-4,
    weight_decay=1e-5,
    lambda_phys=0.02
)

# %% Cell 98
plt.figure(figsize=(8, 5))

plt.plot(history_final["train_mse"], label="train MSE")
plt.plot(history_final["val_mse"], label="validation MSE")

plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE on standardized channel labels")
plt.title("Shot-aware ChannelNet training curve")
plt.legend()
plt.grid(True)
plt.show()

# %% Cell 99
train_final_metrics = evaluate_model(model_final, train_final_loader, lambda_phys=0.02)
val_final_metrics = evaluate_model(model_final, val_final_loader, lambda_phys=0.02)
test_final_metrics = evaluate_model(model_final, test_final_loader, lambda_phys=0.02)

print("Final train metrics:", train_final_metrics)
print("Final val metrics:  ", val_final_metrics)
print("Final test metrics: ", test_final_metrics)

# %% Cell 100
def predict_physical_y_final(model, X_s_np):
    """
    Predict physical, unstandardized channel labels using final standardizer.
    """
    model.eval()

    X_t = torch.tensor(X_s_np, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_s = model(X_t).cpu().numpy()

    pred_phys = invert_standardizer(pred_s, Y_mean_final, Y_std_final)

    return pred_phys.astype(np.float32)

# %% Cell 101
Y_train_final_pred = predict_physical_y_final(model_final, X_train_final_s)
Y_val_final_pred = predict_physical_y_final(model_final, X_val_final_s)
Y_test_final_pred = predict_physical_y_final(model_final, X_test_final_s)

print("Y_test_final_pred shape:", Y_test_final_pred.shape)

# %% Cell 102
Y_train_final_direct = direct_channels_for_dataset(X_train_final[:, :36])
Y_val_final_direct = direct_channels_for_dataset(X_val_final[:, :36])
Y_test_final_direct = direct_channels_for_dataset(X_test_final[:, :36])

print("Y_test_final_direct shape:", Y_test_final_direct.shape)

# %% Cell 103
Y_mean_baseline_final = Y_train_final.mean(axis=0, keepdims=True)

Y_train_final_baseline = np.repeat(Y_mean_baseline_final, len(Y_train_final), axis=0)
Y_val_final_baseline = np.repeat(Y_mean_baseline_final, len(Y_val_final), axis=0)
Y_test_final_baseline = np.repeat(Y_mean_baseline_final, len(Y_test_final), axis=0)

# %% Cell 104
final_baseline_train_mse = mse_np(Y_train_final_baseline, Y_train_final)
final_baseline_val_mse = mse_np(Y_val_final_baseline, Y_val_final)
final_baseline_test_mse = mse_np(Y_test_final_baseline, Y_test_final)

final_direct_train_mse = mse_np(Y_train_final_direct, Y_train_final)
final_direct_val_mse = mse_np(Y_val_final_direct, Y_val_final)
final_direct_test_mse = mse_np(Y_test_final_direct, Y_test_final)

final_channelnet_train_mse = mse_np(Y_train_final_pred, Y_train_final)
final_channelnet_val_mse = mse_np(Y_val_final_pred, Y_val_final)
final_channelnet_test_mse = mse_np(Y_test_final_pred, Y_test_final)

print("Final physical-space channel-label MSE")
print("-" * 75)
print(f"{'Split':<12s} {'Mean baseline':>18s} {'Direct tomography':>20s} {'ChannelNet':>18s}")
print("-" * 75)
print(f"{'Train':<12s} {final_baseline_train_mse:18.6e} {final_direct_train_mse:20.6e} {final_channelnet_train_mse:18.6e}")
print(f"{'Validation':<12s} {final_baseline_val_mse:18.6e} {final_direct_val_mse:20.6e} {final_channelnet_val_mse:18.6e}")
print(f"{'Test':<12s} {final_baseline_test_mse:18.6e} {final_direct_test_mse:20.6e} {final_channelnet_test_mse:18.6e}")

# %% Cell 105
err_final_mean_test = per_example_l2_error(Y_test_final_baseline, Y_test_final)
err_final_direct_test = per_example_l2_error(Y_test_final_direct, Y_test_final)
err_final_channelnet_test = per_example_l2_error(Y_test_final_pred, Y_test_final)

print("Final test per-example L2 error:")
print("Mean baseline:")
print("  mean:", err_final_mean_test.mean(), "median:", np.median(err_final_mean_test))

print("Direct tomography:")
print("  mean:", err_final_direct_test.mean(), "median:", np.median(err_final_direct_test))

print("ChannelNet:")
print("  mean:", err_final_channelnet_test.mean(), "median:", np.median(err_final_channelnet_test))

# %% Cell 106
methods = ["Mean\nbaseline", "Direct\nnoisy tomo", "Shot-aware\nChannelNet"]

mean_errors = [
    err_final_mean_test.mean(),
    err_final_direct_test.mean(),
    err_final_channelnet_test.mean()
]

median_errors = [
    np.median(err_final_mean_test),
    np.median(err_final_direct_test),
    np.median(err_final_channelnet_test)
]

xpos = np.arange(len(methods))

plt.figure(figsize=(7, 4))
plt.bar(xpos - 0.18, mean_errors, width=0.36, label="Mean L2 error")
plt.bar(xpos + 0.18, median_errors, width=0.36, label="Median L2 error")

plt.xticks(xpos, methods)
plt.ylabel("Channel-label L2 error")
plt.title("Final effective-channel learning performance")
plt.legend()
plt.grid(axis="y")
plt.show()

# %% Cell 107
def grouped_errors_by_shots(Y_est, Y_true, shots_array):
    """
    Return mean/median L2 error grouped by shot count.
    """
    out = {}

    for s in sorted(np.unique(shots_array)):
        mask = shots_array == s

        errs = per_example_l2_error(Y_est[mask], Y_true[mask])

        out[int(s)] = {
            "n": int(mask.sum()),
            "mean": float(errs.mean()),
            "median": float(np.median(errs)),
            "std": float(errs.std())
        }

    return out


direct_by_shots = grouped_errors_by_shots(
    Y_test_final_direct,
    Y_test_final,
    shots_test_final
)

channelnet_by_shots = grouped_errors_by_shots(
    Y_test_final_pred,
    Y_test_final,
    shots_test_final
)

print("Direct tomography errors by shot count:")
for s, vals in direct_by_shots.items():
    print(s, vals)

print("\nChannelNet errors by shot count:")
for s, vals in channelnet_by_shots.items():
    print(s, vals)

# %% Cell 108
shots_sorted = sorted(np.unique(shots_test_final))

direct_means = [direct_by_shots[int(s)]["mean"] for s in shots_sorted]
channelnet_means = [channelnet_by_shots[int(s)]["mean"] for s in shots_sorted]

direct_medians = [direct_by_shots[int(s)]["median"] for s in shots_sorted]
channelnet_medians = [channelnet_by_shots[int(s)]["median"] for s in shots_sorted]

plt.figure(figsize=(7, 4))

plt.plot(shots_sorted, direct_means, marker="o", label="Direct tomography mean")
plt.plot(shots_sorted, channelnet_means, marker="s", label="ChannelNet mean")

plt.plot(shots_sorted, direct_medians, marker="o", linestyle="--", label="Direct tomography median")
plt.plot(shots_sorted, channelnet_medians, marker="s", linestyle="--", label="ChannelNet median")

plt.xscale("log")
plt.xlabel("Number of tomography shots")
plt.ylabel("Channel-label L2 error")
plt.title("Channel learning accuracy versus finite-shot data")
plt.legend()
plt.grid(True)
plt.show()

# %% Cell 109
plt.figure(figsize=(5, 5))

plt.scatter(Y_test_final.flatten(), Y_test_final_direct.flatten(), alpha=0.35, label="Direct tomography")
plt.scatter(Y_test_final.flatten(), Y_test_final_pred.flatten(), alpha=0.35, label="ChannelNet")

min_val = min(Y_test_final.min(), Y_test_final_direct.min(), Y_test_final_pred.min())
max_val = max(Y_test_final.max(), Y_test_final_direct.max(), Y_test_final_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("True channel parameter")
plt.ylabel("Estimated channel parameter")
plt.title("Final channel estimates: direct tomography vs ChannelNet")
plt.legend()
plt.grid(True)
plt.show()

# %% Cell 110
phys_true_final = physicality_violation_np(Y_test_final)
phys_direct_final = physicality_violation_np(Y_test_final_direct)
phys_channelnet_final = physicality_violation_np(Y_test_final_pred)

print("Final physicality violation comparison on test set")
print("-" * 60)
print("True reference channels:  ", phys_true_final)
print("Direct noisy tomography:  ", phys_direct_final)
print("ChannelNet predictions:  ", phys_channelnet_final)

# %% Cell 111
torch.save(
    {
        "model_state_dict": model_final.state_dict(),
        "X_mean": X_mean_final,
        "X_std": X_std_final,
        "Y_mean": Y_mean_final,
        "Y_std": Y_std_final,
        "history": history_final,
        "shot_choices": shot_choices,
        "N_train": N_train_final,
        "N_val": N_val_final,
        "N_test": N_test_final,
    },
    "channelnet_mixedshot_final_model.pt"
)

print("Saved final shot-aware ChannelNet model.")

# %% Cell 112
# Ideal 2-qubit Hilbert space, separate from 9D transmon space

I2 = np.eye(2, dtype=complex)

X = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

Y = np.array([
    [0, -1j],
    [1j, 0]
], dtype=complex)

Z = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)

I4 = np.eye(4, dtype=complex)

X1_4 = kron(X, I2)
Y1_4 = kron(Y, I2)
Z1_4 = kron(Z, I2)

X2_4 = kron(I2, X)
Y2_4 = kron(I2, Y)
Z2_4 = kron(I2, Z)

ZZ_4 = Z1_4 @ Z2_4

# MaxCut cost Hamiltonian for one edge
H_cost = 0.5 * (I4 - ZZ_4)

# Mixer Hamiltonian
H_mixer = X1_4 + X2_4

print("H_cost:")
print(np.real_if_close(H_cost))

print("\nH_mixer:")
print(np.real_if_close(H_mixer))

print("\nH_cost eigenvalues:", np.linalg.eigvalsh(H_cost))

# %% Cell 113
ket0_2 = basis(2, 0)
ket1_2 = basis(2, 1)

ket00_4 = kron(ket0_2, ket0_2)
ket01_4 = kron(ket0_2, ket1_2)
ket10_4 = kron(ket1_2, ket0_2)
ket11_4 = kron(ket1_2, ket1_2)

ket_plus_2 = normalize_ket((ket0_2 + ket1_2) / np.sqrt(2))
ketplusplus_4 = kron(ket_plus_2, ket_plus_2)

basis4 = {
    "00": ket00_4,
    "01": ket01_4,
    "10": ket10_4,
    "11": ket11_4,
}

print("Norm of |++>:", np.linalg.norm(ketplusplus_4))
print("Initial MaxCut cost <++|H_C|++>:", float(np.real(dagger(ketplusplus_4) @ H_cost @ ketplusplus_4)))

# %% Cell 114
def ideal_qaoa_state(gamma, beta):
    """
    Return ideal p=1 QAOA state for 2-qubit MaxCut.
    """
    U_cost = expm(-1j * gamma * H_cost)
    U_mixer = expm(-1j * beta * H_mixer)

    psi = U_mixer @ U_cost @ ketplusplus_4
    psi = normalize_ket(psi)

    return psi


def ideal_qaoa_cost(gamma, beta):
    """
    Ideal QAOA cost expectation <H_C>.
    """
    psi = ideal_qaoa_state(gamma, beta)
    cost = dagger(psi) @ H_cost @ psi
    return float(np.real_if_close(cost[0, 0]))


def bitstring_probabilities_4(psi):
    """
    Return computational-basis probabilities.
    """
    probs = {}
    for label, ket in basis4.items():
        probs[label] = float(np.abs((dagger(ket) @ psi)[0, 0]) ** 2)
    return probs

# %% Cell 115
test_angles = [
    (0.0, 0.0),
    (np.pi / 4, np.pi / 8),
    (np.pi / 2, np.pi / 4),
    (np.pi, np.pi / 4),
]

for gamma, beta in test_angles:
    psi = ideal_qaoa_state(gamma, beta)
    cost = ideal_qaoa_cost(gamma, beta)
    probs = bitstring_probabilities_4(psi)

    print(f"\ngamma={gamma:.3f}, beta={beta:.3f}")
    print(f"cost = {cost:.6f}")
    print("probabilities:", {k: round(v, 4) for k, v in probs.items()})
    print("prob sum:", sum(probs.values()))

# %% Cell 116
n_gamma = 81
n_beta = 81

gamma_grid = np.linspace(0, np.pi, n_gamma)
beta_grid = np.linspace(0, np.pi / 2, n_beta)

C_ideal_grid = np.zeros((n_gamma, n_beta), dtype=np.float64)

for i, gamma in enumerate(gamma_grid):
    for j, beta in enumerate(beta_grid):
        C_ideal_grid[i, j] = ideal_qaoa_cost(gamma, beta)

print("Ideal cost grid shape:", C_ideal_grid.shape)
print("Minimum ideal cost:", C_ideal_grid.min())
print("Maximum ideal cost:", C_ideal_grid.max())

max_idx = np.unravel_index(np.argmax(C_ideal_grid), C_ideal_grid.shape)
gamma_star = gamma_grid[max_idx[0]]
beta_star = beta_grid[max_idx[1]]
C_star = C_ideal_grid[max_idx]

print("\nBest grid point:")
print("gamma* =", gamma_star)
print("beta*  =", beta_star)
print("C*     =", C_star)

# %% Cell 117
plt.figure(figsize=(7, 5))

plt.imshow(
    C_ideal_grid,
    origin="lower",
    aspect="auto",
    extent=[
        beta_grid[0],
        beta_grid[-1],
        gamma_grid[0],
        gamma_grid[-1]
    ]
)

plt.colorbar(label="Ideal QAOA cost")
plt.scatter([beta_star], [gamma_star], marker="x", s=100, label="best grid point")

plt.xlabel(r"$\beta$")
plt.ylabel(r"$\gamma$")
plt.title("Ideal 2-qubit QAOA/MaxCut cost landscape")
plt.legend()
plt.show()

# %% Cell 118
def maxcut_cost_from_probs(probs):
    """
    MaxCut cost for the single-edge two-qubit graph.
    """
    return probs["01"] + probs["10"]


for gamma, beta in test_angles:
    psi = ideal_qaoa_state(gamma, beta)
    probs = bitstring_probabilities_4(psi)

    cost_from_probs = maxcut_cost_from_probs(probs)
    cost_from_H = ideal_qaoa_cost(gamma, beta)

    print(f"\ngamma={gamma:.3f}, beta={beta:.3f}")
    print("cost from probabilities:", cost_from_probs)
    print("cost from Hamiltonian:  ", cost_from_H)
    print("difference:", abs(cost_from_probs - cost_from_H))

# %% Cell 119
np.savez(
    "ideal_qaoa_maxcut_benchmark.npz",
    gamma_grid=gamma_grid,
    beta_grid=beta_grid,
    C_ideal_grid=C_ideal_grid,
    gamma_star=gamma_star,
    beta_star=beta_star,
    C_star=C_star
)

print("Saved ideal QAOA benchmark.")

# %% Cell 120
# Single-qubit Pauli basis in order: I, X, Y, Z
paulis_1q = [I2, X, Y, Z]
pauli_labels_1q = ["I", "X", "Y", "Z"]

# Two-qubit Pauli basis
paulis_2q = []
pauli_labels_2q = []

for la, Pa in zip(pauli_labels_1q, paulis_1q):
    for lb, Pb in zip(pauli_labels_1q, paulis_1q):
        paulis_2q.append(kron(Pa, Pb))
        pauli_labels_2q.append(la + lb)

paulis_2q = np.array(paulis_2q)

def rho_to_pauli_coeffs_2q(rho):
    """
    Return coefficients c_ab = Tr(rho sigma_a ⊗ sigma_b).
    Shape: (16,)
    """
    coeffs = []
    for P in paulis_2q:
        coeffs.append(float(np.real_if_close(np.trace(rho @ P))))
    return np.array(coeffs, dtype=np.float64)

def pauli_coeffs_to_rho_2q(coeffs):
    """
    Reconstruct rho = 1/4 sum c_ab sigma_a ⊗ sigma_b.
    """
    rho = np.zeros((4, 4), dtype=complex)
    for c, P in zip(coeffs, paulis_2q):
        rho += c * P
    rho = rho / 4.0
    return clean_density_matrix(rho)

# %% Cell 121
rho_pp = ket_to_rho(ketplusplus_4)

coeffs_pp = rho_to_pauli_coeffs_2q(rho_pp)
rho_pp_rec = pauli_coeffs_to_rho_2q(coeffs_pp)

print("Reconstruction error for |++><++|:", fro_norm(rho_pp - rho_pp_rec))

print("\nNonzero-ish Pauli coefficients for |++>:")
for label, c in zip(pauli_labels_2q, coeffs_pp):
    if abs(c) > 1e-8:
        print(f"{label}: {c:.3f}")

# %% Cell 122
def affine_channel_to_ptm(A, b):
    """
    Convert affine Bloch map r_out = A r_in + b
    into 4x4 Pauli transfer matrix acting on [I, X, Y, Z] coefficients.
    """
    R = np.zeros((4, 4), dtype=np.float64)

    R[0, 0] = 1.0
    R[1:4, 0] = b
    R[1:4, 1:4] = A

    return R

# %% Cell 123
def apply_product_local_channel_to_rho(rho, A1, b1, A2, b2):
    """
    Apply approximate product local channel E1 ⊗ E2 to a two-qubit density matrix.
    """
    R1 = affine_channel_to_ptm(A1, b1)
    R2 = affine_channel_to_ptm(A2, b2)

    R12 = np.kron(R1, R2)

    c_in = rho_to_pauli_coeffs_2q(rho)
    c_out = R12 @ c_in

    rho_out = pauli_coeffs_to_rho_2q(c_out)

    return rho_out

# %% Cell 124
def cost_from_rho_2q(rho):
    """
    MaxCut cost = Tr(rho H_cost).
    """
    val = np.trace(rho @ H_cost)
    return float(np.real_if_close(val))

def zz_from_rho_2q(rho):
    """
    Return <Z1 Z2>.
    """
    val = np.trace(rho @ ZZ_4)
    return float(np.real_if_close(val))

# %% Cell 125
A_id = np.eye(3)
b_zero = np.zeros(3)

gamma_test = np.pi / 4
beta_test = np.pi / 8

psi_test = ideal_qaoa_state(gamma_test, beta_test)
rho_test = ket_to_rho(psi_test)

rho_after_id = apply_product_local_channel_to_rho(
    rho_test,
    A_id, b_zero,
    A_id, b_zero
)

print("Identity-channel state error:", fro_norm(rho_test - rho_after_id))
print("Ideal cost:", cost_from_rho_2q(rho_test))
print("After identity channel cost:", cost_from_rho_2q(rho_after_id))

# %% Cell 126
idx_device = 0

A1_true, b1_true, A2_true, b2_true = unflatten_two_local_channels(Y_test_final[idx_device])

rho_noisy_eff = apply_product_local_channel_to_rho(
    rho_test,
    A1_true, b1_true,
    A2_true, b2_true
)

print("Ideal cost:", cost_from_rho_2q(rho_test))
print("Effective noisy cost:", cost_from_rho_2q(rho_noisy_eff))
print("Ideal <ZZ>:", zz_from_rho_2q(rho_test))
print("Noisy <ZZ>:", zz_from_rho_2q(rho_noisy_eff))
print("Trace noisy rho:", np.trace(rho_noisy_eff))
print("Hermiticity error:", fro_norm(rho_noisy_eff - dagger(rho_noisy_eff)))
print("Eigenvalues noisy rho:", np.linalg.eigvalsh(rho_noisy_eff))

# %% Cell 127
def bitstring_probs_from_rho_2q(rho):
    """
    Computational-basis probabilities from a two-qubit density matrix.
    Returns dict with keys 00,01,10,11.
    """
    probs = {}
    for label, ket in basis4.items():
        P = ket @ dagger(ket)
        p = float(np.real_if_close(np.trace(rho @ P)))
        probs[label] = p

    # Clip and renormalize for numerical stability
    keys = ["00", "01", "10", "11"]
    arr = np.array([probs[k] for k in keys], dtype=np.float64)
    arr = np.clip(arr, 0.0, 1.0)

    s = arr.sum()
    if s <= 1e-12:
        arr = np.ones(4) / 4
    else:
        arr = arr / s

    return {k: float(v) for k, v in zip(keys, arr)}

def sample_qaoa_cost_from_probs(probs, shots=2048, rng=None):
    """
    Sample bitstrings and estimate MaxCut cost.
    """
    if rng is None:
        rng = np.random.default_rng()

    keys = ["00", "01", "10", "11"]
    p = np.array([probs[k] for k in keys], dtype=np.float64)
    p = p / p.sum()

    counts = rng.multinomial(shots, p)
    count_dict = {k: int(c) for k, c in zip(keys, counts)}

    cost_hat = (count_dict["01"] + count_dict["10"]) / shots

    return cost_hat, count_dict

# %% Cell 128
rng = np.random.default_rng(777)

probs_ideal = bitstring_probs_from_rho_2q(rho_test)
probs_noisy_eff = bitstring_probs_from_rho_2q(rho_noisy_eff)

cost_ideal_exact = maxcut_cost_from_probs(probs_ideal)
cost_noisy_exact = maxcut_cost_from_probs(probs_noisy_eff)

cost_noisy_sampled, counts_noisy = sample_qaoa_cost_from_probs(
    probs_noisy_eff,
    shots=2048,
    rng=rng
)

print("Ideal probs:", probs_ideal)
print("Noisy effective probs:", probs_noisy_eff)

print("\nIdeal exact cost:", cost_ideal_exact)
print("Noisy exact cost:", cost_noisy_exact)
print("Noisy sampled cost:", cost_noisy_sampled)
print("Noisy sampled counts:", counts_noisy)

# %% Cell 129
def effective_noisy_qaoa_cost(
    gamma,
    beta,
    A1,
    b1,
    A2,
    b2,
    shots=None,
    rng=None
):
    """
    Compute noisy QAOA cost after applying product local effective channel.

    If shots is None:
        return exact probability-level cost.

    If shots is integer:
        return finite-shot sampled cost.
    """
    psi = ideal_qaoa_state(gamma, beta)
    rho = ket_to_rho(psi)

    rho_noisy = apply_product_local_channel_to_rho(
        rho,
        A1, b1,
        A2, b2
    )

    probs_noisy = bitstring_probs_from_rho_2q(rho_noisy)

    if shots is None:
        return maxcut_cost_from_probs(probs_noisy)
    else:
        cost_hat, _ = sample_qaoa_cost_from_probs(
            probs_noisy,
            shots=shots,
            rng=rng
        )
        return cost_hat

# %% Cell 130
n_gamma_small = 41
n_beta_small = 41

gamma_grid_small = np.linspace(0, np.pi, n_gamma_small)
beta_grid_small = np.linspace(0, np.pi / 2, n_beta_small)

# Choose one test device
idx_device = 0
A1_true, b1_true, A2_true, b2_true = unflatten_two_local_channels(Y_test_final[idx_device])

C_ideal_small = np.zeros((n_gamma_small, n_beta_small))
C_noisy_true_small = np.zeros((n_gamma_small, n_beta_small))

for i, gamma in enumerate(gamma_grid_small):
    for j, beta in enumerate(beta_grid_small):
        C_ideal_small[i, j] = ideal_qaoa_cost(gamma, beta)
        C_noisy_true_small[i, j] = effective_noisy_qaoa_cost(
            gamma,
            beta,
            A1_true,
            b1_true,
            A2_true,
            b2_true,
            shots=None
        )

print("Ideal small landscape max:", C_ideal_small.max())
print("Noisy true small landscape max:", C_noisy_true_small.max())

print("Mean absolute distortion:")
print(np.mean(np.abs(C_noisy_true_small - C_ideal_small)))

# %% Cell 131
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

im0 = axes[0].imshow(
    C_ideal_small,
    origin="lower",
    aspect="auto",
    extent=[
        beta_grid_small[0],
        beta_grid_small[-1],
        gamma_grid_small[0],
        gamma_grid_small[-1]
    ],
    vmin=0,
    vmax=1
)
axes[0].set_title("Ideal QAOA cost")
axes[0].set_xlabel(r"$\beta$")
axes[0].set_ylabel(r"$\gamma$")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(
    C_noisy_true_small,
    origin="lower",
    aspect="auto",
    extent=[
        beta_grid_small[0],
        beta_grid_small[-1],
        gamma_grid_small[0],
        gamma_grid_small[-1]
    ],
    vmin=0,
    vmax=1
)
axes[1].set_title("Noisy effective QAOA cost")
axes[1].set_xlabel(r"$\beta$")
axes[1].set_ylabel(r"$\gamma$")
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.show()

# %% Cell 132
idx_device = 0

# True reference channel
y_true_channel = Y_test_final[idx_device]

# Direct finite-shot tomography channel
y_direct_channel = Y_test_final_direct[idx_device]

# ChannelNet channel
y_pred_channel = Y_test_final_pred[idx_device]

A1_true, b1_true, A2_true, b2_true = unflatten_two_local_channels(y_true_channel)
A1_direct, b1_direct, A2_direct, b2_direct = unflatten_two_local_channels(y_direct_channel)
A1_pred, b1_pred, A2_pred, b2_pred = unflatten_two_local_channels(y_pred_channel)

print("Channel errors for selected device:")
print("Direct tomography L2 error:", np.linalg.norm(y_direct_channel - y_true_channel))
print("ChannelNet L2 error:", np.linalg.norm(y_pred_channel - y_true_channel))

# %% Cell 133
def qaoa_landscape_from_channel(
    gamma_grid,
    beta_grid,
    A1,
    b1,
    A2,
    b2,
    shots=None,
    seed=0
):
    """
    Compute QAOA cost landscape using a product local effective channel.
    """
    rng = np.random.default_rng(seed)

    C = np.zeros((len(gamma_grid), len(beta_grid)), dtype=np.float64)

    for i, gamma in enumerate(gamma_grid):
        for j, beta in enumerate(beta_grid):
            C[i, j] = effective_noisy_qaoa_cost(
                gamma,
                beta,
                A1,
                b1,
                A2,
                b2,
                shots=shots,
                rng=rng
            )

    return C

# %% Cell 134
C_noisy_true = qaoa_landscape_from_channel(
    gamma_grid_small,
    beta_grid_small,
    A1_true, b1_true,
    A2_true, b2_true,
    shots=None,
    seed=1
)

C_noisy_direct_model = qaoa_landscape_from_channel(
    gamma_grid_small,
    beta_grid_small,
    A1_direct, b1_direct,
    A2_direct, b2_direct,
    shots=None,
    seed=1
)

C_noisy_channelnet_model = qaoa_landscape_from_channel(
    gamma_grid_small,
    beta_grid_small,
    A1_pred, b1_pred,
    A2_pred, b2_pred,
    shots=None,
    seed=1
)

print("Model landscape errors relative to true noisy landscape:")
print("Direct tomography model:",
      np.mean(np.abs(C_noisy_direct_model - C_noisy_true)))
print("ChannelNet model:",
      np.mean(np.abs(C_noisy_channelnet_model - C_noisy_true)))

# %% Cell 135
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

landscapes = [
    (C_noisy_true, "True hidden noisy\nlandscape"),
    (C_noisy_direct_model, "Direct-tomography\nmodel"),
    (C_noisy_channelnet_model, "ChannelNet\nmodel"),
]

for ax, (C, title) in zip(axes, landscapes):
    im = ax.imshow(
        C,
        origin="lower",
        aspect="auto",
        extent=[
            beta_grid_small[0],
            beta_grid_small[-1],
            gamma_grid_small[0],
            gamma_grid_small[-1]
        ],
        vmin=0,
        vmax=1
    )
    ax.set_title(title)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\gamma$")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()

# %% Cell 136
qaoa_shots = 2048

C_noisy_true_sampled = qaoa_landscape_from_channel(
    gamma_grid_small,
    beta_grid_small,
    A1_true, b1_true,
    A2_true, b2_true,
    shots=qaoa_shots,
    seed=2026
)

print("Finite-shot noisy landscape mean absolute difference from exact noisy:")
print(np.mean(np.abs(C_noisy_true_sampled - C_noisy_true)))

# %% Cell 137
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

im0 = axes[0].imshow(
    C_noisy_true,
    origin="lower",
    aspect="auto",
    extent=[
        beta_grid_small[0],
        beta_grid_small[-1],
        gamma_grid_small[0],
        gamma_grid_small[-1]
    ],
    vmin=0,
    vmax=1
)
axes[0].set_title("Exact true noisy QAOA cost")
axes[0].set_xlabel(r"$\beta$")
axes[0].set_ylabel(r"$\gamma$")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(
    C_noisy_true_sampled,
    origin="lower",
    aspect="auto",
    extent=[
        beta_grid_small[0],
        beta_grid_small[-1],
        gamma_grid_small[0],
        gamma_grid_small[-1]
    ],
    vmin=0,
    vmax=1
)
axes[1].set_title(f"Finite-shot noisy QAOA cost\n{qaoa_shots} shots")
axes[1].set_xlabel(r"$\beta$")
axes[1].set_ylabel(r"$\gamma$")
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.show()

# %% Cell 138
def mitigate_qaoa_landscape_by_bias_correction(
    C_measured_noisy,
    C_ideal,
    C_model_noisy,
    clip=True
):
    """
    Bias-correction mitigation.

    Estimated bias:
        b_hat = C_model_noisy - C_ideal

    Mitigated:
        C_mitigated = C_measured_noisy - b_hat
                    = C_measured_noisy - (C_model_noisy - C_ideal)
    """
    bias_hat = C_model_noisy - C_ideal
    C_mitigated = C_measured_noisy - bias_hat

    if clip:
        C_mitigated = np.clip(C_mitigated, 0.0, 1.0)

    return C_mitigated, bias_hat

# %% Cell 139
C_mitigated_direct, bias_direct = mitigate_qaoa_landscape_by_bias_correction(
    C_measured_noisy=C_noisy_true_sampled,
    C_ideal=C_ideal_small,
    C_model_noisy=C_noisy_direct_model,
    clip=True
)

C_mitigated_channelnet, bias_channelnet = mitigate_qaoa_landscape_by_bias_correction(
    C_measured_noisy=C_noisy_true_sampled,
    C_ideal=C_ideal_small,
    C_model_noisy=C_noisy_channelnet_model,
    clip=True
)

print("Shapes:")
print("C_ideal_small:", C_ideal_small.shape)
print("C_noisy_true_sampled:", C_noisy_true_sampled.shape)
print("C_mitigated_direct:", C_mitigated_direct.shape)
print("C_mitigated_channelnet:", C_mitigated_channelnet.shape)

# %% Cell 140
def landscape_error_metrics(C_est, C_ideal):
    """
    Compute error metrics between estimated and ideal cost landscapes.
    """
    abs_err = np.abs(C_est - C_ideal)

    return {
        "MAE": float(abs_err.mean()),
        "MedianAE": float(np.median(abs_err)),
        "MaxAE": float(abs_err.max()),
        "RMSE": float(np.sqrt(np.mean(abs_err**2))),
    }


def improvement_ratio(noisy_error, mitigated_error, eps=1e-12):
    """
    Ratio > 1 means mitigation improves the result.
    """
    return noisy_error / (mitigated_error + eps)

# %% Cell 141
metrics_noisy = landscape_error_metrics(
    C_noisy_true_sampled,
    C_ideal_small
)

metrics_direct = landscape_error_metrics(
    C_mitigated_direct,
    C_ideal_small
)

metrics_channelnet = landscape_error_metrics(
    C_mitigated_channelnet,
    C_ideal_small
)

print("Landscape error metrics relative to ideal")
print("-" * 70)
print(f"{'Method':<25s} {'MAE':>12s} {'MedianAE':>12s} {'RMSE':>12s} {'MaxAE':>12s}")
print("-" * 70)

for name, m in [
    ("Noisy measured", metrics_noisy),
    ("Direct mitigated", metrics_direct),
    ("ChannelNet mitigated", metrics_channelnet),
]:
    print(
        f"{name:<25s} "
        f"{m['MAE']:12.6f} "
        f"{m['MedianAE']:12.6f} "
        f"{m['RMSE']:12.6f} "
        f"{m['MaxAE']:12.6f}"
    )

print("\nImprovement ratios based on MAE:")
print("Direct mitigation:",
      improvement_ratio(metrics_noisy["MAE"], metrics_direct["MAE"]))
print("ChannelNet mitigation:",
      improvement_ratio(metrics_noisy["MAE"], metrics_channelnet["MAE"]))

# %% Cell 142
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

landscapes = [
    (C_ideal_small, "Ideal"),
    (C_noisy_true_sampled, f"Noisy measured\n{qaoa_shots} shots"),
    (C_mitigated_direct, "Mitigated\nDirect tomography"),
    (C_mitigated_channelnet, "Mitigated\nChannelNet"),
]

for ax, (C, title) in zip(axes, landscapes):
    im = ax.imshow(
        C,
        origin="lower",
        aspect="auto",
        extent=[
            beta_grid_small[0],
            beta_grid_small[-1],
            gamma_grid_small[0],
            gamma_grid_small[-1]
        ],
        vmin=0,
        vmax=1
    )

    ax.set_title(title)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\gamma$")

    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()

# %% Cell 143
err_noisy_landscape = np.abs(C_noisy_true_sampled - C_ideal_small)
err_direct_landscape = np.abs(C_mitigated_direct - C_ideal_small)
err_channelnet_landscape = np.abs(C_mitigated_channelnet - C_ideal_small)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

error_landscapes = [
    (err_noisy_landscape, "Noisy measured\nabsolute error"),
    (err_direct_landscape, "Direct mitigation\nabsolute error"),
    (err_channelnet_landscape, "ChannelNet mitigation\nabsolute error"),
]

vmax_err = max(
    err_noisy_landscape.max(),
    err_direct_landscape.max(),
    err_channelnet_landscape.max()
)

for ax, (E, title) in zip(axes, error_landscapes):
    im = ax.imshow(
        E,
        origin="lower",
        aspect="auto",
        extent=[
            beta_grid_small[0],
            beta_grid_small[-1],
            gamma_grid_small[0],
            gamma_grid_small[-1]
        ],
        vmin=0,
        vmax=vmax_err
    )

    ax.set_title(title)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\gamma$")

    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()

# %% Cell 144
methods = ["Noisy\nmeasured", "Direct\nmitigated", "ChannelNet\nmitigated"]

mae_values = [
    metrics_noisy["MAE"],
    metrics_direct["MAE"],
    metrics_channelnet["MAE"]
]

rmse_values = [
    metrics_noisy["RMSE"],
    metrics_direct["RMSE"],
    metrics_channelnet["RMSE"]
]

xpos = np.arange(len(methods))

plt.figure(figsize=(7, 4))

plt.bar(xpos - 0.18, mae_values, width=0.36, label="MAE")
plt.bar(xpos + 0.18, rmse_values, width=0.36, label="RMSE")

plt.xticks(xpos, methods)
plt.ylabel("Error relative to ideal QAOA landscape")
plt.title("QAOA reliability improvement by error mitigation")
plt.legend()
plt.grid(axis="y")
plt.show()

# %% Cell 145
def best_grid_point(C_grid, gamma_grid, beta_grid):
    """
    Return best gamma, beta, and cost from a grid.
    """
    idx = np.unravel_index(np.argmax(C_grid), C_grid.shape)

    gamma_best = gamma_grid[idx[0]]
    beta_best = beta_grid[idx[1]]
    C_best = C_grid[idx]

    return {
        "idx": idx,
        "gamma": float(gamma_best),
        "beta": float(beta_best),
        "cost": float(C_best),
    }


def ideal_cost_at_grid_index(idx, C_ideal):
    return float(C_ideal[idx])

# %% Cell 146
best_ideal = best_grid_point(C_ideal_small, gamma_grid_small, beta_grid_small)
best_noisy = best_grid_point(C_noisy_true_sampled, gamma_grid_small, beta_grid_small)
best_direct = best_grid_point(C_mitigated_direct, gamma_grid_small, beta_grid_small)
best_channelnet = best_grid_point(C_mitigated_channelnet, gamma_grid_small, beta_grid_small)

best_methods = [
    ("Ideal landscape", best_ideal),
    ("Noisy measured landscape", best_noisy),
    ("Direct-mitigated landscape", best_direct),
    ("ChannelNet-mitigated landscape", best_channelnet),
]

print("Best grid points")
print("-" * 95)
print(
    f"{'Method':<30s} {'gamma':>10s} {'beta':>10s} "
    f"{'landscape cost':>18s} {'ideal cost at chosen point':>28s}"
)
print("-" * 95)

for name, best in best_methods:
    ideal_cost_chosen = ideal_cost_at_grid_index(best["idx"], C_ideal_small)

    print(
        f"{name:<30s} "
        f"{best['gamma']:10.4f} "
        f"{best['beta']:10.4f} "
        f"{best['cost']:18.6f} "
        f"{ideal_cost_chosen:28.6f}"
    )

# %% Cell 147
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

landscapes_and_best = [
    (C_ideal_small, "Ideal", best_ideal),
    (C_noisy_true_sampled, "Noisy measured", best_noisy),
    (C_mitigated_direct, "Direct mitigated", best_direct),
    (C_mitigated_channelnet, "ChannelNet mitigated", best_channelnet),
]

for ax, (C, title, best) in zip(axes, landscapes_and_best):
    im = ax.imshow(
        C,
        origin="lower",
        aspect="auto",
        extent=[
            beta_grid_small[0],
            beta_grid_small[-1],
            gamma_grid_small[0],
            gamma_grid_small[-1]
        ],
        vmin=0,
        vmax=1
    )

    ax.scatter(
        [best["beta"]],
        [best["gamma"]],
        marker="x",
        s=100
    )

    ax.set_title(title)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\gamma$")

    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()

# %% Cell 148
def evaluate_qaoa_mitigation_for_device(
    idx_device,
    gamma_grid,
    beta_grid,
    C_ideal_grid,
    Y_true_all,
    Y_direct_all,
    Y_channelnet_all,
    qaoa_shots=2048,
    seed=0
):
    """
    Evaluate noisy and mitigated QAOA landscapes for one test device.
    """
    y_true = Y_true_all[idx_device]
    y_direct = Y_direct_all[idx_device]
    y_channelnet = Y_channelnet_all[idx_device]

    A1_true, b1_true, A2_true, b2_true = unflatten_two_local_channels(y_true)
    A1_direct, b1_direct, A2_direct, b2_direct = unflatten_two_local_channels(y_direct)
    A1_pred, b1_pred, A2_pred, b2_pred = unflatten_two_local_channels(y_channelnet)

    # True noisy exact and finite-shot measured
    C_noisy_exact = qaoa_landscape_from_channel(
        gamma_grid,
        beta_grid,
        A1_true, b1_true,
        A2_true, b2_true,
        shots=None,
        seed=seed
    )

    C_noisy_sampled = qaoa_landscape_from_channel(
        gamma_grid,
        beta_grid,
        A1_true, b1_true,
        A2_true, b2_true,
        shots=qaoa_shots,
        seed=seed + 123
    )

    # Model-predicted noisy landscapes
    C_direct_model = qaoa_landscape_from_channel(
        gamma_grid,
        beta_grid,
        A1_direct, b1_direct,
        A2_direct, b2_direct,
        shots=None,
        seed=seed
    )

    C_channelnet_model = qaoa_landscape_from_channel(
        gamma_grid,
        beta_grid,
        A1_pred, b1_pred,
        A2_pred, b2_pred,
        shots=None,
        seed=seed
    )

    # Bias correction
    C_direct_mitigated, _ = mitigate_qaoa_landscape_by_bias_correction(
        C_measured_noisy=C_noisy_sampled,
        C_ideal=C_ideal_grid,
        C_model_noisy=C_direct_model,
        clip=True
    )

    C_channelnet_mitigated, _ = mitigate_qaoa_landscape_by_bias_correction(
        C_measured_noisy=C_noisy_sampled,
        C_ideal=C_ideal_grid,
        C_model_noisy=C_channelnet_model,
        clip=True
    )

    # Metrics
    metrics = {
        "noisy": landscape_error_metrics(C_noisy_sampled, C_ideal_grid),
        "direct": landscape_error_metrics(C_direct_mitigated, C_ideal_grid),
        "channelnet": landscape_error_metrics(C_channelnet_mitigated, C_ideal_grid),
    }

    # Best-point recovery
    bests = {
        "ideal": best_grid_point(C_ideal_grid, gamma_grid, beta_grid),
        "noisy": best_grid_point(C_noisy_sampled, gamma_grid, beta_grid),
        "direct": best_grid_point(C_direct_mitigated, gamma_grid, beta_grid),
        "channelnet": best_grid_point(C_channelnet_mitigated, gamma_grid, beta_grid),
    }

    ideal_costs_at_chosen = {
        key: ideal_cost_at_grid_index(best["idx"], C_ideal_grid)
        for key, best in bests.items()
    }

    return {
        "metrics": metrics,
        "bests": bests,
        "ideal_costs_at_chosen": ideal_costs_at_chosen,
        "C_noisy_sampled": C_noisy_sampled,
        "C_direct_mitigated": C_direct_mitigated,
        "C_channelnet_mitigated": C_channelnet_mitigated,
    }

# %% Cell 149
n_eval_devices = min(20, len(Y_test_final))

results_multi = []

for k in range(n_eval_devices):
    result = evaluate_qaoa_mitigation_for_device(
        idx_device=k,
        gamma_grid=gamma_grid_small,
        beta_grid=beta_grid_small,
        C_ideal_grid=C_ideal_small,
        Y_true_all=Y_test_final,
        Y_direct_all=Y_test_final_direct,
        Y_channelnet_all=Y_test_final_pred,
        qaoa_shots=qaoa_shots,
        seed=5000 + k
    )

    results_multi.append(result)

    print(
        f"Device {k:02d} | "
        f"MAE noisy {result['metrics']['noisy']['MAE']:.4f} | "
        f"direct {result['metrics']['direct']['MAE']:.4f} | "
        f"ChannelNet {result['metrics']['channelnet']['MAE']:.4f}"
    )

# %% Cell 150
def collect_metric(results, method, metric_name):
    return np.array([
        r["metrics"][method][metric_name]
        for r in results
    ], dtype=np.float64)

mae_noisy_multi = collect_metric(results_multi, "noisy", "MAE")
mae_direct_multi = collect_metric(results_multi, "direct", "MAE")
mae_channelnet_multi = collect_metric(results_multi, "channelnet", "MAE")

rmse_noisy_multi = collect_metric(results_multi, "noisy", "RMSE")
rmse_direct_multi = collect_metric(results_multi, "direct", "RMSE")
rmse_channelnet_multi = collect_metric(results_multi, "channelnet", "RMSE")

print("Multi-device MAE summary")
print("-" * 70)
print(f"{'Method':<20s} {'Mean':>12s} {'Median':>12s} {'Std':>12s}")
print("-" * 70)

for name, arr in [
    ("Noisy", mae_noisy_multi),
    ("Direct mitigated", mae_direct_multi),
    ("ChannelNet mitigated", mae_channelnet_multi),
]:
    print(
        f"{name:<20s} "
        f"{arr.mean():12.6f} "
        f"{np.median(arr):12.6f} "
        f"{arr.std():12.6f}"
    )

print("\nMean MAE improvement ratios:")
print("Direct:",
      improvement_ratio(mae_noisy_multi.mean(), mae_direct_multi.mean()))
print("ChannelNet:",
      improvement_ratio(mae_noisy_multi.mean(), mae_channelnet_multi.mean()))

# %% Cell 151
methods = ["Noisy", "Direct\nmitigated", "ChannelNet\nmitigated"]

means = [
    mae_noisy_multi.mean(),
    mae_direct_multi.mean(),
    mae_channelnet_multi.mean()
]

stds = [
    mae_noisy_multi.std(),
    mae_direct_multi.std(),
    mae_channelnet_multi.std()
]

xpos = np.arange(len(methods))

plt.figure(figsize=(7, 4))

plt.bar(xpos, means, yerr=stds, capsize=5)

plt.xticks(xpos, methods)
plt.ylabel("Mean absolute error vs ideal QAOA landscape")
plt.title(f"QAOA reliability over {n_eval_devices} hidden devices")
plt.grid(axis="y")
plt.show()

# %% Cell 152
plt.figure(figsize=(6, 5))

plt.scatter(mae_noisy_multi, mae_direct_multi, label="Direct mitigation", alpha=0.8)
plt.scatter(mae_noisy_multi, mae_channelnet_multi, label="ChannelNet mitigation", alpha=0.8)

max_axis = max(
    mae_noisy_multi.max(),
    mae_direct_multi.max(),
    mae_channelnet_multi.max()
)

plt.plot([0, max_axis], [0, max_axis], linestyle="--")

plt.xlabel("Noisy MAE")
plt.ylabel("Mitigated MAE")
plt.title("Paired QAOA reliability improvement")
plt.legend()
plt.grid(True)
plt.show()

# %% Cell 153
def collect_ideal_chosen_cost(results, method):
    return np.array([
        r["ideal_costs_at_chosen"][method]
        for r in results
    ], dtype=np.float64)

chosen_ideal_noisy = collect_ideal_chosen_cost(results_multi, "noisy")
chosen_ideal_direct = collect_ideal_chosen_cost(results_multi, "direct")
chosen_ideal_channelnet = collect_ideal_chosen_cost(results_multi, "channelnet")
chosen_ideal_opt = collect_ideal_chosen_cost(results_multi, "ideal")

print("Ideal cost at selected optimum")
print("-" * 70)
print(f"{'Selection method':<25s} {'Mean':>12s} {'Median':>12s} {'Std':>12s}")
print("-" * 70)

for name, arr in [
    ("Ideal optimum", chosen_ideal_opt),
    ("Noisy-selected", chosen_ideal_noisy),
    ("Direct-mitigated-selected", chosen_ideal_direct),
    ("ChannelNet-selected", chosen_ideal_channelnet),
]:
    print(
        f"{name:<25s} "
        f"{arr.mean():12.6f} "
        f"{np.median(arr):12.6f} "
        f"{arr.std():12.6f}"
    )

# %% Cell 154
methods = [
    "Ideal\noptimum",
    "Noisy\nselected",
    "Direct\nselected",
    "ChannelNet\nselected"
]

means = [
    chosen_ideal_opt.mean(),
    chosen_ideal_noisy.mean(),
    chosen_ideal_direct.mean(),
    chosen_ideal_channelnet.mean()
]

stds = [
    chosen_ideal_opt.std(),
    chosen_ideal_noisy.std(),
    chosen_ideal_direct.std(),
    chosen_ideal_channelnet.std()
]

xpos = np.arange(len(methods))

plt.figure(figsize=(8, 4))

plt.bar(xpos, means, yerr=stds, capsize=5)

plt.xticks(xpos, methods)
plt.ylabel("Ideal cost at selected grid optimum")
plt.title("Does mitigation recover better QAOA parameters?")
plt.grid(axis="y")
plt.show()

# %% Cell 155
np.savez(
    "qaoa_mitigation_results_step12.npz",
    gamma_grid_small=gamma_grid_small,
    beta_grid_small=beta_grid_small,
    C_ideal_small=C_ideal_small,
    C_noisy_true_sampled=C_noisy_true_sampled,
    C_mitigated_direct=C_mitigated_direct,
    C_mitigated_channelnet=C_mitigated_channelnet,
    mae_noisy_multi=mae_noisy_multi,
    mae_direct_multi=mae_direct_multi,
    mae_channelnet_multi=mae_channelnet_multi,
    rmse_noisy_multi=rmse_noisy_multi,
    rmse_direct_multi=rmse_direct_multi,
    rmse_channelnet_multi=rmse_channelnet_multi,
    chosen_ideal_noisy=chosen_ideal_noisy,
    chosen_ideal_direct=chosen_ideal_direct,
    chosen_ideal_channelnet=chosen_ideal_channelnet,
    chosen_ideal_opt=chosen_ideal_opt
)

print("Saved QAOA mitigation results.")

# %% Cell 156
def drift_hidden_device(params, rng=None, drift_strength=1.0):
    """
    Create a drifted version of a hidden device.

    This simulates realistic calibration drift:
    - detuning drift
    - pulse amplitude drift
    - phase drift
    - decoherence drift
    - readout drift
    - coupling/ZZ drift

    The original params are not modified.
    """
    if rng is None:
        rng = np.random.default_rng()

    p = dict(params)

    # Detuning drift, GHz converted to rad/ns through two_pi
    p["Delta1"] += two_pi(rng.normal(0.000, 0.0025 * drift_strength))
    p["Delta2"] += two_pi(rng.normal(0.000, 0.0025 * drift_strength))

    # Coupling and ZZ drift
    p["g"] *= rng.normal(1.0, 0.08 * drift_strength)
    p["zeta"] *= rng.normal(1.0, 0.12 * drift_strength)

    # Pulse calibration drift
    p["amp_scale_1"] *= rng.normal(1.0, 0.020 * drift_strength)
    p["amp_scale_2"] *= rng.normal(1.0, 0.020 * drift_strength)

    p["phase_error_1"] += rng.normal(0.0, 0.020 * drift_strength)
    p["phase_error_2"] += rng.normal(0.0, 0.020 * drift_strength)

    # Decoherence drift
    p["T1_1"] *= rng.normal(1.0, 0.12 * drift_strength)
    p["T1_2"] *= rng.normal(1.0, 0.12 * drift_strength)

    p["Tphi_1"] *= rng.normal(1.0, 0.12 * drift_strength)
    p["Tphi_2"] *= rng.normal(1.0, 0.12 * drift_strength)

    # Keep decoherence times positive and reasonable
    p["T1_1"] = float(np.clip(p["T1_1"], 10000.0, 100000.0))
    p["T1_2"] = float(np.clip(p["T1_2"], 10000.0, 100000.0))
    p["Tphi_1"] = float(np.clip(p["Tphi_1"], 8000.0, 100000.0))
    p["Tphi_2"] = float(np.clip(p["Tphi_2"], 8000.0, 100000.0))

    # Readout drift
    for key in ["r01_1", "r10_1", "r01_2", "r10_2"]:
        p[key] += rng.normal(0.0, 0.006 * drift_strength)
        p[key] = float(np.clip(p[key], 0.0, 0.12))

    return p

# %% Cell 157
rng = np.random.default_rng(888)

idx = 0
p_original = params_test_final[idx]
p_drifted = drift_hidden_device(p_original, rng=rng, drift_strength=1.0)

print("Original vs drifted hidden device")
print("-" * 60)

for key in [
    "Delta1", "Delta2", "g", "zeta",
    "amp_scale_1", "amp_scale_2",
    "phase_error_1", "phase_error_2",
    "T1_1", "Tphi_1",
    "r01_1", "r10_1"
]:
    old = p_original[key]
    new = p_drifted[key]

    if key in ["Delta1", "Delta2", "g", "zeta"]:
        old_print = old / (2 * np.pi)
        new_print = new / (2 * np.pi)
        unit = "GHz"
    else:
        old_print = old
        new_print = new
        unit = ""

    print(f"{key:<15s}: {old_print: .6f} -> {new_print: .6f} {unit}")

# %% Cell 158
def generate_drifted_example_from_params(
    original_params,
    rng,
    drift_strength=1.0,
    shots=1024,
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0
):
    """
    Start from an existing hidden device, apply drift,
    then generate finite-shot tomography input and exact reference label.
    """
    drifted_params = drift_hidden_device(
        original_params,
        rng=rng,
        drift_strength=drift_strength
    )

    x_tomo = generate_local_tomography_vector(
        params=drifted_params,
        shots=shots,
        pulse_theta=pulse_theta,
        pulse_phase=pulse_phase,
        pulse_T=pulse_T,
        rng=rng,
        include_decoherence=True
    )

    x_aug = append_shot_feature(x_tomo, shots)

    y_true = reference_two_local_channels_label(
        params=drifted_params,
        spectator_label="0",
        pulse_theta=pulse_theta,
        pulse_phase=pulse_phase,
        pulse_T=pulse_T,
        include_decoherence=True,
        include_readout=True
    )

    return x_aug.astype(np.float32), y_true.astype(np.float32), drifted_params

# %% Cell 159
def build_drifted_test_set(
    original_params_list,
    n_devices=20,
    seed=9000,
    drift_strength=1.0,
    shots=1024
):
    """
    Build drifted evaluation set from original hidden devices.
    """
    rng = np.random.default_rng(seed)

    X_drift = []
    Y_drift_true = []
    params_drift = []

    n_devices = min(n_devices, len(original_params_list))

    for i in range(n_devices):
        x_aug, y_true, p_drift = generate_drifted_example_from_params(
            original_params=original_params_list[i],
            rng=rng,
            drift_strength=drift_strength,
            shots=shots
        )

        X_drift.append(x_aug)
        Y_drift_true.append(y_true)
        params_drift.append(p_drift)

        print(f"Generated drifted device {i+1:02d}/{n_devices}")

    X_drift = np.stack(X_drift).astype(np.float32)
    Y_drift_true = np.stack(Y_drift_true).astype(np.float32)

    return X_drift, Y_drift_true, params_drift

# %% Cell 160
drift_shots = 1024
n_drift_devices = min(20, len(params_test_final))

X_drift, Y_drift_true, params_drift = build_drifted_test_set(
    original_params_list=params_test_final,
    n_devices=n_drift_devices,
    seed=9100,
    drift_strength=1.0,
    shots=drift_shots
)

print("X_drift shape:", X_drift.shape)
print("Y_drift_true shape:", Y_drift_true.shape)

# %% Cell 161
# Standardize drifted inputs using final training statistics
X_drift_s = apply_standardizer(X_drift, X_mean_final, X_std_final)

# ChannelNet prediction
Y_drift_channelnet = predict_physical_y_final(model_final, X_drift_s)

# Direct tomography baseline uses only first 36 tomography features
Y_drift_direct = direct_channels_for_dataset(X_drift[:, :36])

print("Y_drift_channelnet shape:", Y_drift_channelnet.shape)
print("Y_drift_direct shape:", Y_drift_direct.shape)

# %% Cell 162
err_drift_direct = per_example_l2_error(Y_drift_direct, Y_drift_true)
err_drift_channelnet = per_example_l2_error(Y_drift_channelnet, Y_drift_true)

print("Drifted device channel-estimation errors")
print("-" * 60)
print("Direct tomography:")
print("  mean:", err_drift_direct.mean())
print("  median:", np.median(err_drift_direct))
print("  std:", err_drift_direct.std())

print("\nChannelNet:")
print("  mean:", err_drift_channelnet.mean())
print("  median:", np.median(err_drift_channelnet))
print("  std:", err_drift_channelnet.std())

# %% Cell 163
methods = ["Direct\nnoisy tomo", "ChannelNet"]

means = [
    err_drift_direct.mean(),
    err_drift_channelnet.mean()
]

medians = [
    np.median(err_drift_direct),
    np.median(err_drift_channelnet)
]

xpos = np.arange(len(methods))

plt.figure(figsize=(6, 4))
plt.bar(xpos - 0.18, means, width=0.36, label="Mean L2 error")
plt.bar(xpos + 0.18, medians, width=0.36, label="Median L2 error")

plt.xticks(xpos, methods)
plt.ylabel("Channel-label L2 error")
plt.title("Effective channel learning under hidden device drift")
plt.legend()
plt.grid(axis="y")
plt.show()

# %% Cell 164
results_drift = []

for k in range(n_drift_devices):
    result = evaluate_qaoa_mitigation_for_device(
        idx_device=k,
        gamma_grid=gamma_grid_small,
        beta_grid=beta_grid_small,
        C_ideal_grid=C_ideal_small,
        Y_true_all=Y_drift_true,
        Y_direct_all=Y_drift_direct,
        Y_channelnet_all=Y_drift_channelnet,
        qaoa_shots=qaoa_shots,
        seed=12000 + k
    )

    results_drift.append(result)

    print(
        f"Drifted device {k:02d} | "
        f"MAE noisy {result['metrics']['noisy']['MAE']:.4f} | "
        f"direct {result['metrics']['direct']['MAE']:.4f} | "
        f"ChannelNet {result['metrics']['channelnet']['MAE']:.4f}"
    )

# %% Cell 165
mae_drift_noisy = collect_metric(results_drift, "noisy", "MAE")
mae_drift_direct = collect_metric(results_drift, "direct", "MAE")
mae_drift_channelnet = collect_metric(results_drift, "channelnet", "MAE")

rmse_drift_noisy = collect_metric(results_drift, "noisy", "RMSE")
rmse_drift_direct = collect_metric(results_drift, "direct", "RMSE")
rmse_drift_channelnet = collect_metric(results_drift, "channelnet", "RMSE")

print("QAOA mitigation under hidden device drift")
print("-" * 75)
print(f"{'Method':<22s} {'MAE mean':>12s} {'MAE median':>12s} {'RMSE mean':>12s}")
print("-" * 75)

for name, mae_arr, rmse_arr in [
    ("Noisy", mae_drift_noisy, rmse_drift_noisy),
    ("Direct mitigated", mae_drift_direct, rmse_drift_direct),
    ("ChannelNet mitigated", mae_drift_channelnet, rmse_drift_channelnet),
]:
    print(
        f"{name:<22s} "
        f"{mae_arr.mean():12.6f} "
        f"{np.median(mae_arr):12.6f} "
        f"{rmse_arr.mean():12.6f}"
    )

print("\nMean MAE improvement ratios under drift:")
print("Direct:",
      improvement_ratio(mae_drift_noisy.mean(), mae_drift_direct.mean()))
print("ChannelNet:",
      improvement_ratio(mae_drift_noisy.mean(), mae_drift_channelnet.mean()))

# %% Cell 166
methods = ["Noisy", "Direct\nmitigated", "ChannelNet\nmitigated"]

means = [
    mae_drift_noisy.mean(),
    mae_drift_direct.mean(),
    mae_drift_channelnet.mean()
]

stds = [
    mae_drift_noisy.std(),
    mae_drift_direct.std(),
    mae_drift_channelnet.std()
]

xpos = np.arange(len(methods))

plt.figure(figsize=(7, 4))

plt.bar(xpos, means, yerr=stds, capsize=5)

plt.xticks(xpos, methods)
plt.ylabel("MAE vs ideal QAOA landscape")
plt.title("QAOA reliability under hidden device drift")
plt.grid(axis="y")
plt.show()

# %% Cell 167
plt.figure(figsize=(6, 5))

plt.scatter(
    mae_drift_noisy,
    mae_drift_direct,
    label="Direct mitigation",
    alpha=0.8
)

plt.scatter(
    mae_drift_noisy,
    mae_drift_channelnet,
    label="ChannelNet mitigation",
    alpha=0.8
)

max_axis = max(
    mae_drift_noisy.max(),
    mae_drift_direct.max(),
    mae_drift_channelnet.max()
)

plt.plot([0, max_axis], [0, max_axis], linestyle="--")

plt.xlabel("Noisy MAE under drift")
plt.ylabel("Mitigated MAE under drift")
plt.title("Paired reliability improvement under drift")
plt.legend()
plt.grid(True)
plt.show()

# %% Cell 168
np.savez(
    "robustness_under_drift_step14.npz",
    X_drift=X_drift,
    Y_drift_true=Y_drift_true,
    Y_drift_direct=Y_drift_direct,
    Y_drift_channelnet=Y_drift_channelnet,
    err_drift_direct=err_drift_direct,
    err_drift_channelnet=err_drift_channelnet,
    mae_drift_noisy=mae_drift_noisy,
    mae_drift_direct=mae_drift_direct,
    mae_drift_channelnet=mae_drift_channelnet,
    rmse_drift_noisy=rmse_drift_noisy,
    rmse_drift_direct=rmse_drift_direct,
    rmse_drift_channelnet=rmse_drift_channelnet,
    mae_drift_stale=mae_drift_stale if "mae_drift_stale" in globals() else np.array([])
)

print("Saved robustness-under-drift results.")

# %% Cell 169
def add_em_control_params(params, rng=None):
    """
    Add EM/control-line-inspired hidden parameters to an existing device.

    These parameters represent:
    - finite bandwidth / control-line filtering
    - microwave crosstalk
    - DRAG calibration error
    - quasi-static detuning jitter
    - correlated dephasing
    """
    if rng is None:
        rng = np.random.default_rng()

    p = dict(params)

    # Control-line filtering time constants in ns
    # Larger tau = stronger pulse distortion / slower response
    p["filter_tau_1"] = rng.uniform(1.0, 5.0)
    p["filter_tau_2"] = rng.uniform(1.0, 5.0)

    # Microwave crosstalk matrix:
    # drive intended for q1 leaks to q2 and vice versa
    p["xtalk_12"] = rng.normal(0.015, 0.006)  # q2 receives fraction of q1 drive
    p["xtalk_21"] = rng.normal(0.015, 0.006)  # q1 receives fraction of q2 drive

    # Keep crosstalk small and positive-ish
    p["xtalk_12"] = float(np.clip(p["xtalk_12"], 0.0, 0.05))
    p["xtalk_21"] = float(np.clip(p["xtalk_21"], 0.0, 0.05))

    # DRAG coefficient calibration
    # ideal rough scale is around 1/alpha, but we keep it dimensionless here
    p["drag_lambda_1"] = rng.normal(0.35, 0.08)
    p["drag_lambda_2"] = rng.normal(0.35, 0.08)

    p["drag_lambda_1"] = float(np.clip(p["drag_lambda_1"], 0.0, 0.8))
    p["drag_lambda_2"] = float(np.clip(p["drag_lambda_2"], 0.0, 0.8))

    # Quasi-static detuning jitter per experiment, GHz converted later to rad/ns
    p["detuning_jitter_std_1"] = rng.uniform(0.0002, 0.0015)
    p["detuning_jitter_std_2"] = rng.uniform(0.0002, 0.0015)

    # Correlated dephasing rate scale
    # This is a small shared dephasing component.
    p["gamma_phi_corr"] = rng.uniform(0.0, 1.0 / 80000.0)

    return p


def sample_hidden_device_em(rng=None):
    """
    Sample a hidden device with the previous transmon parameters
    plus EM/control-line-inspired parameters.
    """
    if rng is None:
        rng = np.random.default_rng()

    p = sample_hidden_device(rng)
    p = add_em_control_params(p, rng)

    return p

# %% Cell 170
rng = np.random.default_rng(202605)

for k in range(3):
    p = sample_hidden_device_em(rng)

    print(f"\nEM-informed hidden device {k+1}")
    print("-" * 55)
    print(f"Delta1 / 2pi       = {p['Delta1']/(2*np.pi): .5f} GHz")
    print(f"Delta2 / 2pi       = {p['Delta2']/(2*np.pi): .5f} GHz")
    print(f"alpha1 / 2pi       = {p['alpha1']/(2*np.pi): .5f} GHz")
    print(f"alpha2 / 2pi       = {p['alpha2']/(2*np.pi): .5f} GHz")
    print(f"g / 2pi            = {p['g']/(2*np.pi): .5f} GHz")
    print(f"amp_scale_1        = {p['amp_scale_1']: .4f}")
    print(f"phase_error_1      = {p['phase_error_1']: .4f} rad")
    print(f"filter_tau_1       = {p['filter_tau_1']: .3f} ns")
    print(f"filter_tau_2       = {p['filter_tau_2']: .3f} ns")
    print(f"xtalk_12           = {p['xtalk_12']: .4f}")
    print(f"xtalk_21           = {p['xtalk_21']: .4f}")
    print(f"drag_lambda_1      = {p['drag_lambda_1']: .4f}")
    print(f"drag_lambda_2      = {p['drag_lambda_2']: .4f}")
    print(f"detuning jitter q1 = {p['detuning_jitter_std_1']: .5f} GHz")
    print(f"gamma_phi_corr     = {p['gamma_phi_corr']: .3e} 1/ns")

# %% Cell 171
def gaussian_envelope(times, T, area=np.pi, sigma_fraction=0.18):
    """
    Gaussian pulse envelope normalized so that integral Omega(t) dt = area.

    times: array in ns
    T: total pulse duration in ns
    area: desired pulse area, e.g. pi or pi/2
    sigma_fraction: sigma = sigma_fraction * T
    """
    sigma = sigma_fraction * T
    center = T / 2.0

    g = np.exp(-0.5 * ((times - center) / sigma) ** 2)

    # Normalize area numerically
    integral = np.trapz(g, times)
    Omega = area * g / integral

    return Omega


def derivative_numeric(y, times):
    """
    Numerical derivative dy/dt.
    """
    return np.gradient(y, times)


def make_drag_pulse(
    times,
    T,
    theta=np.pi,
    phase=0.0,
    alpha=two_pi(0.220),
    drag_lambda=0.35,
    sigma_fraction=0.18
):
    """
    Generate Gaussian/DRAG quadratures.

    The ideal quadrature is:
        I(t) = Gaussian envelope
        Q(t) = -lambda * dI/dt / alpha

    Then rotate by phase.
    """
    I_env = gaussian_envelope(
        times=times,
        T=T,
        area=theta,
        sigma_fraction=sigma_fraction
    )

    dI_dt = derivative_numeric(I_env, times)

    # DRAG quadrature
    Q_env = -drag_lambda * dI_dt / alpha

    # Apply phase rotation
    Ix = I_env * np.cos(phase) - Q_env * np.sin(phase)
    Iy = I_env * np.sin(phase) + Q_env * np.cos(phase)

    return Ix, Iy, I_env, Q_env

# %% Cell 172
T_test = 40.0
n_steps_test = 300
times_test = np.linspace(0.0, T_test, n_steps_test)

Ix, Iy, I_env, Q_env = make_drag_pulse(
    times=times_test,
    T=T_test,
    theta=np.pi,
    phase=0.0,
    alpha=two_pi(0.220),
    drag_lambda=0.35,
    sigma_fraction=0.18
)

print("Numerical area of Ix:", np.trapz(Ix, times_test))
print("Target area pi:", np.pi)
print("Max |Iy| / max |Ix|:", np.max(np.abs(Iy)) / np.max(np.abs(Ix)))

plt.figure(figsize=(8, 4))
plt.plot(times_test, Ix, label="I / X quadrature")
plt.plot(times_test, Iy, label="Q / Y quadrature")
plt.xlabel("Time (ns)")
plt.ylabel("Drive amplitude (rad/ns)")
plt.title("Gaussian/DRAG pulse")
plt.legend()
plt.grid(True)
plt.show()

# %% Cell 173
def lowpass_filter_pulse(u, times, tau):
    """
    First-order low-pass filtering of a pulse.

    u: input array
    tau: filter time constant in ns
    """
    u = np.asarray(u, dtype=np.float64)

    if tau <= 1e-12:
        return u.copy()

    y = np.zeros_like(u)
    y[0] = u[0]

    for k in range(1, len(u)):
        dt = times[k] - times[k - 1]
        alpha = dt / (tau + dt)
        y[k] = y[k - 1] + alpha * (u[k] - y[k - 1])

    return y

# %% Cell 174
def make_em_aware_drive_waveforms(
    params,
    target_qubit=1,
    theta=np.pi,
    phase=0.0,
    T=40.0,
    n_steps=300,
    sigma_fraction=0.18
):
    """
    Build EM-informed control waveforms for both qubits.

    Returns:
        times
        drives dict with keys:
            q1_x, q1_y, q2_x, q2_y
            intended_q1_x, intended_q1_y, intended_q2_x, intended_q2_y
    """
    times = np.linspace(0.0, T, n_steps)

    # Start with no intended drive on either qubit
    q1_x_int = np.zeros_like(times)
    q1_y_int = np.zeros_like(times)
    q2_x_int = np.zeros_like(times)
    q2_y_int = np.zeros_like(times)

    if target_qubit == 1:
        q1_x_int, q1_y_int, _, _ = make_drag_pulse(
            times=times,
            T=T,
            theta=theta,
            phase=phase,
            alpha=params["alpha1"],
            drag_lambda=params["drag_lambda_1"],
            sigma_fraction=sigma_fraction
        )
    elif target_qubit == 2:
        q2_x_int, q2_y_int, _, _ = make_drag_pulse(
            times=times,
            T=T,
            theta=theta,
            phase=phase,
            alpha=params["alpha2"],
            drag_lambda=params["drag_lambda_2"],
            sigma_fraction=sigma_fraction
        )
    else:
        raise ValueError("target_qubit must be 1 or 2")

    # Apply amplitude scale and phase errors approximately
    # Phase error rotates x/y quadratures.
    def apply_amp_phase(x, y, amp_scale, phase_error):
        xr = amp_scale * (x * np.cos(phase_error) - y * np.sin(phase_error))
        yr = amp_scale * (x * np.sin(phase_error) + y * np.cos(phase_error))
        return xr, yr

    q1_x_cal, q1_y_cal = apply_amp_phase(
        q1_x_int,
        q1_y_int,
        params["amp_scale_1"],
        params["phase_error_1"]
    )

    q2_x_cal, q2_y_cal = apply_amp_phase(
        q2_x_int,
        q2_y_int,
        params["amp_scale_2"],
        params["phase_error_2"]
    )

    # Crosstalk before filtering
    # q2 receives xtalk_12 of q1 intended drive
    # q1 receives xtalk_21 of q2 intended drive
    q1_x_cross = q1_x_cal + params["xtalk_21"] * q2_x_cal
    q1_y_cross = q1_y_cal + params["xtalk_21"] * q2_y_cal

    q2_x_cross = q2_x_cal + params["xtalk_12"] * q1_x_cal
    q2_y_cross = q2_y_cal + params["xtalk_12"] * q1_y_cal

    # Finite-bandwidth filtering
    q1_x_real = lowpass_filter_pulse(q1_x_cross, times, params["filter_tau_1"])
    q1_y_real = lowpass_filter_pulse(q1_y_cross, times, params["filter_tau_1"])

    q2_x_real = lowpass_filter_pulse(q2_x_cross, times, params["filter_tau_2"])
    q2_y_real = lowpass_filter_pulse(q2_y_cross, times, params["filter_tau_2"])

    drives = {
        "q1_x": q1_x_real,
        "q1_y": q1_y_real,
        "q2_x": q2_x_real,
        "q2_y": q2_y_real,

        "intended_q1_x": q1_x_int,
        "intended_q1_y": q1_y_int,
        "intended_q2_x": q2_x_int,
        "intended_q2_y": q2_y_int,
    }

    return times, drives

# %% Cell 175
rng = np.random.default_rng(333)
p_em = sample_hidden_device_em(rng)

times_em, drives_em = make_em_aware_drive_waveforms(
    params=p_em,
    target_qubit=1,
    theta=np.pi,
    phase=0.0,
    T=40.0,
    n_steps=300
)

plt.figure(figsize=(10, 5))

plt.plot(times_em, drives_em["intended_q1_x"], label="intended q1 X", linewidth=2)
plt.plot(times_em, drives_em["q1_x"], label="real q1 X after filtering", linestyle="--")
plt.plot(times_em, drives_em["q2_x"], label="crosstalk q2 X", linestyle=":")

plt.xlabel("Time (ns)")
plt.ylabel("Drive amplitude (rad/ns)")
plt.title("EM-informed drive distortion and crosstalk")
plt.legend()
plt.grid(True)
plt.show()

print("q1 intended X area:", np.trapz(drives_em["intended_q1_x"], times_em))
print("q1 real X area:    ", np.trapz(drives_em["q1_x"], times_em))
print("q2 leaked X area:  ", np.trapz(drives_em["q2_x"], times_em))
print("crosstalk ratio area q2/q1:",
      np.trapz(np.abs(drives_em["q2_x"]), times_em) /
      np.trapz(np.abs(drives_em["q1_x"]), times_em))

# %% Cell 176
def control_hamiltonian_from_quadratures(q1_x, q1_y, q2_x, q2_y):
    """
    Build instantaneous control Hamiltonian from real quadrature amplitudes.
    """
    H_q1 = (
        0.5 * q1_x * (a1 + adag1)
        +
        0.5 * q1_y * (1j * (adag1 - a1))
    )

    H_q2 = (
        0.5 * q2_x * (a2 + adag2)
        +
        0.5 * q2_y * (1j * (adag2 - a2))
    )

    return H_q1 + H_q2

# %% Cell 177
def collapse_operators_em(params, include_correlated_dephasing=True):
    """
    Collapse operators with optional correlated dephasing.
    """
    cops = collapse_operators(params)

    if include_correlated_dephasing:
        gamma_c = params.get("gamma_phi_corr", 0.0)
        if gamma_c > 0:
            cops.append(np.sqrt(gamma_c) * (n1 + n2))

    return cops

# %% Cell 178
def evolve_rho_em_pulse(
    rho0,
    params,
    target_qubit=1,
    theta=np.pi,
    phase=0.0,
    T=40.0,
    n_steps=160,
    sigma_fraction=0.18,
    include_decoherence=True,
    include_correlated_dephasing=True,
    include_detuning_jitter=True,
    rng=None
):
    """
    Time-dependent EM-informed pulse evolution.

    Includes:
    - Gaussian/DRAG pulse
    - finite-bandwidth filtering
    - crosstalk
    - optional quasi-static detuning jitter per experiment
    - optional decoherence and correlated dephasing
    """
    if rng is None:
        rng = np.random.default_rng()

    # Copy params and optionally add per-experiment detuning jitter
    p = dict(params)

    if include_detuning_jitter:
        p["Delta1"] = p["Delta1"] + two_pi(rng.normal(0.0, p["detuning_jitter_std_1"]))
        p["Delta2"] = p["Delta2"] + two_pi(rng.normal(0.0, p["detuning_jitter_std_2"]))

    H0 = build_H0(p)

    times, drives = make_em_aware_drive_waveforms(
        params=p,
        target_qubit=target_qubit,
        theta=theta,
        phase=phase,
        T=T,
        n_steps=n_steps,
        sigma_fraction=sigma_fraction
    )

    rho = rho0.copy()

    if include_decoherence:
        cops = collapse_operators_em(
            p,
            include_correlated_dephasing=include_correlated_dephasing
        )
    else:
        cops = []

    for k in range(n_steps - 1):
        dt = times[k + 1] - times[k]

        Hc = control_hamiltonian_from_quadratures(
            drives["q1_x"][k],
            drives["q1_y"][k],
            drives["q2_x"][k],
            drives["q2_y"][k]
        )

        H = H0 + Hc

        if include_decoherence:
            L = liouvillian(H, cops)
            rho_vec = expm(L * dt) @ vec(rho)
            rho = unvec(rho_vec, rho.shape[0])
        else:
            U = expm(-1j * H * dt)
            rho = U @ rho @ dagger(U)

        rho = clean_density_matrix(rho)

    return rho, times, drives

# %% Cell 179
rng = np.random.default_rng(444)

p_em = sample_hidden_device_em(rng)
rho00 = ket_to_rho(comp_states["00"])

rho_xpi_em, times_em, drives_em = evolve_rho_em_pulse(
    rho0=rho00,
    params=p_em,
    target_qubit=1,
    theta=np.pi,
    phase=0.0,
    T=40.0,
    n_steps=160,
    include_decoherence=True,
    include_correlated_dephasing=True,
    include_detuning_jitter=True,
    rng=rng
)

pops_em = computational_populations_rho(rho_xpi_em)

print("Trace:", np.trace(rho_xpi_em))
print("Hermiticity error:", fro_norm(rho_xpi_em - dagger(rho_xpi_em)))

print("\nEM-aware noisy X_pi final populations:")
for k, v in pops_em.items():
    print(f"{k}: {v:.6f}")

print("\nLocal Bloch vector qubit 1:")
print(bloch_vector_from_rho_qubit(rho_xpi_em, qubit=1))

print("\nLocal Bloch vector qubit 2:")
print(bloch_vector_from_rho_qubit(rho_xpi_em, qubit=2))

# %% Cell 180
# Use same EM params but square-pulse evolution ignores EM-specific pulse distortions
rho_xpi_square = evolve_rho_square_pulse(
    rho0=rho00,
    params=p_em,
    qubit=1,
    theta=np.pi,
    phase=0.0,
    T=40.0,
    include_decoherence=True
)

pops_square = computational_populations_rho(rho_xpi_square)
pops_em = computational_populations_rho(rho_xpi_em)

print("Square pulse populations:")
for k, v in pops_square.items():
    print(f"{k}: {v:.6f}")

print("\nEM-aware pulse populations:")
for k, v in pops_em.items():
    print(f"{k}: {v:.6f}")

print("\nDifference in leakage:")
print("square leakage:", pops_square["leak"])
print("EM leakage:    ", pops_em["leak"])

# %% Cell 181
plt.figure(figsize=(10, 5))

plt.plot(times_em, drives_em["q1_x"], label="q1 X real")
plt.plot(times_em, drives_em["q1_y"], label="q1 Y real / DRAG")
plt.plot(times_em, drives_em["q2_x"], label="q2 X leaked", linestyle="--")
plt.plot(times_em, drives_em["q2_y"], label="q2 Y leaked", linestyle="--")

plt.xlabel("Time (ns)")
plt.ylabel("Drive amplitude (rad/ns)")
plt.title("Real EM-informed drives used in evolution")
plt.legend()
plt.grid(True)
plt.show()

labels = ["00", "01", "10", "11", "leak"]
square_vals = [pops_square[k] for k in labels]
em_vals = [pops_em[k] for k in labels]

xpos = np.arange(len(labels))

plt.figure(figsize=(7, 4))
plt.bar(xpos - 0.18, square_vals, width=0.36, label="square")
plt.bar(xpos + 0.18, em_vals, width=0.36, label="EM-aware")

plt.xticks(xpos, labels)
plt.ylabel("Final population")
plt.title("Square pulse vs EM-aware pulse")
plt.legend()
plt.grid(axis="y")
plt.show()

# %% Cell 182
def exact_output_bloch_for_local_input_em(
    params,
    target_qubit=1,
    local_label="0",
    spectator_label="0",
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=60.0,
    n_steps_em=100,
    sigma_fraction=0.18,
    include_decoherence=True,
    include_correlated_dephasing=True,
    include_detuning_jitter=True,
    include_readout=True,
    rng=None
):
    """
    EM-aware exact local output Bloch vector for one local input state.

    This uses:
    - Gaussian/DRAG pulse
    - finite-bandwidth filtering
    - crosstalk
    - optional detuning jitter
    - Lindblad decoherence
    - optional correlated dephasing
    - optional readout error

    Returns:
        r_out = [<X>, <Y>, <Z>]
    """
    if rng is None:
        rng = np.random.default_rng()

    psi0 = prepare_local_tomography_state(
        target_qubit=target_qubit,
        local_label=local_label,
        spectator_label=spectator_label
    )
    rho0 = ket_to_rho(psi0)

    rho_out, _, _ = evolve_rho_em_pulse(
        rho0=rho0,
        params=params,
        target_qubit=target_qubit,
        theta=pulse_theta,
        phase=pulse_phase,
        T=pulse_T,
        n_steps=n_steps_em,
        sigma_fraction=sigma_fraction,
        include_decoherence=include_decoherence,
        include_correlated_dephasing=include_correlated_dephasing,
        include_detuning_jitter=include_detuning_jitter,
        rng=rng
    )

    r_out = []

    for basis_label in measurement_labels:
        if include_readout:
            m = exact_measured_expectation_with_readout(
                rho_out,
                params,
                qubit=target_qubit,
                basis_label=basis_label
            )
        else:
            _, _, m = ideal_measurement_probabilities(
                rho_out,
                qubit=target_qubit,
                basis_label=basis_label
            )

        r_out.append(m)

    return np.array(r_out, dtype=np.float64)

# %% Cell 183
def finite_shot_output_bloch_for_local_input_em(
    params,
    target_qubit=1,
    local_label="0",
    spectator_label="0",
    shots=1024,
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=60.0,
    n_steps_em=100,
    sigma_fraction=0.18,
    include_decoherence=True,
    include_correlated_dephasing=True,
    include_detuning_jitter=True,
    rng=None
):
    """
    EM-aware finite-shot local output Bloch vector.

    Returns:
        r_hat = finite-shot measured [<X>, <Y>, <Z>]
    """
    if rng is None:
        rng = np.random.default_rng()

    psi0 = prepare_local_tomography_state(
        target_qubit=target_qubit,
        local_label=local_label,
        spectator_label=spectator_label
    )
    rho0 = ket_to_rho(psi0)

    rho_out, _, _ = evolve_rho_em_pulse(
        rho0=rho0,
        params=params,
        target_qubit=target_qubit,
        theta=pulse_theta,
        phase=pulse_phase,
        T=pulse_T,
        n_steps=n_steps_em,
        sigma_fraction=sigma_fraction,
        include_decoherence=include_decoherence,
        include_correlated_dephasing=include_correlated_dephasing,
        include_detuning_jitter=include_detuning_jitter,
        rng=rng
    )

    r_hat = []

    for basis_label in measurement_labels:
        meas = sample_binary_measurement(
            rho=rho_out,
            params=params,
            qubit=target_qubit,
            basis_label=basis_label,
            shots=shots,
            rng=rng
        )
        r_hat.append(meas["m_hat"])

    return np.array(r_hat, dtype=np.float32)

# %% Cell 184
def generate_local_tomography_vector_em(
    params,
    shots=1024,
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=60.0,
    n_steps_em=100,
    sigma_fraction=0.18,
    spectator_label="0",
    include_decoherence=True,
    include_correlated_dephasing=True,
    include_detuning_jitter=True,
    rng=None
):
    """
    Generate EM-aware finite-shot local tomography vector.

    Feature order:
        qubit 1, six input states, X/Y/Z
        qubit 2, six input states, X/Y/Z

    Output:
        x_tomo_em shape: (36,)
    """
    if rng is None:
        rng = np.random.default_rng()

    features = []

    for target_qubit in [1, 2]:
        for state_label in tomography_input_labels:
            r_hat = finite_shot_output_bloch_for_local_input_em(
                params=params,
                target_qubit=target_qubit,
                local_label=state_label,
                spectator_label=spectator_label,
                shots=shots,
                pulse_theta=pulse_theta,
                pulse_phase=pulse_phase,
                pulse_T=pulse_T,
                n_steps_em=n_steps_em,
                sigma_fraction=sigma_fraction,
                include_decoherence=include_decoherence,
                include_correlated_dephasing=include_correlated_dephasing,
                include_detuning_jitter=include_detuning_jitter,
                rng=rng
            )

            features.extend(list(r_hat))

    return np.array(features, dtype=np.float32)

# %% Cell 185
def averaged_exact_output_bloch_for_local_input_em(
    params,
    target_qubit=1,
    local_label="0",
    spectator_label="0",
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=60.0,
    n_steps_em=100,
    sigma_fraction=0.18,
    include_decoherence=True,
    include_correlated_dephasing=True,
    include_detuning_jitter=True,
    include_readout=True,
    n_avg_ref=3,
    rng=None
):
    """
    Average the EM-aware exact output Bloch vector over several
    quasi-static detuning-jitter realizations.

    This gives a stable reference effective channel.
    """
    if rng is None:
        rng = np.random.default_rng()

    r_accum = np.zeros(3, dtype=np.float64)

    for _ in range(n_avg_ref):
        r_out = exact_output_bloch_for_local_input_em(
            params=params,
            target_qubit=target_qubit,
            local_label=local_label,
            spectator_label=spectator_label,
            pulse_theta=pulse_theta,
            pulse_phase=pulse_phase,
            pulse_T=pulse_T,
            n_steps_em=n_steps_em,
            sigma_fraction=sigma_fraction,
            include_decoherence=include_decoherence,
            include_correlated_dephasing=include_correlated_dephasing,
            include_detuning_jitter=include_detuning_jitter,
            include_readout=include_readout,
            rng=rng
        )

        r_accum += r_out

    return r_accum / n_avg_ref

# %% Cell 186
def reference_local_affine_channel_em(
    params,
    target_qubit=1,
    spectator_label="0",
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=60.0,
    n_steps_em=100,
    sigma_fraction=0.18,
    include_decoherence=True,
    include_correlated_dephasing=True,
    include_detuning_jitter=True,
    include_readout=True,
    n_avg_ref=3,
    rng=None
):
    """
    Compute EM-aware reference effective affine Bloch map:

        r_out = A r_in + b

    for one target qubit.
    """
    if rng is None:
        rng = np.random.default_rng()

    r_in_list = []
    r_out_list = []

    for local_label in tomography_input_labels:
        r_in = expected_bloch[local_label]

        r_out = averaged_exact_output_bloch_for_local_input_em(
            params=params,
            target_qubit=target_qubit,
            local_label=local_label,
            spectator_label=spectator_label,
            pulse_theta=pulse_theta,
            pulse_phase=pulse_phase,
            pulse_T=pulse_T,
            n_steps_em=n_steps_em,
            sigma_fraction=sigma_fraction,
            include_decoherence=include_decoherence,
            include_correlated_dephasing=include_correlated_dephasing,
            include_detuning_jitter=include_detuning_jitter,
            include_readout=include_readout,
            n_avg_ref=n_avg_ref,
            rng=rng
        )

        r_in_list.append(r_in)
        r_out_list.append(r_out)

    A, b = fit_affine_bloch_map(r_in_list, r_out_list)

    return A, b

# %% Cell 187
def reference_two_local_channels_label_em(
    params,
    spectator_label="0",
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=60.0,
    n_steps_em=100,
    sigma_fraction=0.18,
    include_decoherence=True,
    include_correlated_dephasing=True,
    include_detuning_jitter=True,
    include_readout=True,
    n_avg_ref=3,
    rng=None
):
    """
    Compute EM-aware full length-24 reference label:
        [A1, b1, A2, b2]
    """
    if rng is None:
        rng = np.random.default_rng()

    A1_ref, b1_ref = reference_local_affine_channel_em(
        params=params,
        target_qubit=1,
        spectator_label=spectator_label,
        pulse_theta=pulse_theta,
        pulse_phase=pulse_phase,
        pulse_T=pulse_T,
        n_steps_em=n_steps_em,
        sigma_fraction=sigma_fraction,
        include_decoherence=include_decoherence,
        include_correlated_dephasing=include_correlated_dephasing,
        include_detuning_jitter=include_detuning_jitter,
        include_readout=include_readout,
        n_avg_ref=n_avg_ref,
        rng=rng
    )

    A2_ref, b2_ref = reference_local_affine_channel_em(
        params=params,
        target_qubit=2,
        spectator_label=spectator_label,
        pulse_theta=pulse_theta,
        pulse_phase=pulse_phase,
        pulse_T=pulse_T,
        n_steps_em=n_steps_em,
        sigma_fraction=sigma_fraction,
        include_decoherence=include_decoherence,
        include_correlated_dephasing=include_correlated_dephasing,
        include_detuning_jitter=include_detuning_jitter,
        include_readout=include_readout,
        n_avg_ref=n_avg_ref,
        rng=rng
    )

    y = flatten_two_local_channels(A1_ref, b1_ref, A2_ref, b2_ref)

    return y.astype(np.float32)

# %% Cell 188
rng = np.random.default_rng(1601)

p_em_test = sample_hidden_device_em(rng)

x_em = generate_local_tomography_vector_em(
    params=p_em_test,
    shots=1024,
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=60.0,
    n_steps_em=100,
    sigma_fraction=0.18,
    include_decoherence=True,
    include_correlated_dephasing=True,
    include_detuning_jitter=True,
    rng=rng
)

print("EM-aware tomography vector shape:", x_em.shape)
print("First 12 entries:")
print(x_em[:12])
print("Minimum:", x_em.min())
print("Maximum:", x_em.max())

# %% Cell 189
plt.figure(figsize=(10, 4))
plt.plot(x_em, marker="o")
plt.axhline(1.0, linestyle="--")
plt.axhline(-1.0, linestyle="--")
plt.xlabel("Tomography feature index")
plt.ylabel("Finite-shot expectation")
plt.title("One EM-aware finite-shot local tomography vector")
plt.grid(True)
plt.show()

# %% Cell 190
rng = np.random.default_rng(1602)

y_em_ref = reference_two_local_channels_label_em(
    params=p_em_test,
    spectator_label="0",
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=60.0,
    n_steps_em=100,
    sigma_fraction=0.18,
    include_decoherence=True,
    include_correlated_dephasing=True,
    include_detuning_jitter=True,
    include_readout=True,
    n_avg_ref=3,
    rng=rng
)

print("EM-aware reference label shape:", y_em_ref.shape)
print("First 12 entries:")
print(y_em_ref[:12])

A1_em, b1_em, A2_em, b2_em = unflatten_two_local_channels(y_em_ref)

print("\nA1 EM reference:")
print(A1_em)

print("\nb1 EM reference:")
print(b1_em)

print("\nA2 EM reference:")
print(A2_em)

print("\nb2 EM reference:")
print(b2_em)

# %% Cell 191
def check_affine_map_fit_em(
    params,
    target_qubit=1,
    spectator_label="0",
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=60.0,
    n_steps_em=100,
    sigma_fraction=0.18,
    include_decoherence=True,
    include_correlated_dephasing=True,
    include_detuning_jitter=True,
    include_readout=True,
    n_avg_ref=3,
    seed=1700
):
    """
    Check whether the EM-aware affine map reproduces the averaged exact outputs.
    """
    rng_fit = np.random.default_rng(seed)

    A, b = reference_local_affine_channel_em(
        params=params,
        target_qubit=target_qubit,
        spectator_label=spectator_label,
        pulse_theta=pulse_theta,
        pulse_phase=pulse_phase,
        pulse_T=pulse_T,
        n_steps_em=n_steps_em,
        sigma_fraction=sigma_fraction,
        include_decoherence=include_decoherence,
        include_correlated_dephasing=include_correlated_dephasing,
        include_detuning_jitter=include_detuning_jitter,
        include_readout=include_readout,
        n_avg_ref=n_avg_ref,
        rng=rng_fit
    )

    print(f"\nEM-aware affine map check for qubit {target_qubit}")
    print("-" * 80)

    max_err = 0.0
    mean_errs = []

    # Use a fresh RNG for checking averaged outputs
    rng_check = np.random.default_rng(seed + 999)

    for local_label in tomography_input_labels:
        r_in = expected_bloch[local_label]

        r_exact_avg = averaged_exact_output_bloch_for_local_input_em(
            params=params,
            target_qubit=target_qubit,
            local_label=local_label,
            spectator_label=spectator_label,
            pulse_theta=pulse_theta,
            pulse_phase=pulse_phase,
            pulse_T=pulse_T,
            n_steps_em=n_steps_em,
            sigma_fraction=sigma_fraction,
            include_decoherence=include_decoherence,
            include_correlated_dephasing=include_correlated_dephasing,
            include_detuning_jitter=include_detuning_jitter,
            include_readout=include_readout,
            n_avg_ref=n_avg_ref,
            rng=rng_check
        )

        r_pred = A @ r_in + b
        err = np.linalg.norm(r_pred - r_exact_avg)

        max_err = max(max_err, err)
        mean_errs.append(err)

        print(
            f"input {local_label:>2s} | "
            f"avg exact [{r_exact_avg[0]: .3f}, {r_exact_avg[1]: .3f}, {r_exact_avg[2]: .3f}] | "
            f"affine [{r_pred[0]: .3f}, {r_pred[1]: .3f}, {r_pred[2]: .3f}] | "
            f"err {err:.3e}"
        )

    print("Mean fit/check error:", np.mean(mean_errs))
    print("Maximum fit/check error:", max_err)

    return max_err, np.mean(mean_errs)

# %% Cell 192
err_q1_em, mean_err_q1_em = check_affine_map_fit_em(
    p_em_test,
    target_qubit=1,
    n_steps_em=100,
    n_avg_ref=3,
    seed=1801
)

err_q2_em, mean_err_q2_em = check_affine_map_fit_em(
    p_em_test,
    target_qubit=2,
    n_steps_em=100,
    n_avg_ref=3,
    seed=1802
)

# %% Cell 193
# The EM device contains all old parameters too, so we can compare old square-pulse label vs EM-aware label.

y_square_ref = reference_two_local_channels_label(
    params=p_em_test,
    spectator_label="0",
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=60.0,
    include_decoherence=True,
    include_readout=True
)

y_em_ref = reference_two_local_channels_label_em(
    params=p_em_test,
    spectator_label="0",
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=60.0,
    n_steps_em=100,
    sigma_fraction=0.18,
    include_decoherence=True,
    include_correlated_dephasing=True,
    include_detuning_jitter=True,
    include_readout=True,
    n_avg_ref=3,
    rng=np.random.default_rng(1940)
)

print("L2 difference between square-pulse and EM-aware effective labels:")
print(np.linalg.norm(y_square_ref - y_em_ref))

plt.figure(figsize=(10, 4))
plt.plot(y_square_ref, marker="o", label="Square-pulse label")
plt.plot(y_em_ref, marker="s", label="EM-aware label", alpha=0.8)
plt.xlabel("Channel-label index")
plt.ylabel("Channel parameter")
plt.title("Square-pulse vs EM-aware effective local channels")
plt.legend()
plt.grid(True)
plt.show()

# %% Cell 194
def generate_one_supervised_example_em(
    rng,
    shots=1024,
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=60.0,
    n_steps_em=100,
    sigma_fraction=0.18,
    n_avg_ref=3,
    include_decoherence=True,
    include_correlated_dephasing=True,
    include_detuning_jitter=True,
    include_readout=True
):
    """
    Generate one EM-aware supervised example.

    Hidden EM-aware device:
        randomized transmon + control-line parameters.

    Input x:
        finite-shot local tomography vector, shape (36,)

    Label y:
        averaged effective local affine channels, shape (24,)
    """
    params = sample_hidden_device_em(rng)

    x = generate_local_tomography_vector_em(
        params=params,
        shots=shots,
        pulse_theta=pulse_theta,
        pulse_phase=pulse_phase,
        pulse_T=pulse_T,
        n_steps_em=n_steps_em,
        sigma_fraction=sigma_fraction,
        include_decoherence=include_decoherence,
        include_correlated_dephasing=include_correlated_dephasing,
        include_detuning_jitter=include_detuning_jitter,
        rng=rng
    )

    y = reference_two_local_channels_label_em(
        params=params,
        spectator_label="0",
        pulse_theta=pulse_theta,
        pulse_phase=pulse_phase,
        pulse_T=pulse_T,
        n_steps_em=n_steps_em,
        sigma_fraction=sigma_fraction,
        include_decoherence=include_decoherence,
        include_correlated_dephasing=include_correlated_dephasing,
        include_detuning_jitter=include_detuning_jitter,
        include_readout=include_readout,
        n_avg_ref=n_avg_ref,
        rng=rng
    )

    return x.astype(np.float32), y.astype(np.float32), params

# %% Cell 195
rng = np.random.default_rng(1960)

x_one_em, y_one_em, p_one_em = generate_one_supervised_example_em(
    rng=rng,
    shots=1024,
    pulse_T=60.0,
    n_steps_em=100,
    n_avg_ref=3
)

print("x_one_em shape:", x_one_em.shape)
print("y_one_em shape:", y_one_em.shape)

print("\nx range:", x_one_em.min(), x_one_em.max())
print("y range:", y_one_em.min(), y_one_em.max())

A1_one_em, b1_one_em, A2_one_em, b2_one_em = unflatten_two_local_channels(y_one_em)

print("\nA1 one EM:")
print(A1_one_em)

print("\nb1 one EM:")
print(b1_one_em)

# %% Cell 196
shot_choices_em = np.array([128, 512, 1024, 4096], dtype=int)

def generate_one_mixedshot_example_em(
    rng,
    shot_choices=shot_choices_em,
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=60.0,
    n_steps_em=80,
    sigma_fraction=0.18,
    n_avg_ref=2,
    include_decoherence=True,
    include_correlated_dephasing=True,
    include_detuning_jitter=True,
    include_readout=True
):
    """
    Generate one EM-aware mixed-shot supervised example.

    Input:
        finite-shot EM-aware tomography vector + log10(shots), shape (37,)

    Label:
        averaged EM-aware effective local affine channels, shape (24,)
    """
    shots = int(rng.choice(shot_choices))

    x_tomo, y, params = generate_one_supervised_example_em(
        rng=rng,
        shots=shots,
        pulse_theta=pulse_theta,
        pulse_phase=pulse_phase,
        pulse_T=pulse_T,
        n_steps_em=n_steps_em,
        sigma_fraction=sigma_fraction,
        n_avg_ref=n_avg_ref,
        include_decoherence=include_decoherence,
        include_correlated_dephasing=include_correlated_dephasing,
        include_detuning_jitter=include_detuning_jitter,
        include_readout=include_readout
    )

    x_aug = append_shot_feature(x_tomo, shots)

    return x_aug.astype(np.float32), y.astype(np.float32), params, shots

# %% Cell 197
def build_mixedshot_dataset_em(
    n_examples,
    seed=0,
    shot_choices=shot_choices_em,
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=60.0,
    n_steps_em=80,
    sigma_fraction=0.18,
    n_avg_ref=2,
    include_decoherence=True,
    include_correlated_dephasing=True,
    include_detuning_jitter=True,
    include_readout=True,
    print_every=5
):
    """
    Build EM-aware mixed-shot dataset.

    X shape: (n_examples, 37)
    Y shape: (n_examples, 24)
    """
    rng = np.random.default_rng(seed)

    X = []
    Y = []
    hidden_params_list = []
    shots_list = []

    t0 = time.time()

    for i in range(n_examples):
        x, y, params, shots = generate_one_mixedshot_example_em(
            rng=rng,
            shot_choices=shot_choices,
            pulse_theta=pulse_theta,
            pulse_phase=pulse_phase,
            pulse_T=pulse_T,
            n_steps_em=n_steps_em,
            sigma_fraction=sigma_fraction,
            n_avg_ref=n_avg_ref,
            include_decoherence=include_decoherence,
            include_correlated_dephasing=include_correlated_dephasing,
            include_detuning_jitter=include_detuning_jitter,
            include_readout=include_readout
        )

        X.append(x)
        Y.append(y)
        hidden_params_list.append(params)
        shots_list.append(shots)

        if (i + 1) % print_every == 0 or (i + 1) == n_examples:
            elapsed = time.time() - t0
            print(
                f"Generated {i+1:4d}/{n_examples} EM-aware examples "
                f"| elapsed {elapsed:.1f} s"
            )

    X = np.stack(X).astype(np.float32)
    Y = np.stack(Y).astype(np.float32)
    shots_array = np.array(shots_list, dtype=int)

    return X, Y, hidden_params_list, shots_array

# %% Cell 198
# Debug/medium EM-aware dataset
N_train_em = 60
N_val_em = 20
N_test_em = 20

# Runtime-control parameters
pulse_T_em = 60.0
n_steps_em_dataset = 80
n_avg_ref_em = 2

print("EM-aware dataset size:")
print("Train:", N_train_em)
print("Val:  ", N_val_em)
print("Test: ", N_test_em)

print("\nEM simulation settings:")
print("pulse_T_em:", pulse_T_em)
print("n_steps_em_dataset:", n_steps_em_dataset)
print("n_avg_ref_em:", n_avg_ref_em)

# %% Cell 199
X_train_em, Y_train_em, params_train_em, shots_train_em = build_mixedshot_dataset_em(
    n_examples=N_train_em,
    seed=4100,
    pulse_T=pulse_T_em,
    n_steps_em=n_steps_em_dataset,
    n_avg_ref=n_avg_ref_em,
    print_every=5
)

X_val_em, Y_val_em, params_val_em, shots_val_em = build_mixedshot_dataset_em(
    n_examples=N_val_em,
    seed=4200,
    pulse_T=pulse_T_em,
    n_steps_em=n_steps_em_dataset,
    n_avg_ref=n_avg_ref_em,
    print_every=5
)

X_test_em, Y_test_em, params_test_em, shots_test_em = build_mixedshot_dataset_em(
    n_examples=N_test_em,
    seed=4300,
    pulse_T=pulse_T_em,
    n_steps_em=n_steps_em_dataset,
    n_avg_ref=n_avg_ref_em,
    print_every=5
)

# %% [markdown]
# PArtial TOmo

# %% Cell 201
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.linalg import expm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(7)
torch.manual_seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# Basic helpers
# ============================================================

def dagger(A):
    return A.conj().T

def kron(*ops):
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

def fro_norm(A):
    return np.linalg.norm(A, ord="fro")

def two_pi(x):
    return 2 * np.pi * x

def trace_expectation(rho, O):
    val = np.trace(rho @ O)
    return float(np.real_if_close(val))

# ============================================================
# Two-transmon/qutrit Hilbert space
# ============================================================

d = 3
I3 = np.eye(d, dtype=complex)
I9 = np.eye(d * d, dtype=complex)

# Qutrit annihilation operator
a = np.zeros((d, d), dtype=complex)
for n in range(1, d):
    a[n - 1, n] = np.sqrt(n)

adag = dagger(a)
n_op = adag @ a

# Two-transmon operators
a1 = kron(a, I3)
a2 = kron(I3, a)

adag1 = dagger(a1)
adag2 = dagger(a2)

n1 = adag1 @ a1
n2 = adag2 @ a2

def two_qutrit_basis(i, j):
    return kron(basis(d, i), basis(d, j))

comp_states = {
    "00": two_qutrit_basis(0, 0),
    "01": two_qutrit_basis(0, 1),
    "10": two_qutrit_basis(1, 0),
    "11": two_qutrit_basis(1, 1),
}

P_comp = sum(v @ dagger(v) for v in comp_states.values())

def op_from_terms(terms):
    O = np.zeros((d * d, d * d), dtype=complex)
    for ket_label, bra_label, coeff in terms:
        O += coeff * comp_states[ket_label] @ dagger(comp_states[bra_label])
    return O

# Embedded Pauli operators on qubit 1
X1 = op_from_terms([
    ("10", "00", 1), ("00", "10", 1),
    ("11", "01", 1), ("01", "11", 1)
])

Y1 = op_from_terms([
    ("10", "00", 1j), ("00", "10", -1j),
    ("11", "01", 1j), ("01", "11", -1j)
])

Z1 = op_from_terms([
    ("00", "00", 1), ("01", "01", 1),
    ("10", "10", -1), ("11", "11", -1)
])

# Embedded Pauli operators on qubit 2
X2 = op_from_terms([
    ("01", "00", 1), ("00", "01", 1),
    ("11", "10", 1), ("10", "11", 1)
])

Y2 = op_from_terms([
    ("01", "00", 1j), ("00", "01", -1j),
    ("11", "10", 1j), ("10", "11", -1j)
])

Z2 = op_from_terms([
    ("00", "00", 1), ("10", "10", 1),
    ("01", "01", -1), ("11", "11", -1)
])

measurement_ops = {
    1: {"X": X1, "Y": Y1, "Z": Z1},
    2: {"X": X2, "Y": Y2, "Z": Z2},
}

# ============================================================
# Local tomography states
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
measurement_labels = ["X", "Y", "Z"]

def two_transmon_product_state(q1_state, q2_state):
    return kron(q1_state, q2_state)

def prepare_local_tomography_state(target_qubit, local_label, spectator_label="0"):
    local = local_state_dict[local_label]
    spectator = local_state_dict[spectator_label]

    if target_qubit == 1:
        return two_transmon_product_state(local, spectator)
    elif target_qubit == 2:
        return two_transmon_product_state(spectator, local)
    else:
        raise ValueError("target_qubit must be 1 or 2")

# ============================================================
# Hidden transmon device
# ============================================================

def sample_hidden_device(rng=None):
    if rng is None:
        rng = np.random.default_rng()

    params = {}

    params["Delta1"] = two_pi(rng.normal(0.000, 0.004))
    params["Delta2"] = two_pi(rng.normal(0.150, 0.006))

    params["alpha1"] = two_pi(rng.normal(0.220, 0.008))
    params["alpha2"] = two_pi(rng.normal(0.230, 0.008))

    params["g"] = two_pi(rng.normal(0.0030, 0.0005))
    params["zeta"] = two_pi(rng.normal(0.0008, 0.0002))

    params["amp_scale_1"] = rng.normal(1.00, 0.025)
    params["amp_scale_2"] = rng.normal(1.00, 0.025)

    params["phase_error_1"] = rng.normal(0.0, 0.025)
    params["phase_error_2"] = rng.normal(0.0, 0.025)

    params["T1_1"] = rng.uniform(25000.0, 60000.0)
    params["T1_2"] = rng.uniform(25000.0, 60000.0)

    params["Tphi_1"] = rng.uniform(20000.0, 50000.0)
    params["Tphi_2"] = rng.uniform(20000.0, 50000.0)

    params["r01_1"] = rng.uniform(0.005, 0.040)
    params["r10_1"] = rng.uniform(0.005, 0.050)

    params["r01_2"] = rng.uniform(0.005, 0.040)
    params["r10_2"] = rng.uniform(0.005, 0.050)

    return params

def build_H0(params):
    Delta1 = params["Delta1"]
    Delta2 = params["Delta2"]
    alpha1 = params["alpha1"]
    alpha2 = params["alpha2"]
    g = params["g"]
    zeta = params["zeta"]

    H1 = Delta1 * n1 - 0.5 * alpha1 * (n1 @ (n1 - I9))
    H2 = Delta2 * n2 - 0.5 * alpha2 * (n2 @ (n2 - I9))

    H_coupling = g * (adag1 @ a2 + a1 @ adag2)
    H_zz = zeta * (n1 @ n2)

    return H1 + H2 + H_coupling + H_zz

def control_hamiltonian(params, qubit=1, Omega=0.0, phase=0.0):
    if qubit == 1:
        aq = a1
        adagq = adag1
        amp_scale = params["amp_scale_1"]
        phase_error = params["phase_error_1"]
    elif qubit == 2:
        aq = a2
        adagq = adag2
        amp_scale = params["amp_scale_2"]
        phase_error = params["phase_error_2"]
    else:
        raise ValueError("qubit must be 1 or 2")

    effective_phase = phase + phase_error
    effective_Omega = amp_scale * Omega

    Hx = 0.5 * effective_Omega * np.cos(effective_phase) * (aq + adagq)
    Hy = 0.5 * effective_Omega * np.sin(effective_phase) * (1j * (adagq - aq))

    return Hx + Hy

# ============================================================
# Open-system Lindblad evolution
# ============================================================

def vec(rho):
    return rho.reshape((-1, 1), order="F")

def unvec(v, dim):
    return v.reshape((dim, dim), order="F")

def clean_density_matrix(rho):
    rho = 0.5 * (rho + dagger(rho))
    tr = np.trace(rho)
    if abs(tr) > 1e-12:
        rho = rho / tr
    return rho

def collapse_operators(params):
    cops = []

    gamma1_1 = 1.0 / params["T1_1"]
    gamma1_2 = 1.0 / params["T1_2"]

    cops.append(np.sqrt(gamma1_1) * a1)
    cops.append(np.sqrt(gamma1_2) * a2)

    gammaphi_1 = 1.0 / params["Tphi_1"]
    gammaphi_2 = 1.0 / params["Tphi_2"]

    cops.append(np.sqrt(gammaphi_1) * n1)
    cops.append(np.sqrt(gammaphi_2) * n2)

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

def evolve_rho_square_pulse(
    rho0,
    params,
    qubit=1,
    theta=np.pi,
    phase=0.0,
    T=40.0,
    include_decoherence=True
):
    H0 = build_H0(params)

    Omega = theta / T
    Hc = control_hamiltonian(params, qubit=qubit, Omega=Omega, phase=phase)
    H = H0 + Hc

    if include_decoherence:
        cops = collapse_operators(params)
        L = liouvillian(H, cops)
        rho_vec_final = expm(L * T) @ vec(rho0)
        rho_final = unvec(rho_vec_final, rho0.shape[0])
        rho_final = clean_density_matrix(rho_final)
    else:
        U = expm(-1j * H * T)
        rho_final = U @ rho0 @ dagger(U)
        rho_final = clean_density_matrix(rho_final)

    return rho_final

# ============================================================
# Measurement/readout
# ============================================================

def ideal_measurement_probabilities(rho, qubit=1, basis_label="Z"):
    O = measurement_ops[qubit][basis_label]
    m = trace_expectation(rho, O)
    m = float(np.clip(m, -1.0, 1.0))

    p_plus = 0.5 * (1.0 + m)
    p_minus = 0.5 * (1.0 - m)

    return p_plus, p_minus, m

def apply_readout_error(p_plus, p_minus, params, qubit=1):
    if qubit == 1:
        r01 = params["r01_1"]
        r10 = params["r10_1"]
    elif qubit == 2:
        r01 = params["r01_2"]
        r10 = params["r10_2"]
    else:
        raise ValueError("qubit must be 1 or 2")

    p_plus_meas = (1.0 - r01) * p_plus + r10 * p_minus
    p_minus_meas = r01 * p_plus + (1.0 - r10) * p_minus

    p_plus_meas = float(np.clip(p_plus_meas, 0.0, 1.0))
    p_minus_meas = float(np.clip(p_minus_meas, 0.0, 1.0))

    s = p_plus_meas + p_minus_meas
    p_plus_meas /= s
    p_minus_meas /= s

    return p_plus_meas, p_minus_meas

def exact_measured_expectation_with_readout(rho, params, qubit=1, basis_label="Z"):
    p_plus, p_minus, _ = ideal_measurement_probabilities(rho, qubit, basis_label)
    p_plus_meas, p_minus_meas = apply_readout_error(p_plus, p_minus, params, qubit)
    return float(p_plus_meas - p_minus_meas)

def sample_binary_measurement(rho, params, qubit=1, basis_label="Z", shots=1024, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    p_plus, p_minus, m_ideal = ideal_measurement_probabilities(rho, qubit, basis_label)
    p_plus_meas, p_minus_meas = apply_readout_error(p_plus, p_minus, params, qubit)

    counts = rng.multinomial(shots, [p_plus_meas, p_minus_meas])
    n_plus, n_minus = counts

    m_hat = (n_plus - n_minus) / shots

    return float(m_hat)

print("Setup complete.")
print("Check Tr(P_comp):", np.trace(P_comp).real)
print("Check X1^2-P_comp:", fro_norm(X1 @ X1 - P_comp))

# %% Cell 202
# ============================================================
# PARTIAL TOMOGRAPHY SETTING
# ============================================================
# This is intentionally limited.
# Full local tomography would use 6 states × 3 bases × 2 qubits = 36 features.
# Here we use 3 states × 2 bases × 2 qubits = 12 features + shot feature = 13 inputs.

partial_state_labels = ["0", "+", "+i"]
partial_measurement_labels = ["X", "Z"]

print("Partial states:", partial_state_labels)
print("Partial measurements:", partial_measurement_labels)
print("Partial tomography dimension:", 2 * len(partial_state_labels) * len(partial_measurement_labels))

# ============================================================
# Partial finite-shot tomography vector
# ============================================================

def finite_shot_output_expectations_for_local_input(
    params,
    target_qubit=1,
    local_label="0",
    spectator_label="0",
    measurement_subset=("X", "Z"),
    shots=1024,
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    psi0 = prepare_local_tomography_state(
        target_qubit=target_qubit,
        local_label=local_label,
        spectator_label=spectator_label
    )
    rho0 = ket_to_rho(psi0)

    rho_out = evolve_rho_square_pulse(
        rho0=rho0,
        params=params,
        qubit=target_qubit,
        theta=pulse_theta,
        phase=pulse_phase,
        T=pulse_T,
        include_decoherence=True
    )

    features = []
    for basis_label in measurement_subset:
        m_hat = sample_binary_measurement(
            rho=rho_out,
            params=params,
            qubit=target_qubit,
            basis_label=basis_label,
            shots=shots,
            rng=rng
        )
        features.append(m_hat)

    return np.array(features, dtype=np.float32)

def generate_partial_tomography_vector(
    params,
    shots=1024,
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    spectator_label="0",
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    features = []

    for target_qubit in [1, 2]:
        for state_label in partial_state_labels:
            vals = finite_shot_output_expectations_for_local_input(
                params=params,
                target_qubit=target_qubit,
                local_label=state_label,
                spectator_label=spectator_label,
                measurement_subset=partial_measurement_labels,
                shots=shots,
                pulse_theta=pulse_theta,
                pulse_phase=pulse_phase,
                pulse_T=pulse_T,
                rng=rng
            )
            features.extend(list(vals))

    return np.array(features, dtype=np.float32)

# ============================================================
# Full reference local affine channels
# ============================================================

def exact_output_bloch_for_local_input(
    params,
    target_qubit=1,
    local_label="0",
    spectator_label="0",
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    include_readout=True
):
    psi0 = prepare_local_tomography_state(
        target_qubit=target_qubit,
        local_label=local_label,
        spectator_label=spectator_label
    )
    rho0 = ket_to_rho(psi0)

    rho_out = evolve_rho_square_pulse(
        rho0=rho0,
        params=params,
        qubit=target_qubit,
        theta=pulse_theta,
        phase=pulse_phase,
        T=pulse_T,
        include_decoherence=True
    )

    r_out = []
    for basis_label in ["X", "Y", "Z"]:
        if include_readout:
            m = exact_measured_expectation_with_readout(
                rho_out,
                params,
                qubit=target_qubit,
                basis_label=basis_label
            )
        else:
            _, _, m = ideal_measurement_probabilities(
                rho_out,
                qubit=target_qubit,
                basis_label=basis_label
            )
        r_out.append(m)

    return np.array(r_out, dtype=np.float64)

def fit_affine_bloch_map(r_in_list, r_out_list):
    R_in = np.asarray(r_in_list, dtype=np.float64)
    R_out = np.asarray(r_out_list, dtype=np.float64)

    X_design = np.hstack([R_in, np.ones((R_in.shape[0], 1))])
    W, *_ = np.linalg.lstsq(X_design, R_out, rcond=None)

    A = W[:3, :].T
    b = W[3, :]

    return A, b

def reference_local_affine_channel(
    params,
    target_qubit=1,
    spectator_label="0",
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    include_readout=True
):
    r_in_list = []
    r_out_list = []

    for local_label in tomography_input_labels:
        r_in = expected_bloch[local_label]
        r_out = exact_output_bloch_for_local_input(
            params=params,
            target_qubit=target_qubit,
            local_label=local_label,
            spectator_label=spectator_label,
            pulse_theta=pulse_theta,
            pulse_phase=pulse_phase,
            pulse_T=pulse_T,
            include_readout=include_readout
        )

        r_in_list.append(r_in)
        r_out_list.append(r_out)

    A, b = fit_affine_bloch_map(r_in_list, r_out_list)
    return A, b

def flatten_two_local_channels(A1, b1, A2, b2):
    return np.concatenate([
        A1.reshape(-1),
        b1.reshape(-1),
        A2.reshape(-1),
        b2.reshape(-1)
    ]).astype(np.float32)

def unflatten_two_local_channels(y):
    y = np.asarray(y)

    A1 = y[0:9].reshape(3, 3)
    b1 = y[9:12]

    A2 = y[12:21].reshape(3, 3)
    b2 = y[21:24]

    return A1, b1, A2, b2

def reference_two_local_channels_label(
    params,
    spectator_label="0",
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    include_readout=True
):
    A1, b1 = reference_local_affine_channel(
        params=params,
        target_qubit=1,
        spectator_label=spectator_label,
        pulse_theta=pulse_theta,
        pulse_phase=pulse_phase,
        pulse_T=pulse_T,
        include_readout=include_readout
    )

    A2, b2 = reference_local_affine_channel(
        params=params,
        target_qubit=2,
        spectator_label=spectator_label,
        pulse_theta=pulse_theta,
        pulse_phase=pulse_phase,
        pulse_T=pulse_T,
        include_readout=include_readout
    )

    return flatten_two_local_channels(A1, b1, A2, b2)

# ============================================================
# One sanity check
# ============================================================

rng = np.random.default_rng(123)
p = sample_hidden_device(rng)

x_partial = generate_partial_tomography_vector(p, shots=1024, rng=rng)
y_ref = reference_two_local_channels_label(p)

print("x_partial shape:", x_partial.shape)
print("y_ref shape:", y_ref.shape)
print("x_partial min/max:", x_partial.min(), x_partial.max())

A1, b1, A2, b2 = unflatten_two_local_channels(y_ref)
print("A1:")
print(A1)
print("b1:", b1)

# %% Cell 203
# ============================================================
# PARTIAL TOMOGRAPHY SETTING
# ============================================================
# This is intentionally limited.
# Full local tomography would use 6 states × 3 bases × 2 qubits = 36 features.
# Here we use 3 states × 2 bases × 2 qubits = 12 features + shot feature = 13 inputs.

partial_state_labels = ["0", "+", "+i"]
partial_measurement_labels = ["X", "Z"]

print("Partial states:", partial_state_labels)
print("Partial measurements:", partial_measurement_labels)
print("Partial tomography dimension:", 2 * len(partial_state_labels) * len(partial_measurement_labels))

# ============================================================
# Partial finite-shot tomography vector
# ============================================================

def finite_shot_output_expectations_for_local_input(
    params,
    target_qubit=1,
    local_label="0",
    spectator_label="0",
    measurement_subset=("X", "Z"),
    shots=1024,
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    psi0 = prepare_local_tomography_state(
        target_qubit=target_qubit,
        local_label=local_label,
        spectator_label=spectator_label
    )
    rho0 = ket_to_rho(psi0)

    rho_out = evolve_rho_square_pulse(
        rho0=rho0,
        params=params,
        qubit=target_qubit,
        theta=pulse_theta,
        phase=pulse_phase,
        T=pulse_T,
        include_decoherence=True
    )

    features = []
    for basis_label in measurement_subset:
        m_hat = sample_binary_measurement(
            rho=rho_out,
            params=params,
            qubit=target_qubit,
            basis_label=basis_label,
            shots=shots,
            rng=rng
        )
        features.append(m_hat)

    return np.array(features, dtype=np.float32)

def generate_partial_tomography_vector(
    params,
    shots=1024,
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    spectator_label="0",
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    features = []

    for target_qubit in [1, 2]:
        for state_label in partial_state_labels:
            vals = finite_shot_output_expectations_for_local_input(
                params=params,
                target_qubit=target_qubit,
                local_label=state_label,
                spectator_label=spectator_label,
                measurement_subset=partial_measurement_labels,
                shots=shots,
                pulse_theta=pulse_theta,
                pulse_phase=pulse_phase,
                pulse_T=pulse_T,
                rng=rng
            )
            features.extend(list(vals))

    return np.array(features, dtype=np.float32)

# ============================================================
# Full reference local affine channels
# ============================================================

def exact_output_bloch_for_local_input(
    params,
    target_qubit=1,
    local_label="0",
    spectator_label="0",
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    include_readout=True
):
    psi0 = prepare_local_tomography_state(
        target_qubit=target_qubit,
        local_label=local_label,
        spectator_label=spectator_label
    )
    rho0 = ket_to_rho(psi0)

    rho_out = evolve_rho_square_pulse(
        rho0=rho0,
        params=params,
        qubit=target_qubit,
        theta=pulse_theta,
        phase=pulse_phase,
        T=pulse_T,
        include_decoherence=True
    )

    r_out = []
    for basis_label in ["X", "Y", "Z"]:
        if include_readout:
            m = exact_measured_expectation_with_readout(
                rho_out,
                params,
                qubit=target_qubit,
                basis_label=basis_label
            )
        else:
            _, _, m = ideal_measurement_probabilities(
                rho_out,
                qubit=target_qubit,
                basis_label=basis_label
            )
        r_out.append(m)

    return np.array(r_out, dtype=np.float64)

def fit_affine_bloch_map(r_in_list, r_out_list):
    R_in = np.asarray(r_in_list, dtype=np.float64)
    R_out = np.asarray(r_out_list, dtype=np.float64)

    X_design = np.hstack([R_in, np.ones((R_in.shape[0], 1))])
    W, *_ = np.linalg.lstsq(X_design, R_out, rcond=None)

    A = W[:3, :].T
    b = W[3, :]

    return A, b

def reference_local_affine_channel(
    params,
    target_qubit=1,
    spectator_label="0",
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    include_readout=True
):
    r_in_list = []
    r_out_list = []

    for local_label in tomography_input_labels:
        r_in = expected_bloch[local_label]
        r_out = exact_output_bloch_for_local_input(
            params=params,
            target_qubit=target_qubit,
            local_label=local_label,
            spectator_label=spectator_label,
            pulse_theta=pulse_theta,
            pulse_phase=pulse_phase,
            pulse_T=pulse_T,
            include_readout=include_readout
        )

        r_in_list.append(r_in)
        r_out_list.append(r_out)

    A, b = fit_affine_bloch_map(r_in_list, r_out_list)
    return A, b

def flatten_two_local_channels(A1, b1, A2, b2):
    return np.concatenate([
        A1.reshape(-1),
        b1.reshape(-1),
        A2.reshape(-1),
        b2.reshape(-1)
    ]).astype(np.float32)

def unflatten_two_local_channels(y):
    y = np.asarray(y)

    A1 = y[0:9].reshape(3, 3)
    b1 = y[9:12]

    A2 = y[12:21].reshape(3, 3)
    b2 = y[21:24]

    return A1, b1, A2, b2

def reference_two_local_channels_label(
    params,
    spectator_label="0",
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0,
    include_readout=True
):
    A1, b1 = reference_local_affine_channel(
        params=params,
        target_qubit=1,
        spectator_label=spectator_label,
        pulse_theta=pulse_theta,
        pulse_phase=pulse_phase,
        pulse_T=pulse_T,
        include_readout=include_readout
    )

    A2, b2 = reference_local_affine_channel(
        params=params,
        target_qubit=2,
        spectator_label=spectator_label,
        pulse_theta=pulse_theta,
        pulse_phase=pulse_phase,
        pulse_T=pulse_T,
        include_readout=include_readout
    )

    return flatten_two_local_channels(A1, b1, A2, b2)

# ============================================================
# One sanity check
# ============================================================

rng = np.random.default_rng(123)
p = sample_hidden_device(rng)

x_partial = generate_partial_tomography_vector(p, shots=1024, rng=rng)
y_ref = reference_two_local_channels_label(p)

print("x_partial shape:", x_partial.shape)
print("y_ref shape:", y_ref.shape)
print("x_partial min/max:", x_partial.min(), x_partial.max())

A1, b1, A2, b2 = unflatten_two_local_channels(y_ref)
print("A1:")
print(A1)
print("b1:", b1)

# %% Cell 204
# ============================================================
# Dataset generation
# ============================================================

shot_choices = np.array([128, 512, 1024, 4096], dtype=int)

def append_shot_feature(x_tomo, shots):
    shot_feature = np.array([np.log10(shots)], dtype=np.float32)
    return np.concatenate([x_tomo.astype(np.float32), shot_feature])

def generate_one_partial_example(
    rng,
    shot_choices=shot_choices,
    pulse_theta=np.pi / 2,
    pulse_phase=0.0,
    pulse_T=40.0
):
    shots = int(rng.choice(shot_choices))
    params = sample_hidden_device(rng)

    x_tomo = generate_partial_tomography_vector(
        params=params,
        shots=shots,
        pulse_theta=pulse_theta,
        pulse_phase=pulse_phase,
        pulse_T=pulse_T,
        rng=rng
    )

    x = append_shot_feature(x_tomo, shots)

    y = reference_two_local_channels_label(
        params=params,
        pulse_theta=pulse_theta,
        pulse_phase=pulse_phase,
        pulse_T=pulse_T,
        include_readout=True
    )

    return x.astype(np.float32), y.astype(np.float32), params, shots

def build_partial_dataset(
    n_examples,
    seed=0,
    print_every=25
):
    rng = np.random.default_rng(seed)

    X = []
    Y = []
    params_list = []
    shots_list = []

    t0 = time.time()

    for i in range(n_examples):
        x, y, params, shots = generate_one_partial_example(rng)

        X.append(x)
        Y.append(y)
        params_list.append(params)
        shots_list.append(shots)

        if (i + 1) % print_every == 0 or (i + 1) == n_examples:
            print(f"Generated {i+1:4d}/{n_examples} | elapsed {time.time() - t0:.1f} s")

    return (
        np.stack(X).astype(np.float32),
        np.stack(Y).astype(np.float32),
        params_list,
        np.array(shots_list, dtype=int)
    )

# ============================================================
# Choose dataset size
# ============================================================

# Increase these later if runtime is acceptable.
N_train = 160
N_val = 50
N_test = 50

X_train, Y_train, params_train, shots_train = build_partial_dataset(N_train, seed=1000, print_every=20)
X_val, Y_val, params_val, shots_val = build_partial_dataset(N_val, seed=2000, print_every=10)
X_test, Y_test, params_test, shots_test = build_partial_dataset(N_test, seed=3000, print_every=10)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_val shape:", X_val.shape)
print("Y_val shape:", Y_val.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

print("X range:", X_train[:, :-1].min(), X_train[:, :-1].max())
print("Y range:", Y_train.min(), Y_train.max())

print("Shot counts:")
for s, c in zip(*np.unique(shots_train, return_counts=True)):
    print(s, c)

plt.figure(figsize=(8, 4))
plt.hist(X_train[:, :-1].flatten(), bins=40)
plt.xlabel("Partial tomography feature")
plt.ylabel("Count")
plt.title("Partial finite-shot tomography features")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 4))
plt.hist(Y_train.flatten(), bins=40)
plt.xlabel("Reference full-channel parameter")
plt.ylabel("Count")
plt.title("Reference effective-channel labels")
plt.grid(True)
plt.show()

# %% Cell 205
# ============================================================
# Standardization
# ============================================================

def compute_standardizer(X, eps=1e-8):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return mean.astype(np.float32), std.astype(np.float32)

def apply_standardizer(X, mean, std):
    return ((X - mean) / std).astype(np.float32)

def invert_standardizer(X_std, mean, std):
    return (X_std * std + mean).astype(np.float32)

X_mean, X_std = compute_standardizer(X_train)
Y_mean, Y_std = compute_standardizer(Y_train)

X_train_s = apply_standardizer(X_train, X_mean, X_std)
X_val_s = apply_standardizer(X_val, X_mean, X_std)
X_test_s = apply_standardizer(X_test, X_mean, X_std)

Y_train_s = apply_standardizer(Y_train, Y_mean, Y_std)
Y_val_s = apply_standardizer(Y_val, Y_mean, Y_std)
Y_test_s = apply_standardizer(Y_test, Y_mean, Y_std)

# ============================================================
# PyTorch datasets
# ============================================================

X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train_s, dtype=torch.float32)

X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
Y_val_t = torch.tensor(Y_val_s, dtype=torch.float32)

X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test_s, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=128, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_t, Y_test_t), batch_size=128, shuffle=False)

print("Input dimension:", X_train.shape[1])
print("Output dimension:", Y_train.shape[1])

# ============================================================
# ChannelNet
# ============================================================

class ChannelNet(nn.Module):
    def __init__(self, input_dim, output_dim=24):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

model = ChannelNet(input_dim=X_train.shape[1], output_dim=24).to(device)

print(model)
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# ============================================================
# Physics-informed physicality penalty
# ============================================================

Y_mean_t = torch.tensor(Y_mean, dtype=torch.float32).to(device)
Y_std_t = torch.tensor(Y_std, dtype=torch.float32).to(device)

def unstandardize_y_torch(y_s):
    return y_s * Y_std_t + Y_mean_t

def split_channels_torch(y_phys):
    A1 = y_phys[:, 0:9].reshape(-1, 3, 3)
    b1 = y_phys[:, 9:12]

    A2 = y_phys[:, 12:21].reshape(-1, 3, 3)
    b2 = y_phys[:, 21:24]

    return A1, b1, A2, b2

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

bloch_test = torch.tensor(bloch_test_np, dtype=torch.float32).to(device)

def physicality_loss_from_standardized_output(y_pred_s):
    y_phys = unstandardize_y_torch(y_pred_s)
    A1, b1, A2, b2 = split_channels_torch(y_phys)

    r = bloch_test.unsqueeze(0)

    out1 = torch.matmul(r, A1.transpose(1, 2)) + b1.unsqueeze(1)
    out2 = torch.matmul(r, A2.transpose(1, 2)) + b2.unsqueeze(1)

    norm1 = torch.linalg.norm(out1, dim=-1)
    norm2 = torch.linalg.norm(out2, dim=-1)

    penalty1 = torch.relu(norm1 - 1.0) ** 2
    penalty2 = torch.relu(norm2 - 1.0) ** 2

    return penalty1.mean() + penalty2.mean()

# ============================================================
# Training
# ============================================================

mse_loss = nn.MSELoss()

def evaluate_model(model, loader, lambda_phys=0.0):
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_phys = 0.0
    n_total = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)

            mse = mse_loss(pred, yb)
            phys = physicality_loss_from_standardized_output(pred)
            loss = mse + lambda_phys * phys

            bs = xb.shape[0]
            total_loss += loss.item() * bs
            total_mse += mse.item() * bs
            total_phys += phys.item() * bs
            n_total += bs

    return {
        "loss": total_loss / n_total,
        "mse": total_mse / n_total,
        "phys": total_phys / n_total,
    }

def train_channelnet(
    model,
    train_loader,
    val_loader,
    n_epochs=500,
    lr=8e-4,
    weight_decay=1e-5,
    lambda_phys=0.03
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_mse": [],
        "val_mse": [],
        "train_phys": [],
        "val_phys": [],
    }

    best_val = np.inf
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()

        train_mse_total = 0.0
        train_phys_total = 0.0
        n_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)

            mse = mse_loss(pred, yb)
            phys = physicality_loss_from_standardized_output(pred)
            loss = mse + lambda_phys * phys

            loss.backward()
            optimizer.step()

            bs = xb.shape[0]
            train_mse_total += mse.item() * bs
            train_phys_total += phys.item() * bs
            n_total += bs

        train_mse = train_mse_total / n_total
        train_phys = train_phys_total / n_total

        val_metrics = evaluate_model(model, val_loader, lambda_phys=lambda_phys)

        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_metrics["mse"])
        history["train_phys"].append(train_phys)
        history["val_phys"].append(val_metrics["phys"])

        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"train MSE {train_mse:.4e} | "
                f"val MSE {val_metrics['mse']:.4e} | "
                f"val phys {val_metrics['phys']:.4e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history

history = train_channelnet(
    model,
    train_loader,
    val_loader,
    n_epochs=500,
    lr=8e-4,
    weight_decay=1e-5,
    lambda_phys=0.03
)

plt.figure(figsize=(8, 5))
plt.plot(history["train_mse"], label="train MSE")
plt.plot(history["val_mse"], label="validation MSE")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE on standardized channel labels")
plt.title("Partial-tomography ChannelNet training curve")
plt.legend()
plt.grid(True)
plt.show()

print("Train metrics:", evaluate_model(model, train_loader, lambda_phys=0.03))
print("Val metrics:", evaluate_model(model, val_loader, lambda_phys=0.03))
print("Test metrics:", evaluate_model(model, test_loader, lambda_phys=0.03))

# %% Cell 206
# ============================================================
# Standardization
# ============================================================

def compute_standardizer(X, eps=1e-8):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return mean.astype(np.float32), std.astype(np.float32)

def apply_standardizer(X, mean, std):
    return ((X - mean) / std).astype(np.float32)

def invert_standardizer(X_std, mean, std):
    return (X_std * std + mean).astype(np.float32)

X_mean, X_std = compute_standardizer(X_train)
Y_mean, Y_std = compute_standardizer(Y_train)

X_train_s = apply_standardizer(X_train, X_mean, X_std)
X_val_s = apply_standardizer(X_val, X_mean, X_std)
X_test_s = apply_standardizer(X_test, X_mean, X_std)

Y_train_s = apply_standardizer(Y_train, Y_mean, Y_std)
Y_val_s = apply_standardizer(Y_val, Y_mean, Y_std)
Y_test_s = apply_standardizer(Y_test, Y_mean, Y_std)

# ============================================================
# PyTorch datasets
# ============================================================

X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train_s, dtype=torch.float32)

X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
Y_val_t = torch.tensor(Y_val_s, dtype=torch.float32)

X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test_s, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=128, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_t, Y_test_t), batch_size=128, shuffle=False)

print("Input dimension:", X_train.shape[1])
print("Output dimension:", Y_train.shape[1])

# ============================================================
# ChannelNet
# ============================================================

class ChannelNet(nn.Module):
    def __init__(self, input_dim, output_dim=24):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

model = ChannelNet(input_dim=X_train.shape[1], output_dim=24).to(device)

print(model)
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# ============================================================
# Physics-informed physicality penalty
# ============================================================

Y_mean_t = torch.tensor(Y_mean, dtype=torch.float32).to(device)
Y_std_t = torch.tensor(Y_std, dtype=torch.float32).to(device)

def unstandardize_y_torch(y_s):
    return y_s * Y_std_t + Y_mean_t

def split_channels_torch(y_phys):
    A1 = y_phys[:, 0:9].reshape(-1, 3, 3)
    b1 = y_phys[:, 9:12]

    A2 = y_phys[:, 12:21].reshape(-1, 3, 3)
    b2 = y_phys[:, 21:24]

    return A1, b1, A2, b2

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

bloch_test = torch.tensor(bloch_test_np, dtype=torch.float32).to(device)

def physicality_loss_from_standardized_output(y_pred_s):
    y_phys = unstandardize_y_torch(y_pred_s)
    A1, b1, A2, b2 = split_channels_torch(y_phys)

    r = bloch_test.unsqueeze(0)

    out1 = torch.matmul(r, A1.transpose(1, 2)) + b1.unsqueeze(1)
    out2 = torch.matmul(r, A2.transpose(1, 2)) + b2.unsqueeze(1)

    norm1 = torch.linalg.norm(out1, dim=-1)
    norm2 = torch.linalg.norm(out2, dim=-1)

    penalty1 = torch.relu(norm1 - 1.0) ** 2
    penalty2 = torch.relu(norm2 - 1.0) ** 2

    return penalty1.mean() + penalty2.mean()

# ============================================================
# Training
# ============================================================

mse_loss = nn.MSELoss()

def evaluate_model(model, loader, lambda_phys=0.0):
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_phys = 0.0
    n_total = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)

            mse = mse_loss(pred, yb)
            phys = physicality_loss_from_standardized_output(pred)
            loss = mse + lambda_phys * phys

            bs = xb.shape[0]
            total_loss += loss.item() * bs
            total_mse += mse.item() * bs
            total_phys += phys.item() * bs
            n_total += bs

    return {
        "loss": total_loss / n_total,
        "mse": total_mse / n_total,
        "phys": total_phys / n_total,
    }

def train_channelnet(
    model,
    train_loader,
    val_loader,
    n_epochs=500,
    lr=8e-4,
    weight_decay=1e-5,
    lambda_phys=0.03
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_mse": [],
        "val_mse": [],
        "train_phys": [],
        "val_phys": [],
    }

    best_val = np.inf
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()

        train_mse_total = 0.0
        train_phys_total = 0.0
        n_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)

            mse = mse_loss(pred, yb)
            phys = physicality_loss_from_standardized_output(pred)
            loss = mse + lambda_phys * phys

            loss.backward()
            optimizer.step()

            bs = xb.shape[0]
            train_mse_total += mse.item() * bs
            train_phys_total += phys.item() * bs
            n_total += bs

        train_mse = train_mse_total / n_total
        train_phys = train_phys_total / n_total

        val_metrics = evaluate_model(model, val_loader, lambda_phys=lambda_phys)

        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_metrics["mse"])
        history["train_phys"].append(train_phys)
        history["val_phys"].append(val_metrics["phys"])

        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"train MSE {train_mse:.4e} | "
                f"val MSE {val_metrics['mse']:.4e} | "
                f"val phys {val_metrics['phys']:.4e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history

history = train_channelnet(
    model,
    train_loader,
    val_loader,
    n_epochs=500,
    lr=8e-4,
    weight_decay=1e-5,
    lambda_phys=0.03
)

plt.figure(figsize=(8, 5))
plt.plot(history["train_mse"], label="train MSE")
plt.plot(history["val_mse"], label="validation MSE")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE on standardized channel labels")
plt.title("Partial-tomography ChannelNet training curve")
plt.legend()
plt.grid(True)
plt.show()

print("Train metrics:", evaluate_model(model, train_loader, lambda_phys=0.03))
print("Val metrics:", evaluate_model(model, val_loader, lambda_phys=0.03))
print("Test metrics:", evaluate_model(model, test_loader, lambda_phys=0.03))

# %% Cell 207
# ============================================================
# Prediction utility
# ============================================================

def predict_physical_y(model, X_s_np):
    model.eval()

    X_t = torch.tensor(X_s_np, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_s = model(X_t).cpu().numpy()

    return invert_standardizer(pred_s, Y_mean, Y_std).astype(np.float32)

Y_train_pred = predict_physical_y(model, X_train_s)
Y_val_pred = predict_physical_y(model, X_val_s)
Y_test_pred = predict_physical_y(model, X_test_s)

# ============================================================
# Direct partial tomography baseline
# ============================================================

Y_train_mean_channel = Y_train.mean(axis=0)

def direct_partial_channel_from_features(
    x_partial_no_shot,
    mean_channel,
    ridge=1e-6
):
    """
    Direct baseline with incomplete tomography.

    Starts from mean training channel.
    Fits only measured rows/components using partial data.
    Unmeasured channel components remain at mean-channel prior.
    """
    x_partial_no_shot = np.asarray(x_partial_no_shot, dtype=np.float64)

    A1_mean, b1_mean, A2_mean, b2_mean = unflatten_two_local_channels(mean_channel)

    channels = [
        [A1_mean.copy(), b1_mean.copy()],
        [A2_mean.copy(), b2_mean.copy()]
    ]

    offset = 0

    for q_local_index, target_qubit in enumerate([1, 2]):
        A_init, b_init = channels[q_local_index]

        rows_by_component = {0: [], 1: [], 2: []}
        vals_by_component = {0: [], 1: [], 2: []}

        for state_label in partial_state_labels:
            r_in = expected_bloch[state_label]
            design_row = np.concatenate([r_in, [1.0]])

            for meas_label in partial_measurement_labels:
                comp = measurement_labels.index(meas_label)

                m_val = x_partial_no_shot[offset]
                offset += 1

                rows_by_component[comp].append(design_row)
                vals_by_component[comp].append(m_val)

        for comp in [0, 1, 2]:
            if len(rows_by_component[comp]) == 0:
                continue

            Xd = np.stack(rows_by_component[comp])
            yd = np.array(vals_by_component[comp])

            XtX = Xd.T @ Xd
            Xty = Xd.T @ yd

            w = np.linalg.solve(XtX + ridge * np.eye(4), Xty)

            A_init[comp, :] = w[:3]
            b_init[comp] = w[3]

        channels[q_local_index] = [A_init, b_init]

    A1, b1 = channels[0]
    A2, b2 = channels[1]

    return flatten_two_local_channels(A1, b1, A2, b2)

def direct_partial_channels_for_dataset(X_partial_aug, mean_channel):
    Y_direct = []

    for i in range(len(X_partial_aug)):
        x_no_shot = X_partial_aug[i, :-1]
        y_direct = direct_partial_channel_from_features(
            x_partial_no_shot=x_no_shot,
            mean_channel=mean_channel
        )
        Y_direct.append(y_direct)

    return np.stack(Y_direct).astype(np.float32)

Y_train_direct = direct_partial_channels_for_dataset(X_train, Y_train_mean_channel)
Y_val_direct = direct_partial_channels_for_dataset(X_val, Y_train_mean_channel)
Y_test_direct = direct_partial_channels_for_dataset(X_test, Y_train_mean_channel)

Y_train_baseline = np.repeat(Y_train_mean_channel[None, :], len(Y_train), axis=0)
Y_val_baseline = np.repeat(Y_train_mean_channel[None, :], len(Y_val), axis=0)
Y_test_baseline = np.repeat(Y_train_mean_channel[None, :], len(Y_test), axis=0)

# ============================================================
# Metrics
# ============================================================

def mse_np(a, b):
    return float(np.mean((a - b) ** 2))

def per_example_l2_error(Y_est, Y_true):
    return np.linalg.norm(Y_est - Y_true, axis=1)

baseline_test_mse = mse_np(Y_test_baseline, Y_test)
direct_test_mse = mse_np(Y_test_direct, Y_test)
channelnet_test_mse = mse_np(Y_test_pred, Y_test)

print("Partial-tomography physical-space channel-label MSE")
print("-" * 85)
print(f"{'Split':<12s} {'Mean baseline':>18s} {'Direct partial':>20s} {'ChannelNet partial':>22s}")
print("-" * 85)
print(f"{'Train':<12s} {mse_np(Y_train_baseline, Y_train):18.6e} {mse_np(Y_train_direct, Y_train):20.6e} {mse_np(Y_train_pred, Y_train):22.6e}")
print(f"{'Val':<12s} {mse_np(Y_val_baseline, Y_val):18.6e} {mse_np(Y_val_direct, Y_val):20.6e} {mse_np(Y_val_pred, Y_val):22.6e}")
print(f"{'Test':<12s} {baseline_test_mse:18.6e} {direct_test_mse:20.6e} {channelnet_test_mse:22.6e}")

err_mean_test = per_example_l2_error(Y_test_baseline, Y_test)
err_direct_test = per_example_l2_error(Y_test_direct, Y_test)
err_channelnet_test = per_example_l2_error(Y_test_pred, Y_test)

print("\nTest L2 errors")
print("Mean baseline:   mean", err_mean_test.mean(), "median", np.median(err_mean_test))
print("Direct partial:  mean", err_direct_test.mean(), "median", np.median(err_direct_test))
print("ChannelNet:      mean", err_channelnet_test.mean(), "median", np.median(err_channelnet_test))

# ============================================================
# Error by shots
# ============================================================

def grouped_errors_by_shots(Y_est, Y_true, shots_array):
    out = {}

    for s in sorted(np.unique(shots_array)):
        mask = shots_array == s
        errs = per_example_l2_error(Y_est[mask], Y_true[mask])
        out[int(s)] = {
            "n": int(mask.sum()),
            "mean": float(errs.mean()),
            "median": float(np.median(errs)),
            "std": float(errs.std())
        }

    return out

direct_by_shots = grouped_errors_by_shots(Y_test_direct, Y_test, shots_test)
channelnet_by_shots = grouped_errors_by_shots(Y_test_pred, Y_test, shots_test)

print("\nDirect partial by shots:")
for s, vals in direct_by_shots.items():
    print(s, vals)

print("\nChannelNet by shots:")
for s, vals in channelnet_by_shots.items():
    print(s, vals)

shots_sorted = sorted(np.unique(shots_test))

plt.figure(figsize=(7, 4))
plt.plot(
    shots_sorted,
    [direct_by_shots[s]["mean"] for s in shots_sorted],
    marker="o",
    label="Direct partial mean"
)
plt.plot(
    shots_sorted,
    [channelnet_by_shots[s]["mean"] for s in shots_sorted],
    marker="s",
    label="ChannelNet mean"
)
plt.xscale("log")
plt.xlabel("Number of tomography shots")
plt.ylabel("Channel-label L2 error")
plt.title("Learning full channels from partial tomography")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# Bar plot
# ============================================================

methods = ["Mean\nbaseline", "Direct\npartial", "ChannelNet\npartial"]
mean_errors = [err_mean_test.mean(), err_direct_test.mean(), err_channelnet_test.mean()]
median_errors = [np.median(err_mean_test), np.median(err_direct_test), np.median(err_channelnet_test)]

xpos = np.arange(len(methods))

plt.figure(figsize=(7, 4))
plt.bar(xpos - 0.18, mean_errors, width=0.36, label="Mean L2")
plt.bar(xpos + 0.18, median_errors, width=0.36, label="Median L2")
plt.xticks(xpos, methods)
plt.ylabel("Channel-label L2 error")
plt.title("Partial tomography: full-channel inference")
plt.legend()
plt.grid(axis="y")
plt.show()

# ============================================================
# Predicted vs true
# ============================================================

plt.figure(figsize=(5, 5))
plt.scatter(Y_test.flatten(), Y_test_direct.flatten(), alpha=0.35, label="Direct partial")
plt.scatter(Y_test.flatten(), Y_test_pred.flatten(), alpha=0.35, label="ChannelNet partial")

min_val = min(Y_test.min(), Y_test_direct.min(), Y_test_pred.min())
max_val = max(Y_test.max(), Y_test_direct.max(), Y_test_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.xlabel("True full-channel parameter")
plt.ylabel("Estimated channel parameter")
plt.title("Full-channel inference from partial tomography")
plt.legend()
plt.grid(True)
plt.show()

# %% Cell 208
# ============================================================
# Ideal two-qubit QAOA
# ============================================================

I2 = np.eye(2, dtype=complex)

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

I4 = np.eye(4, dtype=complex)

X1_4 = kron(X, I2)
X2_4 = kron(I2, X)

Z1_4 = kron(Z, I2)
Z2_4 = kron(I2, Z)

ZZ_4 = Z1_4 @ Z2_4

H_cost = 0.5 * (I4 - ZZ_4)
H_mixer = X1_4 + X2_4

ket0_2 = basis(2, 0)
ket1_2 = basis(2, 1)

ket00_4 = kron(ket0_2, ket0_2)
ket01_4 = kron(ket0_2, ket1_2)
ket10_4 = kron(ket1_2, ket0_2)
ket11_4 = kron(ket1_2, ket1_2)

ket_plus_2 = normalize_ket((ket0_2 + ket1_2) / np.sqrt(2))
ketplusplus_4 = kron(ket_plus_2, ket_plus_2)

basis4 = {
    "00": ket00_4,
    "01": ket01_4,
    "10": ket10_4,
    "11": ket11_4,
}

def ideal_qaoa_state(gamma, beta):
    U_cost = expm(-1j * gamma * H_cost)
    U_mixer = expm(-1j * beta * H_mixer)

    psi = U_mixer @ U_cost @ ketplusplus_4
    return normalize_ket(psi)

def ideal_qaoa_cost(gamma, beta):
    psi = ideal_qaoa_state(gamma, beta)
    val = dagger(psi) @ H_cost @ psi
    return float(np.real_if_close(val[0, 0]))

# ============================================================
# Product local channel on two-qubit state
# ============================================================

paulis_1q = [I2, X, Y, Z]
pauli_labels_1q = ["I", "X", "Y", "Z"]

paulis_2q = []
pauli_labels_2q = []

for la, Pa in zip(pauli_labels_1q, paulis_1q):
    for lb, Pb in zip(pauli_labels_1q, paulis_1q):
        paulis_2q.append(kron(Pa, Pb))
        pauli_labels_2q.append(la + lb)

def rho_to_pauli_coeffs_2q(rho):
    coeffs = []
    for P in paulis_2q:
        coeffs.append(float(np.real_if_close(np.trace(rho @ P))))
    return np.array(coeffs, dtype=np.float64)

def pauli_coeffs_to_rho_2q(coeffs):
    rho = np.zeros((4, 4), dtype=complex)
    for c, P in zip(coeffs, paulis_2q):
        rho += c * P
    rho = rho / 4.0
    return clean_density_matrix(rho)

def affine_channel_to_ptm(A, b):
    R = np.zeros((4, 4), dtype=np.float64)
    R[0, 0] = 1.0
    R[1:4, 0] = b
    R[1:4, 1:4] = A
    return R

def apply_product_local_channel_to_rho(rho, A1, b1, A2, b2):
    R1 = affine_channel_to_ptm(A1, b1)
    R2 = affine_channel_to_ptm(A2, b2)

    R12 = np.kron(R1, R2)

    c_in = rho_to_pauli_coeffs_2q(rho)
    c_out = R12 @ c_in

    return pauli_coeffs_to_rho_2q(c_out)

def bitstring_probs_from_rho_2q(rho):
    probs = {}

    for label, ket in basis4.items():
        P = ket @ dagger(ket)
        p = float(np.real_if_close(np.trace(rho @ P)))
        probs[label] = p

    keys = ["00", "01", "10", "11"]
    arr = np.array([probs[k] for k in keys], dtype=np.float64)
    arr = np.clip(arr, 0.0, 1.0)

    if arr.sum() <= 1e-12:
        arr = np.ones(4) / 4
    else:
        arr = arr / arr.sum()

    return {k: float(v) for k, v in zip(keys, arr)}

def maxcut_cost_from_probs(probs):
    return probs["01"] + probs["10"]

def sample_qaoa_cost_from_probs(probs, shots=2048, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    keys = ["00", "01", "10", "11"]
    p = np.array([probs[k] for k in keys], dtype=np.float64)
    p = p / p.sum()

    counts = rng.multinomial(shots, p)
    count_dict = {k: int(c) for k, c in zip(keys, counts)}

    cost_hat = (count_dict["01"] + count_dict["10"]) / shots

    return cost_hat, count_dict

def effective_noisy_qaoa_cost(gamma, beta, A1, b1, A2, b2, shots=None, rng=None):
    psi = ideal_qaoa_state(gamma, beta)
    rho = ket_to_rho(psi)

    rho_noisy = apply_product_local_channel_to_rho(
        rho,
        A1, b1,
        A2, b2
    )

    probs_noisy = bitstring_probs_from_rho_2q(rho_noisy)

    if shots is None:
        return maxcut_cost_from_probs(probs_noisy)
    else:
        cost_hat, _ = sample_qaoa_cost_from_probs(probs_noisy, shots=shots, rng=rng)
        return cost_hat

def qaoa_landscape_from_channel(gamma_grid, beta_grid, A1, b1, A2, b2, shots=None, seed=0):
    rng = np.random.default_rng(seed)
    C = np.zeros((len(gamma_grid), len(beta_grid)), dtype=np.float64)

    for i, gamma in enumerate(gamma_grid):
        for j, beta in enumerate(beta_grid):
            C[i, j] = effective_noisy_qaoa_cost(
                gamma, beta, A1, b1, A2, b2, shots=shots, rng=rng
            )

    return C

def mitigate_qaoa_landscape_by_bias_correction(C_measured_noisy, C_ideal, C_model_noisy, clip=True):
    bias_hat = C_model_noisy - C_ideal
    C_mitigated = C_measured_noisy - bias_hat

    if clip:
        C_mitigated = np.clip(C_mitigated, 0.0, 1.0)

    return C_mitigated, bias_hat

def landscape_error_metrics(C_est, C_ideal):
    abs_err = np.abs(C_est - C_ideal)

    return {
        "MAE": float(abs_err.mean()),
        "MedianAE": float(np.median(abs_err)),
        "MaxAE": float(abs_err.max()),
        "RMSE": float(np.sqrt(np.mean(abs_err**2))),
    }

def improvement_ratio(noisy_error, mitigated_error, eps=1e-12):
    return noisy_error / (mitigated_error + eps)

# ============================================================
# Ideal QAOA grid
# ============================================================

n_gamma = 41
n_beta = 41

gamma_grid = np.linspace(0, np.pi, n_gamma)
beta_grid = np.linspace(0, np.pi / 2, n_beta)

C_ideal = np.zeros((n_gamma, n_beta), dtype=np.float64)

for i, gamma in enumerate(gamma_grid):
    for j, beta in enumerate(beta_grid):
        C_ideal[i, j] = ideal_qaoa_cost(gamma, beta)

print("Ideal max cost:", C_ideal.max())

plt.figure(figsize=(6, 4))
plt.imshow(
    C_ideal,
    origin="lower",
    aspect="auto",
    extent=[beta_grid[0], beta_grid[-1], gamma_grid[0], gamma_grid[-1]],
    vmin=0,
    vmax=1
)
plt.colorbar(label="Ideal QAOA cost")
plt.xlabel(r"$\beta$")
plt.ylabel(r"$\gamma$")
plt.title("Ideal QAOA/MaxCut landscape")
plt.show()

# ============================================================
# One-device visual example
# ============================================================

idx_device = 0

A1_true, b1_true, A2_true, b2_true = unflatten_two_local_channels(Y_test[idx_device])
A1_direct, b1_direct, A2_direct, b2_direct = unflatten_two_local_channels(Y_test_direct[idx_device])
A1_pred, b1_pred, A2_pred, b2_pred = unflatten_two_local_channels(Y_test_pred[idx_device])

qaoa_shots = 2048

C_noisy_sampled = qaoa_landscape_from_channel(
    gamma_grid, beta_grid,
    A1_true, b1_true,
    A2_true, b2_true,
    shots=qaoa_shots,
    seed=500
)

C_direct_model = qaoa_landscape_from_channel(
    gamma_grid, beta_grid,
    A1_direct, b1_direct,
    A2_direct, b2_direct,
    shots=None,
    seed=501
)

C_channelnet_model = qaoa_landscape_from_channel(
    gamma_grid, beta_grid,
    A1_pred, b1_pred,
    A2_pred, b2_pred,
    shots=None,
    seed=502
)

C_mitigated_direct, _ = mitigate_qaoa_landscape_by_bias_correction(
    C_noisy_sampled,
    C_ideal,
    C_direct_model,
    clip=True
)

C_mitigated_channelnet, _ = mitigate_qaoa_landscape_by_bias_correction(
    C_noisy_sampled,
    C_ideal,
    C_channelnet_model,
    clip=True
)

metrics_noisy = landscape_error_metrics(C_noisy_sampled, C_ideal)
metrics_direct = landscape_error_metrics(C_mitigated_direct, C_ideal)
metrics_channelnet = landscape_error_metrics(C_mitigated_channelnet, C_ideal)

print("One-device QAOA MAE:")
print("Noisy:", metrics_noisy["MAE"])
print("Direct partial:", metrics_direct["MAE"])
print("ChannelNet partial:", metrics_channelnet["MAE"])
print("Improvement direct:", improvement_ratio(metrics_noisy["MAE"], metrics_direct["MAE"]))
print("Improvement ChannelNet:", improvement_ratio(metrics_noisy["MAE"], metrics_channelnet["MAE"]))

fig, axes = plt.subplots(1, 4, figsize=(20, 4))

plots = [
    (C_ideal, "Ideal"),
    (C_noisy_sampled, "Noisy measured"),
    (C_mitigated_direct, "Mitigated\nDirect partial"),
    (C_mitigated_channelnet, "Mitigated\nChannelNet partial"),
]

for ax, (C, title) in zip(axes, plots):
    im = ax.imshow(
        C,
        origin="lower",
        aspect="auto",
        extent=[beta_grid[0], beta_grid[-1], gamma_grid[0], gamma_grid[-1]],
        vmin=0,
        vmax=1
    )
    ax.set_title(title)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\gamma$")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()

# %% Cell 209
# ============================================================
# Multi-device QAOA evaluation
# ============================================================

def best_grid_point(C_grid, gamma_grid, beta_grid):
    idx = np.unravel_index(np.argmax(C_grid), C_grid.shape)

    return {
        "idx": idx,
        "gamma": float(gamma_grid[idx[0]]),
        "beta": float(beta_grid[idx[1]]),
        "cost": float(C_grid[idx]),
    }

def ideal_cost_at_grid_index(idx, C_ideal):
    return float(C_ideal[idx])

def evaluate_qaoa_mitigation_for_device(
    idx_device,
    gamma_grid,
    beta_grid,
    C_ideal_grid,
    Y_true_all,
    Y_direct_all,
    Y_channelnet_all,
    qaoa_shots=2048,
    seed=0
):
    y_true = Y_true_all[idx_device]
    y_direct = Y_direct_all[idx_device]
    y_channelnet = Y_channelnet_all[idx_device]

    A1_true, b1_true, A2_true, b2_true = unflatten_two_local_channels(y_true)
    A1_direct, b1_direct, A2_direct, b2_direct = unflatten_two_local_channels(y_direct)
    A1_pred, b1_pred, A2_pred, b2_pred = unflatten_two_local_channels(y_channelnet)

    C_noisy_sampled = qaoa_landscape_from_channel(
        gamma_grid, beta_grid,
        A1_true, b1_true,
        A2_true, b2_true,
        shots=qaoa_shots,
        seed=seed + 123
    )

    C_direct_model = qaoa_landscape_from_channel(
        gamma_grid, beta_grid,
        A1_direct, b1_direct,
        A2_direct, b2_direct,
        shots=None,
        seed=seed
    )

    C_channelnet_model = qaoa_landscape_from_channel(
        gamma_grid, beta_grid,
        A1_pred, b1_pred,
        A2_pred, b2_pred,
        shots=None,
        seed=seed
    )

    C_direct_mitigated, _ = mitigate_qaoa_landscape_by_bias_correction(
        C_noisy_sampled,
        C_ideal_grid,
        C_direct_model,
        clip=True
    )

    C_channelnet_mitigated, _ = mitigate_qaoa_landscape_by_bias_correction(
        C_noisy_sampled,
        C_ideal_grid,
        C_channelnet_model,
        clip=True
    )

    metrics = {
        "noisy": landscape_error_metrics(C_noisy_sampled, C_ideal_grid),
        "direct": landscape_error_metrics(C_direct_mitigated, C_ideal_grid),
        "channelnet": landscape_error_metrics(C_channelnet_mitigated, C_ideal_grid),
    }

    bests = {
        "ideal": best_grid_point(C_ideal_grid, gamma_grid, beta_grid),
        "noisy": best_grid_point(C_noisy_sampled, gamma_grid, beta_grid),
        "direct": best_grid_point(C_direct_mitigated, gamma_grid, beta_grid),
        "channelnet": best_grid_point(C_channelnet_mitigated, gamma_grid, beta_grid),
    }

    ideal_costs_at_chosen = {
        key: ideal_cost_at_grid_index(best["idx"], C_ideal_grid)
        for key, best in bests.items()
    }

    return {
        "metrics": metrics,
        "bests": bests,
        "ideal_costs_at_chosen": ideal_costs_at_chosen,
    }

def collect_metric(results, method, metric_name):
    return np.array([r["metrics"][method][metric_name] for r in results], dtype=np.float64)

def collect_ideal_chosen_cost(results, method):
    return np.array([r["ideal_costs_at_chosen"][method] for r in results], dtype=np.float64)

n_eval_devices = min(20, len(Y_test))

results_multi = []

for k in range(n_eval_devices):
    result = evaluate_qaoa_mitigation_for_device(
        idx_device=k,
        gamma_grid=gamma_grid,
        beta_grid=beta_grid,
        C_ideal_grid=C_ideal,
        Y_true_all=Y_test,
        Y_direct_all=Y_test_direct,
        Y_channelnet_all=Y_test_pred,
        qaoa_shots=2048,
        seed=9000 + k
    )

    results_multi.append(result)

    print(
        f"Device {k:02d} | "
        f"MAE noisy {result['metrics']['noisy']['MAE']:.4f} | "
        f"direct partial {result['metrics']['direct']['MAE']:.4f} | "
        f"ChannelNet partial {result['metrics']['channelnet']['MAE']:.4f}"
    )

mae_noisy = collect_metric(results_multi, "noisy", "MAE")
mae_direct = collect_metric(results_multi, "direct", "MAE")
mae_channelnet = collect_metric(results_multi, "channelnet", "MAE")

rmse_noisy = collect_metric(results_multi, "noisy", "RMSE")
rmse_direct = collect_metric(results_multi, "direct", "RMSE")
rmse_channelnet = collect_metric(results_multi, "channelnet", "RMSE")

print("\nQAOA mitigation using partial tomography")
print("-" * 85)
print(f"{'Method':<28s} {'MAE mean':>12s} {'MAE median':>12s} {'RMSE mean':>12s}")
print("-" * 85)

for name, mae_arr, rmse_arr in [
    ("Noisy", mae_noisy, rmse_noisy),
    ("Direct partial mitigated", mae_direct, rmse_direct),
    ("ChannelNet partial mitigated", mae_channelnet, rmse_channelnet),
]:
    print(
        f"{name:<28s} "
        f"{mae_arr.mean():12.6f} "
        f"{np.median(mae_arr):12.6f} "
        f"{rmse_arr.mean():12.6f}"
    )

print("\nMean MAE improvement ratios:")
print("Direct partial:", improvement_ratio(mae_noisy.mean(), mae_direct.mean()))
print("ChannelNet partial:", improvement_ratio(mae_noisy.mean(), mae_channelnet.mean()))

# ============================================================
# Reliability bar plot
# ============================================================

methods = ["Noisy", "Direct\npartial", "ChannelNet\npartial"]

means = [mae_noisy.mean(), mae_direct.mean(), mae_channelnet.mean()]
stds = [mae_noisy.std(), mae_direct.std(), mae_channelnet.std()]

xpos = np.arange(len(methods))

plt.figure(figsize=(7, 4))
plt.bar(xpos, means, yerr=stds, capsize=5)
plt.xticks(xpos, methods)
plt.ylabel("MAE vs ideal QAOA landscape")
plt.title("QAOA reliability from partial tomography")
plt.grid(axis="y")
plt.show()

# ============================================================
# Paired improvement plot
# ============================================================

plt.figure(figsize=(6, 5))
plt.scatter(mae_noisy, mae_direct, label="Direct partial", alpha=0.8)
plt.scatter(mae_noisy, mae_channelnet, label="ChannelNet partial", alpha=0.8)

max_axis = max(mae_noisy.max(), mae_direct.max(), mae_channelnet.max())
plt.plot([0, max_axis], [0, max_axis], linestyle="--")

plt.xlabel("Noisy MAE")
plt.ylabel("Mitigated MAE")
plt.title("Paired QAOA reliability improvement")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# Optimum recovery
# ============================================================

chosen_noisy = collect_ideal_chosen_cost(results_multi, "noisy")
chosen_direct = collect_ideal_chosen_cost(results_multi, "direct")
chosen_channelnet = collect_ideal_chosen_cost(results_multi, "channelnet")
chosen_ideal = collect_ideal_chosen_cost(results_multi, "ideal")

print("\nIdeal cost at selected optimum")
print("-" * 85)
print(f"{'Selection method':<32s} {'Mean':>12s} {'Median':>12s} {'Std':>12s}")
print("-" * 85)

for name, arr in [
    ("Ideal optimum", chosen_ideal),
    ("Noisy-selected", chosen_noisy),
    ("Direct-partial-selected", chosen_direct),
    ("ChannelNet-partial-selected", chosen_channelnet),
]:
    print(
        f"{name:<32s} "
        f"{arr.mean():12.6f} "
        f"{np.median(arr):12.6f} "
        f"{arr.std():12.6f}"
    )

methods = ["Ideal\noptimum", "Noisy\nselected", "Direct\npartial", "ChannelNet\npartial"]
means = [chosen_ideal.mean(), chosen_noisy.mean(), chosen_direct.mean(), chosen_channelnet.mean()]
stds = [chosen_ideal.std(), chosen_noisy.std(), chosen_direct.std(), chosen_channelnet.std()]

xpos = np.arange(len(methods))

plt.figure(figsize=(8, 4))
plt.bar(xpos, means, yerr=stds, capsize=5)
plt.xticks(xpos, methods)
plt.ylabel("Ideal cost at selected grid optimum")
plt.title("QAOA optimum recovery from partial tomography")
plt.grid(axis="y")
plt.show()

# ============================================================
# Save results
# ============================================================

np.savez(
    "partial_tomography_channelnet_results.npz",
    X_train=X_train,
    Y_train=Y_train,
    X_val=X_val,
    Y_val=Y_val,
    X_test=X_test,
    Y_test=Y_test,
    shots_train=shots_train,
    shots_val=shots_val,
    shots_test=shots_test,
    Y_test_direct=Y_test_direct,
    Y_test_pred=Y_test_pred,
    err_mean_test=err_mean_test,
    err_direct_test=err_direct_test,
    err_channelnet_test=err_channelnet_test,
    mae_noisy=mae_noisy,
    mae_direct=mae_direct,
    mae_channelnet=mae_channelnet,
    chosen_noisy=chosen_noisy,
    chosen_direct=chosen_direct,
    chosen_channelnet=chosen_channelnet,
    chosen_ideal=chosen_ideal,
    gamma_grid=gamma_grid,
    beta_grid=beta_grid,
    C_ideal=C_ideal
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "X_mean": X_mean,
        "X_std": X_std,
        "Y_mean": Y_mean,
        "Y_std": Y_std,
        "partial_state_labels": partial_state_labels,
        "partial_measurement_labels": partial_measurement_labels,
        "history": history,
    },
    "channelnet_partial_tomography_model.pt"
)

print("Saved partial-tomography results and model.")

# %% Cell 210
import pandas as pd

# ============================================================
# Summary table 1: channel-learning errors
# ============================================================

channel_error_table = pd.DataFrame({
    "Method": [
        "Mean baseline",
        "Direct partial tomography",
        "ChannelNet partial tomography"
    ],
    "Channel MSE": [
        mse_np(Y_test_baseline, Y_test),
        mse_np(Y_test_direct, Y_test),
        mse_np(Y_test_pred, Y_test)
    ],
    "Mean L2 error": [
        err_mean_test.mean(),
        err_direct_test.mean(),
        err_channelnet_test.mean()
    ],
    "Median L2 error": [
        np.median(err_mean_test),
        np.median(err_direct_test),
        np.median(err_channelnet_test)
    ],
    "Std L2 error": [
        err_mean_test.std(),
        err_direct_test.std(),
        err_channelnet_test.std()
    ]
})

channel_error_table

# %% Cell 211
# ============================================================
# Summary table 2: QAOA mitigation errors
# ============================================================

qaoa_error_table = pd.DataFrame({
    "Method": [
        "Noisy measured",
        "Direct partial mitigated",
        "ChannelNet partial mitigated"
    ],
    "Mean MAE vs ideal": [
        mae_noisy.mean(),
        mae_direct.mean(),
        mae_channelnet.mean()
    ],
    "Median MAE vs ideal": [
        np.median(mae_noisy),
        np.median(mae_direct),
        np.median(mae_channelnet)
    ],
    "Std MAE": [
        mae_noisy.std(),
        mae_direct.std(),
        mae_channelnet.std()
    ],
    "Mean RMSE vs ideal": [
        rmse_noisy.mean(),
        rmse_direct.mean(),
        rmse_channelnet.mean()
    ],
    "Improvement ratio vs noisy": [
        1.0,
        improvement_ratio(mae_noisy.mean(), mae_direct.mean()),
        improvement_ratio(mae_noisy.mean(), mae_channelnet.mean())
    ]
})

qaoa_error_table

# %% Cell 212
# ============================================================
# Summary table 3: QAOA optimum recovery
# ============================================================

optimum_recovery_table = pd.DataFrame({
    "Selection method": [
        "Ideal optimum",
        "Noisy-selected",
        "Direct partial selected",
        "ChannelNet partial selected"
    ],
    "Mean ideal cost at selected point": [
        chosen_ideal.mean(),
        chosen_noisy.mean(),
        chosen_direct.mean(),
        chosen_channelnet.mean()
    ],
    "Median ideal cost at selected point": [
        np.median(chosen_ideal),
        np.median(chosen_noisy),
        np.median(chosen_direct),
        np.median(chosen_channelnet)
    ],
    "Std": [
        chosen_ideal.std(),
        chosen_noisy.std(),
        chosen_direct.std(),
        chosen_channelnet.std()
    ]
})

optimum_recovery_table

# %% Cell 213
# ============================================================
# Compact final summary table
# ============================================================

summary_table = pd.DataFrame({
    "Result": [
        "Channel MSE",
        "Channel mean L2 error",
        "QAOA mean MAE vs ideal",
        "QAOA improvement ratio",
        "Ideal cost at selected optimum"
    ],
    "Mean baseline / Noisy": [
        mse_np(Y_test_baseline, Y_test),
        err_mean_test.mean(),
        mae_noisy.mean(),
        1.0,
        chosen_noisy.mean()
    ],
    "Direct partial": [
        mse_np(Y_test_direct, Y_test),
        err_direct_test.mean(),
        mae_direct.mean(),
        improvement_ratio(mae_noisy.mean(), mae_direct.mean()),
        chosen_direct.mean()
    ],
    "ChannelNet partial": [
        mse_np(Y_test_pred, Y_test),
        err_channelnet_test.mean(),
        mae_channelnet.mean(),
        improvement_ratio(mae_noisy.mean(), mae_channelnet.mean()),
        chosen_channelnet.mean()
    ]
})

summary_table

# %% Cell 214
# ============================================================
# FINAL SUMMARY TABLES — WORKS FOR PARTIAL-TOMOGRAPHY VERSION
# Paste after Cell 7 is finished
# ============================================================

import numpy as np
import pandas as pd
from IPython.display import display, Markdown

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def sci(x, digits=3):
    return f"{float(x):.{digits}e}"

def dec(x, digits=5):
    return f"{float(x):.{digits}f}"

def ratio_fmt(x, digits=2):
    return f"{float(x):.{digits}f}×"

def mse_np_safe(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))

def per_example_l2_error_safe(Y_est, Y_true):
    return np.linalg.norm(np.asarray(Y_est) - np.asarray(Y_true), axis=1)

def improvement_ratio_safe(noisy_error, mitigated_error, eps=1e-12):
    return float(noisy_error / (mitigated_error + eps))

# ------------------------------------------------------------
# Check required partial-tomography variables
# ------------------------------------------------------------

required_vars = [
    "Y_test",
    "Y_test_direct",
    "Y_test_pred",
    "shots_test",
    "Y_test_baseline",
    "err_mean_test",
    "err_direct_test",
    "err_channelnet_test",
    "mae_noisy",
    "mae_direct",
    "mae_channelnet",
    "chosen_noisy",
    "chosen_direct",
    "chosen_channelnet",
    "chosen_ideal",
]

missing = [v for v in required_vars if v not in globals()]

if missing:
    raise NameError(
        "Missing variables. Run the partial-tomography notebook until the end of Cell 7 first.\n\n"
        "Missing:\n" + "\n".join(missing)
    )

# ------------------------------------------------------------
# 1. Channel learning final test MSE
# ------------------------------------------------------------

channel_learning_table = pd.DataFrame({
    "Method": [
        "Mean baseline",
        "Direct partial tomography",
        "ChannelNet partial tomography"
    ],
    "Test MSE": [
        sci(mse_np_safe(Y_test_baseline, Y_test), 3),
        sci(mse_np_safe(Y_test_direct, Y_test), 3),
        sci(mse_np_safe(Y_test_pred, Y_test), 3)
    ],
    "Raw Test MSE": [
        mse_np_safe(Y_test_baseline, Y_test),
        mse_np_safe(Y_test_direct, Y_test),
        mse_np_safe(Y_test_pred, Y_test)
    ]
})

display(Markdown("## 1. Channel learning: final test channel-label MSE"))
display(channel_learning_table[["Method", "Test MSE"]])

# ------------------------------------------------------------
# 1b. Channel learning by shot number
# ------------------------------------------------------------

def grouped_errors_by_shots_safe(Y_est, Y_true, shots_array):
    out = {}

    for s in sorted(np.unique(shots_array)):
        mask = np.asarray(shots_array) == s
        errs = per_example_l2_error_safe(np.asarray(Y_est)[mask], np.asarray(Y_true)[mask])

        out[int(s)] = {
            "n": int(mask.sum()),
            "mean": float(errs.mean()),
            "median": float(np.median(errs)),
            "std": float(errs.std()),
        }

    return out

direct_by_shots = grouped_errors_by_shots_safe(Y_test_direct, Y_test, shots_test)
channelnet_by_shots = grouped_errors_by_shots_safe(Y_test_pred, Y_test, shots_test)

shot_rows = []

for s in sorted(np.unique(shots_test)):
    s = int(s)
    d_mean = direct_by_shots[s]["mean"]
    c_mean = channelnet_by_shots[s]["mean"]

    shot_rows.append({
        "Shots": s,
        "Direct partial tomography L2": dec(d_mean, 3),
        "ChannelNet partial L2": dec(c_mean, 3),
        "Better method": "ChannelNet" if c_mean < d_mean else "Direct partial tomography",
        "Raw Direct L2": d_mean,
        "Raw ChannelNet L2": c_mean,
    })

shot_learning_table = pd.DataFrame(shot_rows)

display(Markdown("## 1b. Channel learning by shot number"))
display(shot_learning_table[[
    "Shots",
    "Direct partial tomography L2",
    "ChannelNet partial L2",
    "Better method"
]])

# ------------------------------------------------------------
# 2. Physicality table
# ------------------------------------------------------------

def unflatten_two_local_channels_safe(y):
    y = np.asarray(y)
    A1 = y[0:9].reshape(3, 3)
    b1 = y[9:12]
    A2 = y[12:21].reshape(3, 3)
    b2 = y[21:24]
    return A1, b1, A2, b2

bloch_test_np_safe = np.array([
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
], dtype=np.float64)

for i in range(6, len(bloch_test_np_safe)):
    bloch_test_np_safe[i] /= np.linalg.norm(bloch_test_np_safe[i])

def physicality_violation_np_safe(Y_phys):
    violations = []

    for yi in np.asarray(Y_phys):
        A1, b1, A2, b2 = unflatten_two_local_channels_safe(yi)

        for A, b in [(A1, b1), (A2, b2)]:
            for r in bloch_test_np_safe:
                rout = A @ r + b
                violation = max(0.0, np.linalg.norm(rout) - 1.0)
                violations.append(violation)

    violations = np.array(violations)

    return {
        "mean_violation": float(violations.mean()),
        "max_violation": float(violations.max()),
        "fraction_violating": float(np.mean(violations > 1e-6)),
    }

phys_true = physicality_violation_np_safe(Y_test)
phys_direct = physicality_violation_np_safe(Y_test_direct)
phys_channelnet = physicality_violation_np_safe(Y_test_pred)

physicality_table = pd.DataFrame({
    "Method": [
        "True channel",
        "Direct partial tomography",
        "ChannelNet partial"
    ],
    "Mean Bloch-ball violation": [
        sci(phys_true["mean_violation"], 3),
        sci(phys_direct["mean_violation"], 3),
        sci(phys_channelnet["mean_violation"], 3)
    ],
    "Max violation": [
        dec(phys_true["max_violation"], 4),
        dec(phys_direct["max_violation"], 4),
        dec(phys_channelnet["max_violation"], 4)
    ],
    "Fraction violating": [
        dec(phys_true["fraction_violating"], 4),
        dec(phys_direct["fraction_violating"], 4),
        dec(phys_channelnet["fraction_violating"], 4)
    ]
})

display(Markdown("## 2. Physicality of effective channels"))
display(physicality_table)

# ------------------------------------------------------------
# 3. Channel-learning L2 summary
# ------------------------------------------------------------

channel_l2_table = pd.DataFrame({
    "Method": [
        "Mean baseline",
        "Direct partial tomography",
        "ChannelNet partial tomography"
    ],
    "Mean L2 error": [
        dec(err_mean_test.mean(), 5),
        dec(err_direct_test.mean(), 5),
        dec(err_channelnet_test.mean(), 5)
    ],
    "Median L2 error": [
        dec(np.median(err_mean_test), 5),
        dec(np.median(err_direct_test), 5),
        dec(np.median(err_channelnet_test), 5)
    ],
    "Std L2 error": [
        dec(err_mean_test.std(), 5),
        dec(err_direct_test.std(), 5),
        dec(err_channelnet_test.std(), 5)
    ]
})

display(Markdown("## 3. Channel-learning L2 error summary"))
display(channel_l2_table)

# ------------------------------------------------------------
# 4. QAOA mitigation over multiple hidden devices
# ------------------------------------------------------------

qaoa_table = pd.DataFrame({
    "Method": [
        "Noisy measured",
        "Direct partial mitigated",
        "ChannelNet partial mitigated"
    ],
    "Mean MAE": [
        dec(mae_noisy.mean(), 5),
        dec(mae_direct.mean(), 5),
        dec(mae_channelnet.mean(), 5)
    ],
    "Median MAE": [
        dec(np.median(mae_noisy), 5),
        dec(np.median(mae_direct), 5),
        dec(np.median(mae_channelnet), 5)
    ],
    "Std MAE": [
        dec(mae_noisy.std(), 5),
        dec(mae_direct.std(), 5),
        dec(mae_channelnet.std(), 5)
    ],
    "Improvement ratio": [
        "1.00×",
        ratio_fmt(improvement_ratio_safe(mae_noisy.mean(), mae_direct.mean()), 2),
        ratio_fmt(improvement_ratio_safe(mae_noisy.mean(), mae_channelnet.mean()), 2)
    ]
})

display(Markdown("## 4. QAOA mitigation from partial tomography"))
display(qaoa_table)

# ------------------------------------------------------------
# 5. QAOA optimum recovery
# ------------------------------------------------------------

optimum_table = pd.DataFrame({
    "Selection method": [
        "Ideal optimum",
        "Noisy-selected",
        "Direct partial selected",
        "ChannelNet partial selected"
    ],
    "Mean ideal cost at selected point": [
        dec(chosen_ideal.mean(), 5),
        dec(chosen_noisy.mean(), 5),
        dec(chosen_direct.mean(), 5),
        dec(chosen_channelnet.mean(), 5)
    ],
    "Median ideal cost at selected point": [
        dec(np.median(chosen_ideal), 5),
        dec(np.median(chosen_noisy), 5),
        dec(np.median(chosen_direct), 5),
        dec(np.median(chosen_channelnet), 5)
    ],
    "Std": [
        dec(chosen_ideal.std(), 5),
        dec(chosen_noisy.std(), 5),
        dec(chosen_direct.std(), 5),
        dec(chosen_channelnet.std(), 5)
    ]
})

display(Markdown("## 5. QAOA optimum recovery"))
display(optimum_table)

# ------------------------------------------------------------
# 6. Automatic interpretation
# ------------------------------------------------------------

display(Markdown("## Automatic interpretation"))

channel_mse_winner = (
    "ChannelNet partial tomography"
    if mse_np_safe(Y_test_pred, Y_test) < mse_np_safe(Y_test_direct, Y_test)
    else "Direct partial tomography"
)

channel_l2_winner = (
    "ChannelNet partial tomography"
    if err_channelnet_test.mean() < err_direct_test.mean()
    else "Direct partial tomography"
)

qaoa_winner = (
    "ChannelNet partial mitigation"
    if mae_channelnet.mean() < mae_direct.mean()
    else "Direct partial mitigation"
)

phys_winner = (
    "ChannelNet partial"
    if phys_channelnet["mean_violation"] < phys_direct["mean_violation"]
    else "Direct partial tomography"
)

lines = [
    f"- Lower channel-parameter MSE: **{channel_mse_winner}**.",
    f"- Lower mean channel L2 error: **{channel_l2_winner}**.",
    f"- More physical effective channels by mean Bloch-ball violation: **{phys_winner}**.",
    f"- Better QAOA mitigation by mean MAE: **{qaoa_winner}**.",
    f"- Direct partial QAOA improvement ratio: **{ratio_fmt(improvement_ratio_safe(mae_noisy.mean(), mae_direct.mean()), 2)}**.",
    f"- ChannelNet partial QAOA improvement ratio: **{ratio_fmt(improvement_ratio_safe(mae_noisy.mean(), mae_channelnet.mean()), 2)}**.",
]

display(Markdown("\n".join(lines)))

# ------------------------------------------------------------
# 7. Save tables
# ------------------------------------------------------------

channel_learning_table.to_csv("partial_table_1_channel_learning_mse.csv", index=False)
shot_learning_table.to_csv("partial_table_1b_channel_learning_by_shots.csv", index=False)
physicality_table.to_csv("partial_table_2_physicality.csv", index=False)
channel_l2_table.to_csv("partial_table_3_channel_l2_errors.csv", index=False)
qaoa_table.to_csv("partial_table_4_qaoa_mitigation.csv", index=False)
optimum_table.to_csv("partial_table_5_qaoa_optimum_recovery.csv", index=False)

print("Saved partial-tomography summary tables as CSV files.")
