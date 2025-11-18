import torch
import matplotlib.pyplot as plt

# -------------------------------
# Problem setup: plant + P feedback + disturbance
# -------------------------------

torch.manual_seed(0)

# System parameters
a = 1.02          # choose |a|<1 or >1 as you like
b = 1.0
eps = 0.05       # stability margin: |a - b*kappa| <= 1 - eps
lambda_u = 0.0   # control effort weight
N = 50           # horizon length

# Regulation reference (origin)
r = torch.zeros(N + 1)

# Disturbance bound
w_max = 0.1

# Fixed disturbance realization (bounded)
w = w_max * (2.0 * torch.rand(N + 1) - 1.0)   # uniform in [-w_max, w_max]


def simulate_closed_loop(kappa, y0=0.5):
    """
    Closed-loop with disturbance:
        e_k = r_k - y_k
        u_k = kappa * e_k
        y_{k+1} = a*y_k + b*u_k + w_k

    Returns y[0..N], u[0..N-1]
    """
    y = torch.zeros(N + 1)
    u = torch.zeros(N)

    y[0] = y0

    for k in range(N):
        e = r[k] - y[k]
        u[k] = kappa * e
        y[k+1] = a * y[k] + b * u[k] + w[k]

    return y, u


def stability_residual(kappa):
    """
    Stability residual (for monitoring only):
        C_stab = |a - b*kappa| - (1 - eps)
    Stability region: C_stab <= 0.
    """
    return torch.abs(a - b * kappa) - (1.0 - eps)


def disturbance_residual(w, w_max):
    """
    Disturbance contract residual (for monitoring):
        C_dist = max_k |d_k| - d_max
    """
    return w.abs().max() - w_max


def loss_function(kappa):
    """
    Objective WITHOUT stability penalty:
        J(kappa) = sum_k (|y_k|^2 + lambda_u |u_k|^2)
    Stability is enforced by projection, not by penalty.
    """
    y, u = simulate_closed_loop(kappa)
    tracking_term = torch.sum(y**2)
    effort_term = lambda_u * torch.sum(u**2)
    return tracking_term + effort_term


# -------------------------------
# Stability interval for projection
# -------------------------------

# Constraint: |a - b*kappa| <= 1 - eps
# => kappa in [(a - (1-eps))/b, (a + (1-eps))/b]
kappa_min = (a - (1.0 - eps)) / b
kappa_max = (a + (1.0 - eps)) / b

print(f"Stability interval for kappa: [{kappa_min:.3f}, {kappa_max:.3f}]")


# -------------------------------
# Untuned gain (for comparison)
# -------------------------------

kappa_untuned = torch.tensor(0.0)
y_untuned, u_untuned = simulate_closed_loop(kappa_untuned)

# -------------------------------
# Projected gradient tuning of kappa
# -------------------------------

# Start from some initial gain
kappa = torch.tensor(0.5, requires_grad=True)
optimizer = torch.optim.Adam([kappa], lr=0.05)

num_iters = 100
history_kappa = []
history_loss = []
history_Cstab = []

for it in range(num_iters):
    optimizer.zero_grad()
    L = loss_function(kappa)
    L.backward()
    optimizer.step()

    # ---- Projection step: enforce |a - b*kappa| <= 1 - eps ----
    with torch.no_grad():
        kappa.clamp_(kappa_min, kappa_max)

    with torch.no_grad():
        C_val = stability_residual(kappa)

    history_kappa.append(kappa.item())
    history_loss.append(L.item())
    history_Cstab.append(C_val.item())

kappa_tuned = kappa.detach().item()
C_stab_tuned = stability_residual(torch.tensor(kappa_tuned)).item()
C_dist = disturbance_residual(w, w_max).item()

print(f"Untuned gain kappa_0       = {kappa_untuned.item():.4f}")
print(f"Tuned gain   kappa_star    = {kappa_tuned:.4f}")
print(f"Stability residual C_stab  = {C_stab_tuned:.4f}")
print(f"Disturbance residual C_dist= {C_dist:.4f}")

# Simulate tuned closed loop
y_tuned, u_tuned = simulate_closed_loop(torch.tensor(kappa_tuned))

# -------------------------------
# Plots
# -------------------------------

time = range(N + 1)
# 1) Closed-loop responses: untuned vs tuned vs disturbance
plt.figure(figsize=(7, 2))  # Increase width and reduce height
plt.plot(time, y_untuned.numpy(), "--", label="untuned output")
plt.plot(time, y_tuned.numpy(), label="tuned output")
plt.plot(time, w.numpy(), ":", label="disturbance $w_k$")
# --- Add disturbance ISS steady-state bounds ---
plt.axhline(+w_max, color="black", linestyle="--", linewidth=1.2, label=r"$\pm w_{\max}$")
plt.axhline(-w_max, color="black", linestyle="--", linewidth=1.2)
plt.xlabel("time step $k$", fontsize=14)
plt.ylabel("$y_k$", fontsize=14)
plt.title("Closed-loop response with bounded disturbance", fontsize=14)
plt.legend(fontsize=12, loc="upper right")
plt.grid(True)
plt.savefig("closed_loop_response.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Combined compact subplot: gain evolution + stability residual
fig, axes = plt.subplots(2, 1, figsize=(7, 3), sharex=True)  # Increase width and reduce height

# --- 1) Gain evolution ---
axes[0].plot(range(num_iters), history_kappa)
axes[0].set_ylabel(r"$\kappa$", fontsize=14)
axes[0].set_title("Gain evolution during tuning", fontsize=14)
axes[0].grid(True)

# --- 2) Stability residual ---
axes[1].plot(range(num_iters), history_Cstab)
axes[1].axhline(0.0, linestyle="--", color="black")
axes[1].set_xlabel("iteration", fontsize=14)
axes[1].set_ylabel(r"$C_{\mathrm{stab}}$", fontsize=14)
axes[1].set_title("Stability residual during tuning", fontsize=14)
axes[1].grid(True)

plt.tight_layout()
plt.savefig("gain_and_stability_residual.pdf", format="pdf", bbox_inches="tight")
plt.show()