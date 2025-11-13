"""
Joint learning of stabilizing neural policy and neural Lyapunov function
for a controlled Van der Pol oscillator using Neuromancer.

- System: Van der Pol oscillator with control
- Controller: explicit neural policy pi_theta(x, r)
- Lyapunov: input-convex/PosDef neural network V_phi(x)
- Objective: stabilize to origin + enforce discrete-time Lyapunov decrease
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import neuromancer.psl as psl
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.modules.activations import activations
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.dynamics import ode, integrators
from neuromancer.plot import pltCL, pltPhase


def plot_Lyapunov(net, trajectories=None, xmin=-2, xmax=2, save_path=None):
    # Sample state space and get function values
    x = torch.arange(xmin, xmax, 0.1)
    y = torch.arange(xmin, xmax, 0.1)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    features = torch.stack([xx, yy], dim=-1)

    uu = net(features)
    plot_u = uu.detach().numpy()[:, :, 0]
    plot_x = xx.detach().numpy()
    plot_y = yy.detach().numpy()

    # Create figure
    fig = plt.figure(figsize=(12, 6))

    # Adjust the size of the left subplot (3D plot)
    ax1 = fig.add_axes([0.05, 0.1, 0.48, 0.8], projection="3d")  # Left subplot (10% bigger)
    ax2 = fig.add_subplot(122)  # Right subplot (2D plot)

    # Plot surface of the Lyapunov function
    ax1.plot_surface(plot_x, plot_y, plot_u,
                     cmap=cm.viridis,
                     linewidth=0, antialiased=False)
    ax1.contour(plot_x, plot_y, plot_u, 20, offset=-1,
                cmap=cm.viridis, linestyles="solid")

    # Plot minimum of the Lyapunov function
    min_idx = np.where(plot_u == np.min(plot_u))
    point_u = plot_u[min_idx]
    point_x = plot_x[min_idx]
    point_y = plot_y[min_idx]
    ax1.scatter(point_x, point_y, point_u, color='red', s=100, marker='o')

    # Set labels with larger font size
    ax1.set_ylabel('$x_1$', fontsize=18)
    ax1.set_xlabel('$x_2$', fontsize=18)
    ax1.set_zlabel('$V$', fontsize=18)
    ax1.title.set_fontsize(18)
    ax1.tick_params(axis='both', labelsize=16)

    # Plot sample trajectory
    if trajectories is not None:
        ax2.contour(plot_x, plot_y, plot_u, 20, alpha=0.5,
                    cmap=cm.viridis, linestyles="solid")
        for i in range(trajectories.shape[0]):
            ax2.plot(trajectories[i, :, 1].detach().numpy(),
                     trajectories[i, :, 0].detach().numpy(),
                     'b--', linewidth=2.0, alpha=0.8)
        ax2.scatter(point_x, point_y, color='red', s=50, marker='o')
        ax2.set_aspect('equal')

        # Set labels with larger font size
        ax2.set_ylabel('$x_1$', fontsize=18)
        ax2.set_xlabel('$x_2$', fontsize=18)
        ax2.title.set_fontsize(18)
        ax2.tick_params(axis='both', labelsize=16)
        ax2.set_xlim([xmin, xmax])
        ax2.set_ylim([xmin, xmax])

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.6)

    # Save and show the plot
    plt.savefig("lyapunov.pdf", format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    """
    # # #  Ground truth system model
    """
    gt_model = psl.nonautonomous.VanDerPolControl()
    ts = gt_model.params[1]['ts']      # sampling time
    nx = gt_model.nx                   # number of states
    nu = gt_model.nu                   # number of inputs
    nref = nx                          # ref dimension

    # constraints bounds
    umin, umax = -5., 5.
    xmin, xmax = -4., 4.

    """
    # # #  Dataset (initial states + reference = 0)
    """
    nsteps = 50          # prediction horizon for training
    n_samples = 2000     # number of sampled scenarios

    # Random initial conditions drawn from a box around the origin
    train_data = DictDataset(
        {
            'x': torch.randn(n_samples, 1, nx),
            'r': torch.zeros(n_samples, nsteps+1, nx),
        },
        name='train'
    )
    dev_data = DictDataset(
        {
            'x': torch.randn(n_samples, 1, nx),
            'r': torch.zeros(n_samples, nsteps+1, nx),
        },
        name='dev'
    )

    batch_size = 200
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        collate_fn=train_data.collate_fn, shuffle=False
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_data, batch_size=batch_size,
        collate_fn=dev_data.collate_fn, shuffle=False
    )

    """
    # # #  White-box ODE model + integrator
    """
    vdp_ode = ode.VanDerPolControl()
    vdp_ode.mu = nn.Parameter(torch.tensor(gt_model.mu), requires_grad=False)
    integrator = integrators.RK4(vdp_ode, h=torch.tensor(ts))
    model = Node(integrator, ['x', 'u'], ['x'], name='model')

    """
    # # #  Neural policy pi_theta(x, r)
    """
    net = blocks.MLP_bounds(
        insize=nx + nref, outsize=nu, hsizes=[32, 32],
        nonlin=activations['gelu'], min=umin, max=umax
    )
    policy = Node(net, ['x', 'r'], ['u'], name='policy')

    """
    # # #  Neural Lyapunov function V_phi(x)
    """
    # Input-convex NN + positive-definite wrapper ensures V(x) >= 0 and V(0)=0
    g = blocks.InputConvexNN(insize=nx, outsize=1, hsizes=[32, 32, 32])
    lyap_net = blocks.PosDef(g, eps=0.0)       # eps=0 => V(0) = 0, V(x) > 0 for x≠0
    lyap_net.requires_grad_(True)
    lyapunov = Node(lyap_net, ['x'], ['V'], name='lyapunov')

    """
    # # #  Closed-loop system with Lyapunov output
    """
    cl_system = System(
        [policy, model, lyapunov],
        nsteps=nsteps,
        name='cl_system'
    )
    cl_system.show()

    """
    # # #  DPC + Lyapunov objectives and constraints
    """
    x = variable('x')
    ref = variable('r')
    V = variable('V')

    # 1) Regulation objective: drive x -> 0
    regulation_loss = 100.0 * ((x == ref) ^ 2)   # squared tracking error
    regulation_loss.name = 'state_loss'

    # 2) State box constraints
    state_lower_bound_penalty = 10.0 * (x > xmin)
    state_upper_bound_penalty = 10.0 * (x < xmax)
    state_lower_bound_penalty.name = 'x_min'
    state_upper_bound_penalty.name = 'x_max'

    # 3) Terminal set tightening around origin
    terminal_lower_bound_penalty = 20.0 * (x[:, [-1], :] > ref - 0.01)
    terminal_upper_bound_penalty = 20.0 * (x[:, [-1], :] < ref + 0.01)
    terminal_lower_bound_penalty.name = 'x_N_min'
    terminal_upper_bound_penalty.name = 'x_N_max'

    # 4) Discrete-time Lyapunov decrease along closed-loop trajectory:
    #    V(x_{k+1}) - V(x_k) < -eps   for all k
    eps_lyap = 1e-2
    lyap_decrease = 10.0 * (V[:, 1:, :] - V[:, :-1, :] < -eps_lyap)
    lyap_decrease.name = 'lyap_dec'

    # Objectives and constraints
    objectives = [
        regulation_loss,
    ]
    constraints = [
        state_lower_bound_penalty,
        state_upper_bound_penalty,
        terminal_lower_bound_penalty,
        terminal_upper_bound_penalty,
        lyap_decrease,        # Lyapunov decrease constraint
    ]

    """
    # # #  Differentiable optimal control problem
    """
    components = [cl_system]
    loss = PenaltyLoss(objectives, constraints)
    problem = Problem(components, loss)
    problem.show()

    """
    # # #  Training (jointly on policy + Lyapunov parameters)
    """
    optimizer = torch.optim.AdamW(problem.parameters(), lr=0.002)

    trainer = Trainer(
        problem,
        train_loader, dev_loader,
        optimizer,
        epochs=100,
        train_metric='train_loss',
        eval_metric='dev_loss',
        warmup=100,
    )

    best_model = trainer.train()
    trainer.model.load_state_dict(best_model)

    """
    # # #  Closed-loop test + Lyapunov trajectory check
    """
    print('\nTest Closed Loop System with learned policy and V(x)\n')
    nsteps_test = 100
    cl_system.nsteps = nsteps_test

    data = {
        'x': torch.randn(1, 1, nx, dtype=torch.float32),
        'r': torch.zeros(1, nsteps_test+1, nx, dtype=torch.float32),
    }
    trajectories = cl_system(data)

    # Extract trajectories
    X_traj = trajectories['x'].detach().reshape(nsteps_test + 1, nx)
    U_traj = trajectories['u'].detach().reshape(nsteps_test, nu)
    R_traj = trajectories['r'].detach().reshape(nsteps_test + 1, nx)
    V_traj = trajectories['V'].detach().reshape(nsteps_test, 1)

    # Constraints bounds for plotting
    Umin = umin * np.ones([nsteps_test, 1])
    Umax = umax * np.ones([nsteps_test, 1])
    Xmin = xmin * np.ones([nsteps_test+1, 1])
    Xmax = xmax * np.ones([nsteps_test+1, 1])


    # Simple Lyapunov check: print ΔV_k
    dV = V_traj[1:] - V_traj[:-1]
    print("V(x_0) =", V_traj[0].item())
    print("min ΔV over trajectory:", dV.min().item())
    print("max ΔV over trajectory:", dV.max().item())

    # --- Three-subplot figure: states + reference, control + bounds, ΔV ---
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=False)

    k_x = np.arange(nsteps_test + 1)
    k_u = np.arange(nsteps_test)
    k_dV = np.arange(nsteps_test - 1)

    small = 11  # tick label font size
    label_fs = 12  # axis label font size
    title_fs = 12  # title font size
    legend_fs = 11  # legend font size

    # ---------------------- Subplot 1: states vs reference ----------------------
    ax0 = axes[0]
    for i in range(nx):
        ax0.plot(k_x, R_traj[:, i], 'r--', linewidth=2,
                 label=r'$r_k$' if i == 0 else None)
    for i in range(nx):
        ax0.plot(k_x, X_traj[:, i], color='orange', linewidth=2,
                 label=r'$x_k$' if i == 0 else None)

    ax0.set_ylabel('States', fontsize=label_fs)
    ax0.set_title('Closed-loop states and reference', fontsize=title_fs, pad=6)
    ax0.grid(True, alpha=0.4)
    ax0.tick_params(axis='both', labelsize=small)
    ax0.legend(fontsize=legend_fs, loc='best')

    # ---------------------- Subplot 2: control with bounds ----------------------
    ax1 = axes[1]
    ax1.plot(k_u, U_traj[:, 0], color='tab:blue', linewidth=2, label=r'$u_k$')
    ax1.plot(k_u, Umin[:, 0], 'k--', linewidth=1.5, label=r'$u_{\min}$')
    ax1.plot(k_u, Umax[:, 0], 'k--', linewidth=1.5, label=r'$u_{\max}$')

    ax1.set_ylabel('Control', fontsize=label_fs)
    ax1.set_title('Control input and bounds', fontsize=title_fs, pad=6)
    ax1.grid(True, alpha=0.4)
    ax1.tick_params(axis='both', labelsize=small)
    ax1.legend(fontsize=legend_fs, loc="lower right")

    # ---------------------- Subplot 3: ΔV trajectory ----------------------
    ax2 = axes[2]
    ax2.plot(k_dV, dV.numpy().flatten(), marker='o', linestyle='-',
             linewidth=2, label=r'$\Delta V_k$')
    ax2.axhline(0.0, color='black', linestyle='--', linewidth=1, label='Zero')

    ax2.set_xlabel('Step k', fontsize=label_fs)
    ax2.set_ylabel(r'$\Delta V_k$', fontsize=label_fs)
    ax2.set_title('Lyapunov contract residual', fontsize=title_fs, pad=6)
    ax2.grid(True, alpha=0.4)
    ax2.tick_params(axis='both', labelsize=small)
    ax2.legend(fontsize=legend_fs, loc='best')

    # ---------------------- Tight layout adjustments ----------------------
    plt.tight_layout(pad=0.8)
    plt.subplots_adjust(hspace=0.35)  # reduce vertical whitespace

    plt.savefig("lyapunov_control.pdf", format="pdf", bbox_inches="tight")
    plt.show()


    """
    Visualize learned Lyapunov function with sampled trajectories
    """

    data = {
        'x': torch.randn(20, 1, nx, dtype=torch.float32),
        'r': torch.zeros(20, nsteps_test+1, nx, dtype=torch.float32),
    }
    trajectories = cl_system(data)

    plot_Lyapunov(lyap_net, trajectories=trajectories['x'].detach(),
                  xmin=-2, xmax=2, save_path=None)


