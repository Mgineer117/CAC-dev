import torch


def flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters() if p.requires_grad])


def set_flat_params(model, flat_params):
    pointer = 0
    for p in model.parameters():
        if p.requires_grad:
            num_param = p.numel()
            p.data.copy_(flat_params[pointer : pointer + num_param].view_as(p))
            pointer += num_param


def compute_kl(old_actor, new_policy, states):
    _, old_infos = old_actor(states)
    _, new_infos = new_policy(states)

    kl = torch.distributions.kl_divergence(old_infos["dist"], new_infos["dist"])
    return kl.mean()


def hessian_vector_product(kl_fn, model, damping, v):
    kl = kl_fn()
    model_params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(kl, model_params, create_graph=True)
    flat_grads = torch.cat([g.view(-1) for g in grads])
    g_v = (flat_grads * v).sum()
    hv = torch.autograd.grad(g_v, model_params)
    flat_hv = torch.cat([h.contiguous().view(-1) for h in hv])
    return flat_hv + damping * v


def conjugate_gradients(Av_func, b, nsteps=10, tol=1e-8):
    """
    Returns:
        x (tensor): The solution
        error (tensor): Final squared residual
        consistency_ratio (float): (increments / decrements).
                                   Lower (near 0) is better.
                                   High values (>0.5) indicate instability.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)

    # Initialize counters for loss behavior
    increments = 0
    decrements = 0

    # Initialize error in case nsteps=0
    new_rdotr = rdotr

    for i in range(nsteps):
        Avp = Av_func(p)

        # Curvature check
        pAvp = torch.dot(p, Avp)
        if pAvp <= 1e-8:
            # Ill-conditioned (curvature failure)
            # We count this as a "instability" but return what we have
            break

        alpha = rdotr / (pAvp + 1e-8)
        x += alpha * p
        r -= alpha * Avp

        # Check new residual error
        new_rdotr = torch.dot(r, r)

        # --- TRACKING LOGIC ---
        if new_rdotr > rdotr:
            increments += 1
        else:
            decrements += 1
        # ----------------------

        if new_rdotr < tol:
            break

        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr

    # Calculate Consistency Ratio
    # Add epsilon to prevent division by zero if it never decreased (bad case)
    consistency_ratio = increments / (decrements + 1e-8)

    return x, new_rdotr, consistency_ratio


def estimate_advantages(
    rewards, terminals, values, gamma=0.99, gae=0.95, device=torch.device("cpu")
):
    rewards, terminals, values = (
        rewards.to(torch.device("cpu")),
        terminals.to(torch.device("cpu")),
        values.to(torch.device("cpu")),
    )
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * (1 - terminals[i]) - values[i]
        advantages[i] = deltas[i] + gamma * gae * prev_advantage * (1 - terminals[i])

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()
    advantages, returns = advantages.to(device), returns.to(device)
    return advantages, returns
