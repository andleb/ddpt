import torch
import torch.nn.functional as F

P_UNCOND = 0.5
V = 0.5
LAMBDAMIN = torch.tensor(1)
LAMBDAMAX = torch.tensor(10)

# NOTE: select something out of support for the null embedding
NULL_CLASS = torch.tensor(-100)


def alphasq(lmbda):
    return 1 / (1 + torch.exp(-lmbda))


def alpha(lmbda):
    return torch.sqrt(alphasq(lmbda))


def sigmasq(lmbda):
    return 1. - alphasq(lmbda)


def sigmasqcond(lmbda, lmbdap):
    return sigmasq(lmbda) * (1 - torch.exp(lmbda - lmbdap))


def sigma(lmbda):
    return torch.sqrt(sigmasq(lmbda))


def zlambda(x, eps, lmbda):
    return alpha(lmbda) * x + sigma(lmbda) * eps


def plambda(lmbdamin, lmbdamax, shape):
    b = torch.arctan(torch.exp(-lmbdamax / 2))
    a = torch.arctan(torch.exp(-lmbdamin / 2)) - b

    u = torch.rand(shape) * a * torch.ones(shape) + b

    return -2 * torch.log(torch.tan(u))


def mutilde(z, x, lmbdap, lmbda):
    return (1 - torch.exp(lmbda - lmbdap)) * alpha(lmbdap) * x + z * (alpha(lmbdap) / alpha(lmbda)) * torch.exp(
        lmbda - lmbdap)


def sigmasqtilde(lmbdap, lmbda):
    return (1 - torch.exp(lmbda - lmbdap)) * sigmasq(lmbdap)


def pz(z, x, v, lmbdap, lmbda):
    return torch.distributions.Normal(
        mutilde(z, x, lmbdap, lmbda),
        # torch.ones_like(z) *
        torch.sqrt(sigmasqtilde(lmbdap, lmbda)**(1 - v) * sigmasqcond(lmbda, lmbdap)**v)
    )


def xtheta(z, eps, lmbda):
    return (z - sigma(lmbda) * eps) / alpha(lmbda)


def compute_loss(X, y, epsmodel, p_unc=0.5):
    # Don't sample the data, full batch

    keep_mask = torch.zeros_like(y).float().uniform_(0, 1) >= p_unc
    yy = y.clone()

    yy = torch.where(
        keep_mask,
        yy,
        NULL_CLASS
    )

    lmbda = plambda(LAMBDAMIN, LAMBDAMAX, y.shape)
    eps = torch.randn_like(X)

    # z = alpha(lmbda) * X + sigma(lmbda) * eps
    z = zlambda(X, eps, lmbda)

    # takes the mean by default
    return F.mse_loss(epsmodel(torch.cat((z, yy), dim=1)), eps)


def sample(epsmodel, y, w=1, v=V, count=1000, timesteps=250):
    lmbdas = torch.linspace(LAMBDAMIN, LAMBDAMAX, timesteps)
    z = torch.randn(count, 1)

    # TODO: adapt to the case when a vector is passed
    y = torch.tensor(y).repeat(count, 1)

    sample_history = [z.detach().cpu()]

    # for t, lmbda in enumerate(tqdm(lmbdas[:-1])):
    for t, lmbda in enumerate(lmbdas[:-1]):
        epstilde = (1 + w) * epsmodel(torch.cat((z, y), dim=1)) \
                   - w * epsmodel(torch.cat((z, NULL_CLASS * torch.ones(count, 1)), dim=1))

        xtilde = xtheta(z, epstilde, lmbda)

        p = pz(z, xtilde, v, lmbdas[t + 1], lmbda)
        z = p.sample()

        sample_history.append(z.detach().cpu())

    # last step
    epstilde = (1 + w) * epsmodel(torch.cat((z, y), dim=1)) \
               - w * epsmodel(torch.cat((z, NULL_CLASS * torch.ones(count, 1)), dim=1))

    xtilde = xtheta(z, epstilde, lmbdas[-1])
    sample_history.append(xtilde.detach().cpu())

    return sample_history
