import numpy as np


def normalizing_constant(gamma, epsilon):
    return (1 - epsilon**(1 - gamma)) / (1 - gamma)


def pdf(gamma=2.5, epsilon=10**-2):
    normalizing_const = normalizing_constant(gamma, epsilon)

    def normalized_pdf(x):
        if not epsilon <= x <= 1.0:
            return 0
        return x**(-gamma) / normalizing_const
    return normalized_pdf


def cdf(gamma=2.5, epsilon=10**-2):
    normalizing_const = normalizing_constant(gamma, epsilon)

    def normalized_cdf(x):
        if x < epsilon:
            return 0
        elif x >= 1:
            return 1
        a = x**(1 - gamma) - epsilon**(1 - gamma)
        b = normalizing_const * (1 - gamma)
        return a / b
    return normalized_cdf


def inv_cdf(gamma=2.5, epsilon=10**-2):
    normalizing_const = normalizing_constant(gamma, epsilon)

    def normalized_inv_cdf(x):
        if not 0 <= x <= 1:
            raise ValueError()
        a = x * normalizing_const * (1 - gamma)
        b = epsilon**(1 - gamma)
        return (a + b)**(1 / (1 - gamma))
    return normalized_inv_cdf


def expected_value(gamma=2.5, epsilon=10**-2):
    normalizing_const = normalizing_constant(gamma, epsilon)
    return (1.0 - epsilon**(2.0 - gamma)) / (normalizing_const * (2 - gamma))


# inverse transform sampling is used to generate samples from the PDF
# https://en.wikipedia.org/wiki/Inverse_transform_sampling
def inverse_transform_sampling(inv_cdf):
    while True:
        yield inv_cdf(np.random.rand())
