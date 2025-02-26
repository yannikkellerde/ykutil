from scipy.stats import beta, norm
from math import sqrt
from scipy.integrate import dblquad


def clopper_pearson_interval(successes, trials, confidence_level=0.95):
    """
    Compute the Clopper-Pearson confidence interval.

    Parameters:
        successes (int): Number of successes.
        trials (int): Total number of trials.
        confidence_level (float): Confidence level (e.g., 0.95 for 95% confidence).

    Returns:
        (lower_bound, upper_bound): The confidence interval bounds.
    """
    assert successes <= trials
    alpha = 1 - confidence_level
    lower_bound = (
        beta.ppf(alpha / 2, successes, trials - successes + 1) if successes > 0 else 0.0
    )
    upper_bound = (
        beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
        if successes < trials
        else 1.0
    )
    return lower_bound, upper_bound


def wald_proportion_diff(s1: int, t1: int, s2: int, t2: int, ci: float = 0.95):
    """Given samples of two bernoulli distributions, compute the Wald test statistic.

    >>> wald_proportion_diff(70, 100, 50, 100)
    (0.06706877501808317, 0.33293122498191674)
    """

    p1 = s1 / t1
    p2 = s2 / t2

    se = sqrt(p1 * (1 - p1) / t1 + p2 * (1 - p2) / t2)

    z = norm.ppf(1 - (1 - ci) / 2)

    interval = (p1 - p2 - z * se, p1 - p2 + z * se)

    return interval


def exact_bernoulli_p_value(s1: int, t1: int, s2: int, t2: int):
    """Given samples of two bernoulli distributions, compute the exact p-value.
    Bayesian approach using beta function integration

    >>> exact_bernoulli_p_value(70, 100, 50, 100)
    0.001979681257445076
    """
    # Beta values for Bayesian estimation
    f1 = t1 - s1 + 1
    f2 = t2 - s2 + 1
    s1 = s1 + 1
    s2 = s2 + 1

    # Define the joint probability density function for Beta distributions
    def joint_beta_pdf(p1, p2, s1, f1, s2, f2):
        return beta.pdf(p1, s1, f1) * beta.pdf(p2, s2, f2)

    # Compute P(p1 > p2) via numerical integration
    prob_integral, _ = dblquad(
        lambda p2, p1: joint_beta_pdf(p1, p2, s1, f1, s2, f2),
        0,
        1,  # p1 bounds
        lambda p1: 0,
        lambda p1: p1,  # p2 bounds (p2 < p1)
    )

    return 1 - prob_integral


def wald_proportion_p_value(s1: int, t1: int, s2: int, t2: int):
    """Given samples of two bernoulli distributions, compute the Wald test p-value.
    Frequentist approach

    >>> wald_proportion_p_value(70, 100, 50, 100)
    0.003189699706216853
    """
    p1 = s1 / t1
    p2 = s2 / t2

    se = sqrt(p1 * (1 - p1) / t1 + p2 * (1 - p2) / t2)

    z = (p1 - p2) / se

    p_value = 2 * (1 - norm.cdf(abs(z)))

    return p_value


def double_diff_p_value(
    s1: int, t1: int, s2: int, t2: int, s3: int, t3: int, s4: int, t4: int
):
    """Given successes s1,s2,s3,s4 of bernoulli distributions of totals t1,t2,t3,t4
    and probabilities p1,p2,p3,p4, compute the p-value that (p1-p2)>(p3-p4)
    """
    p1_hat = s1 / t1
    p2_hat = s2 / t2
    p3_hat = s3 / t3
    p4_hat = s4 / t4

    D1_hat = p1_hat - p2_hat
    D2_hat = p3_hat - p4_hat

    var_D1 = (p1_hat * (1 - p1_hat) / t1) + (p2_hat * (1 - p2_hat) / t2)
    var_D2 = (p3_hat * (1 - p3_hat) / t3) + (p4_hat * (1 - p4_hat) / t4)

    var_X = var_D1 + var_D2
    std_X = sqrt(var_X)

    Z = (D1_hat - D2_hat) / std_X

    p_value = 1 - norm.cdf(Z)

    return p_value


if __name__ == "__main__":
    import doctest

    doctest.testmod()
