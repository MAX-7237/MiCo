"""Finite MiCo stage-one schedules; exact integer nearest-even arithmetic."""
ALPHAS = ((2, 1),)


def validate_alpha(alpha_num, alpha_den):
    if (type(alpha_num) is not int or type(alpha_den) is not int
            or (alpha_num, alpha_den) not in ALPHAS):
        raise ValueError('layer sweep fixes alpha exactly at 2/1')


def round_ratio_even(numerator, denominator):
    if type(numerator) is not int or type(denominator) is not int or numerator < 0 or denominator <= 0:
        raise ValueError('nonnegative integer numerator and positive denominator required')
    quotient, remainder = divmod(numerator, denominator)
    return quotient + int(2 * remainder > denominator or
                          (2 * remainder == denominator and quotient % 2 == 1))


def mico_budget(target, n0, *, alpha_num=2, alpha_den=1, cut_after_layer=4):
    validate_alpha(alpha_num, alpha_den)
    if (type(target) is not int or target < 1 or type(n0) is not int or n0 < 1
            or type(cut_after_layer) is not int or cut_after_layer not in range(1, 16)):
        raise ValueError('positive target/N0 and physical post-layer K1..15 required')
    effective = min(target, n0)
    if effective == n0:
        return dict(n1=n0, n2=n0, effective_target=effective)
    n1 = min(round_ratio_even(alpha_num * effective, alpha_den), n0)
    n2 = round_ratio_even(32 * effective - cut_after_layer * n1, 32 - cut_after_layer)
    if not 1 <= n2 <= n1:
        raise ValueError('unreachable layer-average MiCo budget')
    return dict(n1=n1, n2=n2, effective_target=effective)
