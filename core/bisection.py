def find_min_a_ctrl(A_min, omega, check_fn, lo=0.0, hi_init=0.1, tol=1e-3, max_expand=20, max_iter=40):
    hi = hi_init; expands = 0
    while not check_fn(hi):
        hi *= 2.0; expands += 1
        if expands > max_expand:
            return hi
    left, right, it = lo, hi, 0
    while right - left > tol and it < max_iter:
        mid = 0.5*(left+right)
        if check_fn(mid): right = mid
        else: left = mid
        it += 1
    return 0.5*(left+right)
