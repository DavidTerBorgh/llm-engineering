def is_retryable(err):
    status = getattr(err, "status_code", None)
    if status is not None:
        return status in RETRYABLE_STATUS
    return type(err).__name__ in RETRYABLE_NAMES


def wait_time(err, attempt, base=1.0, cap=30.0):
    """How long to sleep before the next attempt."""
    # 1. The provider knows better than we do.
    response = getattr(err, "response", None)
    header = getattr(response, "headers", {}) or {}
    told = header.get("retry-after")
    if told:
        try:
            return float(told)
        except ValueError:
            pass

    # 2. Otherwise: exponential, capped, with full jitter.
    ceiling = min(cap, base * (2 ** attempt))
    return random.uniform(0, ceiling)

def robust_call(messages, model=None, max_attempts=5, base=1.0,
                verbose=True, **kwargs):
    """A call that retries what is worth retrying and refuses what is not."""
    attempts = []

    for attempt in range(max_attempts):
        try:
            r = client.chat.completions.create(
                model=model or MODEL, messages=messages, **kwargs)
            if attempt and verbose:
                print(f"  succeeded on attempt {attempt + 1}")
            robust_call.last_attempts = attempt + 1
            return r

        except Exception as e:
            status = getattr(e, "status_code", "n/a")
            attempts.append((type(e).__name__, status))

            if not is_retryable(e):
                if verbose:
                    print(f"  giving up immediately: {type(e).__name__} {status}")
                raise CallFailed(f"not retryable: {type(e).__name__} {status}") from e

            if attempt == max_attempts - 1:
                raise CallFailed(f"failed after {max_attempts} attempts") from e

            delay = wait_time(e, attempt, base=base)
            if verbose:
                print(f"  attempt {attempt + 1} failed ({status}), sleeping {delay:.2f}s")
            time.sleep(delay)