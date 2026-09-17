from concurrent.futures import ThreadPoolExecutor


PROMPTS = {}
LABELS = {}
EVAL_SET = []
WORKERS = 4
robust_call = None
model_kwargs = lambda model=None: {}


def configure(labels, eval_set, robust_call_fn, model_kwargs_fn=None, workers=4):
    """Connect the notebook's data and client helpers to this harness."""
    global LABELS, EVAL_SET, WORKERS, robust_call, model_kwargs
    LABELS = labels
    EVAL_SET = eval_set
    WORKERS = workers
    robust_call = robust_call_fn
    model_kwargs = model_kwargs_fn or (lambda model=None: {})


def register(name, version, system, build_user):
    """Store a prompt under a name and version."""
    PROMPTS[f"{name}@{version}"] = {"system": system, "build_user": build_user}


def classify(prompt_key, message, max_out=512, **kwargs):
    """Send one message through one registered prompt."""
    if robust_call is None:
        raise RuntimeError("call configure() before classify()")

    p = PROMPTS[prompt_key]
    messages = []
    if p["system"]:
        messages.append({"role": "system", "content": p["system"]})
    messages.append({"role": "user", "content": p["build_user"](message)})

    r = robust_call(messages, max_completion_tokens=max_out,
                    **model_kwargs(), **kwargs)
    choice = r.choices[0]
    text = (choice.message.content or "").strip().lower()

    if not text and choice.finish_reason == "length":
        text = "<<empty: budget spent on reasoning - raise max_out>>"

    return text, r.usage


def normalise(raw):
    """Map a messy reply onto one of our labels, or None."""
    for label in LABELS:
        if label in raw:
            return label
    return None


def evaluate(prompt_key, eval_set=None, workers=None, **kwargs):
    """Accuracy and token cost for one prompt."""
    eval_set = eval_set or EVAL_SET

    def one(item):
        message, gold = item
        try:
            raw, usage = classify(prompt_key, message, **kwargs)
            return normalise(raw), gold, usage, raw
        except Exception as e:
            return None, gold, None, f"ERROR {type(e).__name__}"

    with ThreadPoolExecutor(max_workers=workers or WORKERS) as pool:
        results = list(pool.map(one, eval_set))

    correct = sum(1 for pred, gold, _, _ in results if pred == gold)
    return {
        "prompt": prompt_key,
        "accuracy": correct / len(eval_set),
        "correct": correct,
        "n": len(eval_set),
        "tokens_in": sum(u.prompt_tokens for _, _, u, _ in results if u),
        "tokens_out": sum(u.completion_tokens for _, _, u, _ in results if u),
        "results": results,
    }


def show(res):
    """One line per evaluation, so runs are easy to compare."""
    print(f"{res['prompt']:<30} acc {res['accuracy']:.2f} "
          f"({res['correct']}/{res['n']})   "
          f"in {res['tokens_in']:>5}  out {res['tokens_out']:>4}")
