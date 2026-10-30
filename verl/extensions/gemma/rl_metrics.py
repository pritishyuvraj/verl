"""Episode-level GRPO diagnostics; variable turn counts must not weight rewards."""
from collections import defaultdict
import math


def group_metrics(keys, scores, *, n, padding=None):
    latest = {}
    for i, (key, reward) in enumerate(zip(keys, scores, strict=True)):
        if padding is not None and padding[i]:
            continue
        uid, session, turn = key.rsplit("_", 2)
        pair = (uid, int(session))
        if pair not in latest or int(turn) > latest[pair][0]:
            latest[pair] = (int(turn), float(reward))
    groups = defaultdict(list)
    for (uid, session), (_, reward) in latest.items():
        if not math.isfinite(reward) or reward not in (0., 1.):
            raise ValueError("SWE executable reward must be binary")
        groups[uid].append(reward)
    if not groups or any(len(v) != n for v in groups.values()):
        raise ValueError("GRPO metrics require complete n-rollout groups")
    all_pass = sum(all(r == 1 for r in v) for v in groups.values())
    all_zero = sum(all(r == 0 for r in v) for v in groups.values())
    count = len(groups)
    return {"reward/mean":sum(sum(v)for v in groups.values())/(count*n),
            "groups/count":count,"groups/all_pass_count":all_pass,"groups/all_zero_count":all_zero,
            "groups/mixed_count":count-all_pass-all_zero,"groups/all_pass_pct":100*all_pass/count,
            "groups/all_zero_pct":100*all_zero/count,"groups/zero_advantage_pct":100*(all_pass+all_zero)/count}
