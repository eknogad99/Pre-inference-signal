result = is_locked_live(declared_emb, actual_emb, declared_bounds, actual_metrics)
if result.is_oriented:
    # Proceed to Engage!
else:
    # Reject / Refine loop (Manage function)
