# src/pre_inference_signal/pipeline/lock_test.py

from typing import Tuple, Optional
from pydantic import BaseModel, Field
import numpy as np  # for vector ops; swap for torch if embeddings are tensor-based

class LockTestConfig(BaseModel):
    """SOL-enforced configuration bounds."""
    fidelity_threshold: float = Field(0.85, ge=0.0, le=1.0, description="Min convergence score (e.g. normalized cosine/concurrence)")
    max_displacement: float = Field(0.15, ge=0.0, description="Max allowable deviation (1 - fidelity or drift norm)")
    enable_audit_log: bool = True

class ConvergenceResult(BaseModel):
    """SOL diagnostic output."""
    is_oriented: bool
    convergence_score: float
    displacement: float
    details: dict  # e.g., {'fidelity': ..., 'drift_norm': ..., 'flags': [...]}
    message: str

def compute_convergence(
    declared_embedding: np.ndarray,      # Vector rep of declared constraints (e.g., policy embedding or ideal context)
    actual_embedding: np.ndarray,         # Vector rep of runtime state (retrieved + compacted context)
    declared_bounds: dict,                # e.g., {'max_tokens': 8000, 'risk_tolerance': 0.1}
    actual_metrics: dict                  # e.g., {'token_count': ..., 'risk_score': ..., 'fidelity': ...}
) -> Tuple[float, float]:
    """Core convergence metric (customizable)."""
    # Example: cosine similarity as base convergence proxy
    cos_sim = np.dot(declared_embedding, actual_embedding) / (
        np.linalg.norm(declared_embedding) * np.linalg.norm(actual_embedding) + 1e-8
    )
    
    # Displacement: scalar deviation (e.g., normalized diff in key metrics)
    token_displacement = abs(actual_metrics.get('token_count', 0) - declared_bounds.get('max_tokens', float('inf'))) / declared_bounds.get('max_tokens', 1.0)
    risk_displacement = abs(actual_metrics.get('risk_score', 0) - declared_bounds.get('risk_tolerance', 0.0))
    displacement = max(token_displacement, risk_displacement, 1 - cos_sim)  # worst-case
    
    return cos_sim, displacement  # or use concurrence/entropy/other entanglement proxy

def is_locked_live(
    declared_embedding: np.ndarray,
    actual_embedding: np.ndarray,
    declared_bounds: dict,
    actual_metrics: dict,
    config: LockTestConfig = LockTestConfig()
) -> ConvergenceResult:
    """
    Implements SOL: System is oriented (locked/live) iff declared constraints converge
    with actual dependencies within bounded displacement.
    """
    convergence_score, displacement = compute_convergence(
        declared_embedding, actual_embedding, declared_bounds, actual_metrics
    )
    
    is_oriented = (
        convergence_score >= config.fidelity_threshold
        and displacement <= config.max_displacement
    )
    
    details = {
        "convergence_score": convergence_score,
        "displacement": displacement,
        "thresholds": {"fidelity": config.fidelity_threshold, "max_disp": config.max_displacement},
        "metrics": actual_metrics,
        "flags": [] if is_oriented else ["displacement_exceeded" if displacement > config.max_displacement else "low_convergence"]
    }
    
    message = (
        "System oriented (locked & live) per SOL."
        if is_oriented
        else f"Mis-oriented: displacement {displacement:.3f} exceeds bound {config.max_displacement} or convergence {convergence_score:.3f} < {config.fidelity_threshold}"
    )
    
    result = ConvergenceResult(
        is_oriented=is_oriented,
        convergence_score=convergence_score,
        displacement=displacement,
        details=details,
        message=message
    )
    
    if config.enable_audit_log:
        # Hook to audit.py for NIST traceability
        from ..governance.audit import log_lock_test
        log_lock_test(result)
    
    return result
