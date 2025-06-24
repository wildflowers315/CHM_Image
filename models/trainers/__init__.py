"""Model-specific trainers and architectures."""

# Import only working modules
try:
    from .shift_aware_trainer import ShiftAwareTrainer, ShiftAwareUNet
    SHIFT_AWARE_AVAILABLE = True
except ImportError:
    SHIFT_AWARE_AVAILABLE = False

__all__ = []

if SHIFT_AWARE_AVAILABLE:
    __all__.extend(['ShiftAwareTrainer', 'ShiftAwareUNet'])