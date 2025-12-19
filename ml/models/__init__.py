"""
ML Models Package
=================
Modular, plugin-based architecture for custom ML models.

Users can add new models by:
1. Creating a Python file in this folder
2. Defining a class that inherits from BaseModel
3. The system auto-discovers and registers it
"""

import os
import logging
import importlib.util
from typing import Dict, Type, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ============================================================
#  BASE MODEL INTERFACE
# ============================================================
class BaseModel(ABC):
    """
    Abstract base class for all ML models.
    
    Users must inherit from this and implement:
    - forward(x: np.ndarray) -> np.ndarray
    - predict(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
    - save(filepath: str)
    - load(filepath: str)
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize model.
        
        Args:
            name: Model identifier (used in CLI)
            **kwargs: Model-specific hyperparameters
        """
        self.name = name
        self.hyperparameters = kwargs
        logger.info(f"Initialized {name}")
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass. Input shape: (batch_size, 256).
        Return: (batch_size, num_classes) logits or probabilities.
        """
        pass
    
    @abstractmethod
    def predict(self, x):
        """
        Predict on input. Returns (classes, probabilities).
        """
        pass
    
    def predict_single(self, x):
        """
        Predict on a single sample (optional, has default implementation).
        
        Args:
            x: Single input (1, 256) or (256,)
        
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        classes, probs = self.predict(x)
        
        if classes.ndim > 0:
            classes = classes[0]
        if isinstance(probs, (list, tuple)):
            probs = probs[0]
        else:
            probs = probs[0] if probs.ndim > 1 else probs
        
        confidence = float(probs[int(classes)]) if hasattr(probs, '__getitem__') else float(probs)
        return int(classes), confidence
    
    @abstractmethod
    def save(self, filepath: str):
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """Load model from disk."""
        pass


# ============================================================
#  MODEL REGISTRY (AUTO-DISCOVERY)
# ============================================================
class ModelRegistry:
    """Auto-discovers and manages all available models."""
    
    _registry: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]) -> None:
        """Register a model class."""
        cls._registry[name] = model_class
        logger.info(f"Registered model: {name}")
    
    @classmethod
    def get(cls, name: str, **kwargs) -> BaseModel:
        """Instantiate a registered model."""
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Model '{name}' not found. Available: {available}"
            )
        return cls._registry[name](name=name, **kwargs)
    
    @classmethod
    def list(cls) -> Dict[str, Type[BaseModel]]:
        """List all registered models."""
        return dict(cls._registry)
    
    @classmethod
    def load_from_folder(cls, folder_path: str) -> None:
        """
        Auto-discover and load all models from a folder.
        
        Looks for Python files with classes inheriting from BaseModel.
        """
        if not os.path.isdir(folder_path):
            logger.warning(f"Models folder not found: {folder_path}")
            return
        
        for filename in os.listdir(folder_path):
            if filename.startswith("_") or not filename.endswith(".py"):
                continue
            
            filepath = os.path.join(folder_path, filename)
            module_name = filename[:-3]  # Remove .py
            
            try:
                # Load module
                spec = importlib.util.spec_from_file_location(
                    f"ml.models.{module_name}",
                    filepath
                )
                if spec is None or spec.loader is None:
                    logger.warning(f"Could not create spec for {filename}")
                    continue
                
                module = importlib.util.module_from_spec(spec)
                if module is None:
                    logger.warning(f"Could not create module from spec: {filename}")
                    continue
                
                # Add to sys.modules BEFORE exec_module to avoid NoneType
                import sys
                sys.modules[spec.name] = module
                    
                spec.loader.exec_module(module)
                
                # Find BaseModel subclasses (skip nn.Module classes)
                for attr_name in dir(module):
                    try:
                        attr = getattr(module, attr_name, None)
                        if attr is None:
                            continue
                            
                        # Only register if it's a proper BaseModel subclass
                        if not isinstance(attr, type):
                            continue
                            
                        # Check if it's a BaseModel subclass (not nn.Module)
                        if not issubclass(attr, BaseModel):
                            continue
                            
                        if attr is BaseModel:
                            continue
                            
                        # Valid BaseModel subclass found
                        model_name = _camel_to_snake(attr_name)
                        cls.register(model_name, attr)
                        logger.debug(f"Registered model: {model_name} from {filename}")
                        
                    except (TypeError, AttributeError, ValueError) as e:
                        # TypeError: issubclass() arg 1 must be a class
                        # AttributeError: problematic getattr
                        # ValueError: certain metaclass issues
                        continue
            
            except Exception as e:
                logger.debug(f"Module loading note for {filename}: {type(e).__name__}: {e}")
                continue


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


# ============================================================
#  AUTO-LOAD PRODUCTION MODELS (rm0, rm1, cm0, cm1)
# ============================================================
try:
    from .rm0 import RM0_RealAutoEncoder
    ModelRegistry.register("rm0", RM0_RealAutoEncoder)
except Exception as e:
    logger.debug(f"Could not load rm0: {e}")

try:
    from .rm1 import RM1_RealAutoEncoder
    ModelRegistry.register("rm1", RM1_RealAutoEncoder)
except Exception as e:
    logger.debug(f"Could not load rm1: {e}")

try:
    from .cm0 import CM0_ComplexAutoEncoder
    ModelRegistry.register("cm0", CM0_ComplexAutoEncoder)
except Exception as e:
    logger.debug(f"Could not load cm0: {e}")

try:
    from .cm1 import CM1_ComplexAutoEncoder
    ModelRegistry.register("cm1", CM1_ComplexAutoEncoder)
except Exception as e:
    logger.debug(f"Could not load cm1: {e}")

