"""
Configuration management for the hybrid AI agent.
Loads and validates settings from config.yaml.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    backbone: str = "resnet18"
    checkpoint: str = "model.pth"
    num_classes: int = 1


@dataclass
class PreprocessingConfig:
    size: int = 224
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class LLMConfig:
    provider: str = "gemini"
    model: str = "gemini-1.5-flash"
    max_retries: int = 3
    timeout: int = 30


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    threshold: float = 0.5
    margin: float = 0.1
    alpha: float = 0.7
    seed: int = 42
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "checkpoints" / "config.yaml"
        
        config = cls()
        
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            
            # Load model config
            if "model" in data:
                config.model = ModelConfig(**data["model"])
            
            # Load preprocessing config
            if "preprocessing" in data:
                config.preprocessing = PreprocessingConfig(**data["preprocessing"])
            
            # Load LLM config
            if "llm" in data:
                config.llm = LLMConfig(**data["llm"])
            
            # Load scalar values
            config.threshold = data.get("threshold", config.threshold)
            config.margin = data.get("margin", config.margin)
            config.alpha = data.get("alpha", config.alpha)
            config.seed = data.get("seed", config.seed)
        
        return config
    
    def save(self, config_path: Optional[str] = None) -> None:
        """Save configuration to YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "checkpoints" / "config.yaml"
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        data = {
            "model": {
                "backbone": self.model.backbone,
                "checkpoint": self.model.checkpoint,
                "num_classes": self.model.num_classes,
            },
            "preprocessing": {
                "size": self.preprocessing.size,
                "mean": self.preprocessing.mean,
                "std": self.preprocessing.std,
            },
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "max_retries": self.llm.max_retries,
                "timeout": self.llm.timeout,
            },
            "threshold": self.threshold,
            "margin": self.margin,
            "alpha": self.alpha,
            "seed": self.seed,
        }
        
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def set_config(config: Config) -> None:
    """Set global config instance."""
    global _config
    _config = config
