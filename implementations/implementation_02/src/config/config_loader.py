"""Configuration loader for the project."""
import json
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Loads and manages project configuration from llm.json."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to llm.json. If None, searches for it in parent directories.
        """
        if config_path is None:
            # Find llm.json relative to this file
            current_file = Path(__file__)
            # Go up: src/config/config_loader.py -> src/config -> src -> implementation_02
            impl_dir = current_file.parent.parent.parent
            config_path = impl_dir / "llm.json"
        
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config: Dict[str, Any] = json.load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def get_project_config(self) -> Dict[str, Any]:
        """Get project configuration."""
        return self.config.get('project', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.config.get('data_config', {})
    
    def get_portfolio_config(self) -> Dict[str, Any]:
        """Get portfolio generation configuration."""
        return self.config.get('portfolio_generation', {})
    
    def get_backtesting_config(self) -> Dict[str, Any]:
        """Get backtesting configuration."""
        return self.config.get('backtesting_and_splits', {})
    
    def get_blocks_config(self) -> Dict[str, Any]:
        """Get blocks configuration (risk and portfolio methods)."""
        return self.config.get('blocks', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.config.get('evaluation_and_comparison', {})
    
    def get_experiment_config(self) -> Dict[str, Any]:
        """Get experiment tracking configuration."""
        return self.config.get('experiment_tracking', {})
    
    def get_dataset_path(self) -> Path:
        """Get the dataset path relative to project root."""
        current_file = Path(__file__)
        # Go up to project root: src/config -> src -> implementation_02 -> implementations -> Code-Space
        project_root = current_file.parent.parent.parent.parent.parent
        dataset_path = project_root / "dataset"
        return dataset_path


# Global config instance
_config_instance: Optional[ConfigLoader] = None


def get_config(config_path: Optional[str] = None) -> ConfigLoader:
    """Get or create the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader(config_path)
    return _config_instance

