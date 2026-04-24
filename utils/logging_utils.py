"""
utils/logging_utils.py
=======================
Centralised logging configuration and YAML config loader.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logging(
    log_dir: str = 'logs',
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """
    Configure root logger with console + optional file handler.
    Returns the root logger.
    """
    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler
    if log_to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = Path(log_dir) / f'pipeline_{ts}.log'
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)
        root.info("Logging to file: %s", log_file)

    return root


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> dict:
    """
    Load YAML config and optionally apply key=value overrides.

    overrides: dict of dot-separated key paths, e.g.
      {'generator.epochs': 200, 'data.max_len': 70}

    Compatible with the config.yaml structure:
      data / features / model / generator / generation / filters / diversity / output
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required: pip install pyyaml")

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Config file is empty: {config_path}")

    if overrides:
        for key_path, value in overrides.items():
            parts = key_path.split('.')
            node = cfg
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value

    return cfg


def versioned_output_dir(base_dir: str) -> str:
    """
    Create and return a timestamped subdirectory under base_dir.
    E.g. outputs/ -> outputs/20250423_143021/
    """
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = Path(base_dir) / ts
    out.mkdir(parents=True, exist_ok=True)
    return str(out)
