import pytest
from bond.config import BondSettings

def test_config_defaults():
    """Test that configuration has sensible defaults"""
    settings = BondSettings()
    assert settings.topk_final > 0
    assert settings.rrf_k > 0

def test_config_environment_override():
    """Test that environment variables can override defaults"""
    import os
    os.environ["BOND_TOPK_FINAL"] = "7"
    settings = BondSettings()
    assert settings.topk_final == 7
    # Clean up
    del os.environ["BOND_TOPK_FINAL"]
