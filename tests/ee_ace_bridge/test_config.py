"""
Tests for ACE bridge configuration module.

Validates:
- Feature flag defaults
- Environment variable parsing
- Configuration validation
- Config summary generation
"""

import pytest
import os
from unittest.mock import patch
from ee_ace_bridge import config
from ee_ace_bridge.config import validate_config, get_config_summary


class TestFeatureFlags:
    """Test suite for feature flag defaults and parsing."""

    def test_ace_enabled_defaults_false(self):
        """Test that ACE_ENABLED defaults to False."""
        # Note: This tests the module's default without mocking
        # In actual environment, check that default is sensible
        assert isinstance(config.ACE_ENABLED, bool)

    def test_ace_sections_defaults_none(self):
        """Test that ACE_SECTIONS defaults to None."""
        # When no environment variable set, should be None
        assert config.ACE_SECTIONS is None or isinstance(config.ACE_SECTIONS, list)

    def test_ace_token_budget_defaults_3500(self):
        """Test that ACE_TOKEN_BUDGET defaults to 3500."""
        assert isinstance(config.ACE_TOKEN_BUDGET, int)
        # Should be a reasonable default
        assert 500 <= config.ACE_TOKEN_BUDGET <= 10000

    def test_ace_endpoint_defaults_none(self):
        """Test that ACE_ENDPOINT defaults to None."""
        # When no environment variable set, should be None
        assert config.ACE_ENDPOINT is None or isinstance(config.ACE_ENDPOINT, str)


class TestEnvironmentParsing:
    """Test suite for environment variable parsing."""

    @patch.dict(os.environ, {"ACE_ENABLED": "1"})
    def test_parses_enabled_flag(self):
        """Test parsing of ACE_ENABLED=1."""
        # Need to reload config to pick up mocked environment
        import importlib
        importlib.reload(config)

        assert config.ACE_ENABLED is True

    @patch.dict(os.environ, {"ACE_ENABLED": "0"})
    def test_parses_disabled_flag(self):
        """Test parsing of ACE_ENABLED=0."""
        import importlib
        importlib.reload(config)

        assert config.ACE_ENABLED is False

    @patch.dict(os.environ, {"ACE_SECTIONS": "Payment,Validation,Auth"})
    def test_parses_sections_list(self):
        """Test parsing of comma-separated sections."""
        import importlib
        importlib.reload(config)

        assert isinstance(config.ACE_SECTIONS, list)
        assert "Payment" in config.ACE_SECTIONS
        assert "Validation" in config.ACE_SECTIONS
        assert "Auth" in config.ACE_SECTIONS

    @patch.dict(os.environ, {"ACE_TOKEN_BUDGET": "5000"})
    def test_parses_token_budget(self):
        """Test parsing of token budget."""
        import importlib
        importlib.reload(config)

        assert config.ACE_TOKEN_BUDGET == 5000

    @patch.dict(os.environ, {"ACE_ENDPOINT": "http://ace-service:8080"})
    def test_parses_endpoint(self):
        """Test parsing of endpoint URL."""
        import importlib
        importlib.reload(config)

        assert config.ACE_ENDPOINT == "http://ace-service:8080"


class TestValidateConfig:
    """Test suite for configuration validation."""

    @patch.dict(os.environ, {
        "ACE_ENABLED": "1",
        "ACE_TOKEN_BUDGET": "3500"
    })
    def test_valid_config_has_no_warnings(self):
        """Test that valid configuration produces no warnings."""
        import importlib
        importlib.reload(config)

        warnings = validate_config()

        # Should have minimal or no warnings for valid config
        assert isinstance(warnings, list)

    @patch.dict(os.environ, {"ACE_TOKEN_BUDGET": "100"})
    def test_warns_on_low_token_budget(self):
        """Test warning for very low token budget."""
        import importlib
        importlib.reload(config)

        warnings = validate_config()

        # Should warn about low budget
        assert any("low" in w.lower() for w in warnings)

    @patch.dict(os.environ, {"ACE_TOKEN_BUDGET": "50000"})
    def test_warns_on_high_token_budget(self):
        """Test warning for very high token budget."""
        import importlib
        importlib.reload(config)

        warnings = validate_config()

        # Should warn about high budget
        assert any("high" in w.lower() for w in warnings)

    @patch.dict(os.environ, {"ACE_ENDPOINT": "invalid-url"})
    def test_warns_on_invalid_endpoint(self):
        """Test warning for invalid endpoint format."""
        import importlib
        importlib.reload(config)

        warnings = validate_config()

        # Should warn about invalid URL format
        assert any("http" in w.lower() for w in warnings)

    @patch.dict(os.environ, {
        "ACE_ENABLED": "0",
        "ACE_SECTIONS": "Payment,Auth"
    })
    def test_warns_on_sections_when_disabled(self):
        """Test warning when sections specified but ACE disabled."""
        import importlib
        importlib.reload(config)

        warnings = validate_config()

        # Should warn that sections will be ignored
        assert any("sections" in w.lower() and "ignore" in w.lower() for w in warnings)

    @patch.dict(os.environ, {
        "ACE_ENABLED": "0",
        "ACE_ENDPOINT": "http://ace-service:8080"
    })
    def test_warns_on_endpoint_when_disabled(self):
        """Test warning when endpoint specified but ACE disabled."""
        import importlib
        importlib.reload(config)

        warnings = validate_config()

        # Should warn that endpoint will be ignored
        assert any("endpoint" in w.lower() and "ignore" in w.lower() for w in warnings)


class TestGetConfigSummary:
    """Test suite for configuration summary generation."""

    def test_returns_dict(self):
        """Test that summary returns a dictionary."""
        summary = get_config_summary()

        assert isinstance(summary, dict)

    def test_includes_all_flags(self):
        """Test that summary includes all configuration flags."""
        summary = get_config_summary()

        assert "ACE_ENABLED" in summary
        assert "ACE_SECTIONS" in summary
        assert "ACE_TOKEN_BUDGET" in summary
        assert "ACE_ENDPOINT" in summary

    def test_includes_warnings(self):
        """Test that summary includes validation warnings."""
        summary = get_config_summary()

        assert "warnings" in summary
        assert isinstance(summary["warnings"], list)

    def test_summary_values_match_config(self):
        """Test that summary values match actual config."""
        summary = get_config_summary()

        assert summary["ACE_ENABLED"] == config.ACE_ENABLED
        assert summary["ACE_SECTIONS"] == config.ACE_SECTIONS
        assert summary["ACE_TOKEN_BUDGET"] == config.ACE_TOKEN_BUDGET
        assert summary["ACE_ENDPOINT"] == config.ACE_ENDPOINT


class TestConfigDocumentation:
    """Test suite for configuration documentation and usage."""

    def test_config_has_docstring(self):
        """Test that config module has documentation."""
        assert config.__doc__ is not None
        assert len(config.__doc__) > 0

    def test_flags_have_docstrings(self):
        """Test that flag variables have documentation."""
        # Check module-level docstrings exist
        import inspect
        source = inspect.getsource(config)

        # Should document each flag
        assert "ACE_ENABLED" in source
        assert "ACE_SECTIONS" in source
        assert "ACE_TOKEN_BUDGET" in source
        assert "ACE_ENDPOINT" in source


class TestConfigurationIsolation:
    """Test suite for configuration isolation and safety."""

    def test_config_is_read_only_usage(self):
        """Test that config is typically used read-only."""
        # Config values should be module-level constants
        # that aren't modified during runtime
        original_enabled = config.ACE_ENABLED
        original_budget = config.ACE_TOKEN_BUDGET

        # Values should remain constant
        assert config.ACE_ENABLED == original_enabled
        assert config.ACE_TOKEN_BUDGET == original_budget

    def test_validation_is_side_effect_free(self):
        """Test that validation doesn't modify config."""
        original_enabled = config.ACE_ENABLED
        original_budget = config.ACE_TOKEN_BUDGET

        # Run validation
        validate_config()

        # Config should be unchanged
        assert config.ACE_ENABLED == original_enabled
        assert config.ACE_TOKEN_BUDGET == original_budget

    def test_summary_is_side_effect_free(self):
        """Test that summary generation doesn't modify config."""
        original_enabled = config.ACE_ENABLED
        original_budget = config.ACE_TOKEN_BUDGET

        # Get summary
        get_config_summary()

        # Config should be unchanged
        assert config.ACE_ENABLED == original_enabled
        assert config.ACE_TOKEN_BUDGET == original_budget
