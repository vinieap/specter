"""
Logging configuration system for the clustering framework.

This module provides a configuration system for managing logging settings across
different components of the framework.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .exceptions import ConfigurationError


class LogLevel(Enum):
    """Available log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(Enum):
    """Available log formats."""

    JSON = "json"
    TEXT = "text"


@dataclass
class LogHandlerConfig:
    """Configuration for a log handler."""

    enabled: bool = True
    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.JSON
    # File handler specific
    filename: Optional[str] = None
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    # Console handler specific
    use_colors: bool = True
    # Additional handler options
    additional_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentConfig:
    """Logging configuration for a specific component."""

    name: str
    level: LogLevel = LogLevel.INFO
    handlers: List[LogHandlerConfig] = field(default_factory=list)
    propagate: bool = False
    extra_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """Global logging configuration."""

    root_level: LogLevel = LogLevel.INFO
    log_dir: Optional[Union[str, Path]] = None
    default_format: LogFormat = LogFormat.JSON
    components: Dict[str, ComponentConfig] = field(default_factory=dict)
    handlers: Dict[str, LogHandlerConfig] = field(default_factory=dict)
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate the logging configuration.

        Raises:
            ConfigurationError: If configuration is invalid
        """
        if self.log_dir:
            self.log_dir = Path(self.log_dir)
            if not self.log_dir.exists():
                try:
                    self.log_dir.mkdir(parents=True)
                except Exception as e:
                    raise ConfigurationError(f"Failed to create log directory: {e}")

        for name, component in self.components.items():
            if not name:
                raise ConfigurationError("Component name cannot be empty")

            for handler in component.handlers:
                if handler.filename and not self.log_dir:
                    raise ConfigurationError(
                        f"Log directory not set for component {name}"
                    )

    def get_component_config(
        self, name: str, create_if_missing: bool = True
    ) -> ComponentConfig:
        """
        Get configuration for a component.

        Args:
            name: Component name
            create_if_missing: Create default config if missing

        Returns:
            Component configuration
        """
        if name not in self.components and create_if_missing:
            self.components[name] = ComponentConfig(
                name=name,
                handlers=[
                    LogHandlerConfig(  # Console handler
                        format=self.default_format
                    ),
                    LogHandlerConfig(  # File handler
                        filename=f"{name}.log", format=self.default_format
                    )
                    if self.log_dir
                    else None,
                ],
            )
        return self.components[name]

    def update_component(
        self,
        name: str,
        level: Optional[LogLevel] = None,
        handlers: Optional[List[LogHandlerConfig]] = None,
        propagate: Optional[bool] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update configuration for a component.

        Args:
            name: Component name
            level: New log level
            handlers: New handlers
            propagate: New propagate setting
            extra_fields: New extra fields
        """
        config = self.get_component_config(name)

        if level is not None:
            config.level = level
        if handlers is not None:
            config.handlers = handlers
        if propagate is not None:
            config.propagate = propagate
        if extra_fields is not None:
            config.extra_fields.update(extra_fields)

        self.validate()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of config
        """
        return {
            "root_level": self.root_level.value,
            "log_dir": str(self.log_dir) if self.log_dir else None,
            "default_format": self.default_format.value,
            "components": {
                name: {
                    "level": comp.level.value,
                    "propagate": comp.propagate,
                    "extra_fields": comp.extra_fields,
                    "handlers": [
                        {
                            "enabled": h.enabled,
                            "level": h.level.value,
                            "format": h.format.value,
                            "filename": h.filename,
                            "max_bytes": h.max_bytes,
                            "backup_count": h.backup_count,
                            "use_colors": h.use_colors,
                            "additional_options": h.additional_options,
                        }
                        for h in comp.handlers
                        if h
                    ],
                }
                for name, comp in self.components.items()
            },
            "handlers": {
                name: {
                    "enabled": h.enabled,
                    "level": h.level.value,
                    "format": h.format.value,
                    "filename": h.filename,
                    "max_bytes": h.max_bytes,
                    "backup_count": h.backup_count,
                    "use_colors": h.use_colors,
                    "additional_options": h.additional_options,
                }
                for name, h in self.handlers.items()
            },
            "extra_fields": self.extra_fields,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoggingConfig":
        """
        Create configuration from dictionary.

        Args:
            data: Dictionary representation of config

        Returns:
            LoggingConfig instance
        """
        components = {}
        for name, comp_data in data.get("components", {}).items():
            handlers = [
                LogHandlerConfig(
                    enabled=h.get("enabled", True),
                    level=LogLevel(h.get("level", "INFO")),
                    format=LogFormat(h.get("format", "json")),
                    filename=h.get("filename"),
                    max_bytes=h.get("max_bytes", 10 * 1024 * 1024),
                    backup_count=h.get("backup_count", 5),
                    use_colors=h.get("use_colors", True),
                    additional_options=h.get("additional_options", {}),
                )
                for h in comp_data.get("handlers", [])
            ]

            components[name] = ComponentConfig(
                name=name,
                level=LogLevel(comp_data.get("level", "INFO")),
                handlers=handlers,
                propagate=comp_data.get("propagate", False),
                extra_fields=comp_data.get("extra_fields", {}),
            )

        handlers = {}
        for name, h_data in data.get("handlers", {}).items():
            handlers[name] = LogHandlerConfig(
                enabled=h_data.get("enabled", True),
                level=LogLevel(h_data.get("level", "INFO")),
                format=LogFormat(h_data.get("format", "json")),
                filename=h_data.get("filename"),
                max_bytes=h_data.get("max_bytes", 10 * 1024 * 1024),
                backup_count=h_data.get("backup_count", 5),
                use_colors=h_data.get("use_colors", True),
                additional_options=h_data.get("additional_options", {}),
            )

        return cls(
            root_level=LogLevel(data.get("root_level", "INFO")),
            log_dir=data.get("log_dir"),
            default_format=LogFormat(data.get("default_format", "json")),
            components=components,
            handlers=handlers,
            extra_fields=data.get("extra_fields", {}),
        )


# Global logging configuration instance
_default_config: Optional[LoggingConfig] = None


def get_logging_config(**kwargs: Any) -> LoggingConfig:
    """
    Get or create the global logging configuration instance.

    Args:
        **kwargs: Arguments passed to LoggingConfig constructor

    Returns:
        The global LoggingConfig instance
    """
    global _default_config

    if _default_config is None:
        _default_config = LoggingConfig(**kwargs)

    return _default_config
