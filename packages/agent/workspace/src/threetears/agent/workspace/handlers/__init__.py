"""format handlers for agent workspace.

importing this package (or any submodule) triggers the side-effect
registration of each concrete handler with the core
:mod:`threetears.core.serialization` registry.
"""

from threetears.agent.workspace.handlers import yaml_handler as yaml_handler
from threetears.agent.workspace.handlers.yaml_handler import YamlHandler

__all__ = ["YamlHandler", "yaml_handler"]
