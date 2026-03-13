"""threetears.core — three-tier caching framework.

Public API re-exports for convenient top-level imports.
"""

__version__ = "0.1.0"

from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import CoreConfig, DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import ConcurrentModificationError, DataLayerUnavailableError
from threetears.core.serialization import deserialize_from_json, serialize_to_json

__all__ = [
    "BaseCollection",
    "BaseEntity",
    "CollectionRegistry",
    "ConcurrentModificationError",
    "CoreConfig",
    "DataLayerUnavailableError",
    "DefaultCoreConfig",
    "deserialize_from_json",
    "serialize_to_json",
]
