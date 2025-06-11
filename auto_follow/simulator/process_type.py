from enum import Enum


class ProcessType(str, Enum):
    """Supported process names."""
    SPHINX = "Sphinx"
    FIRMWARE = "Firmware"
    SCRIPT = "Script"

    def __str__(self) -> str:
        return self.value
