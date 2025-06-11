import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from auto_follow.utils.path_manager import Paths


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the Sphinx simulator."""
    firmware_command: str
    sphinx_command: str | Path

    timeout_seconds: int = 30
    sphinx_init_wait: int = 5
    firmware_init_wait: int = 15
    script_timeout_seconds: int = 300  # 5 minutes

    def __post_init__(self):
        if not isinstance(self.sphinx_command, str | Path):
            raise ValueError("sphinx_command must be a string or Path")
        if not isinstance(self.firmware_command, str):
            raise ValueError("firmware_command must be a string")

        for attr in ('timeout_seconds', 'sphinx_init_wait', 'firmware_init_wait', 'script_timeout_seconds'):
            if getattr(self, attr) < 0:
                raise ValueError(f"{attr} must be positive")

    @property
    def sphinx_cmd(self) -> str:
        """Get the complete sphinx command with arguments."""
        return str(self.sphinx_command)


@dataclass(frozen=True)
class ScriptConfig:
    script_path: str | Path
    script_args: list[str] | None = None

    def __post_init__(self):
        if not Path(self.script_path).exists():
            raise FileNotFoundError(f"Script path does not exist: {self.script_path}")
        if self.script_args is not None:
            if not isinstance(self.script_args, list):
                raise ValueError("script_args must be a list")
            if len(self.script_args) < 1:
                raise ValueError("script_args must contain at least one argument")


@dataclass
class PathConfig:
    sphinx_command: str | Path
    firmware_command: str
    script_path: str | Path
    script_args: list[str] | None = None

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> 'PathConfig':
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        paths = config['paths']
        sphinx = paths['sphinx']
        firmware = sphinx['firmware']
        script = paths['scripts']

        pose = firmware.get('pose')
        firmware_command = f'{firmware["base_command"]} "{firmware["drone_path"]}'
        if pose is not None:
            firmware_command += f'::pose={pose}'
        firmware_command += f'::firmware={firmware["firmware_url"]}"'

        quality = sphinx.get('quality')
        config_file = sphinx.get('config_file') or None
        sphinx_cmd = [str(sphinx["command"]), f'-level={sphinx["level"]}']
        if quality is not None:
            sphinx_cmd.append(f'-quality={quality}')
        if config_file is not None:
            config_file_path = cls._prepare_temp_config(Paths.BASE_DIR / config_file)
            sphinx_cmd.append(f'-config-file={config_file_path}')
        sphinx_cmd_str = ' '.join(sphinx_cmd)

        script_path = Paths.BASE_DIR / script["controller"]
        cls._validate_path(script_path)

        script_save_dir_name = script.get("save_dir_name")
        script_args = None
        if script_save_dir_name is not None:
            script_args = [
                f"--log_path={Paths.OUTPUT_DIR / script_save_dir_name / 'logs'}",
                f"--results_path={Paths.OUTPUT_DIR / script_save_dir_name / 'results'}",
                f"--parquet_logs_path={Paths.OUTPUT_DIR / script_save_dir_name / 'parquet-logs'}",
            ]

        return cls(
            sphinx_command=sphinx_cmd_str,
            firmware_command=firmware_command,
            script_path=script_path,
            script_args=script_args,
        )

    @staticmethod
    def _prepare_temp_config(original_config_path: Path) -> Path:
        PathConfig._validate_path(original_config_path)

        with open(original_config_path, "r") as f:
            content = f.read()

        bunker_dir = original_config_path.parent.parent.resolve()
        models_dir = bunker_dir / "models"
        updated_content = re.sub(r"\$\{MODELS_DIR\}", str(models_dir), content)

        temp_config_file = Path(tempfile.mkstemp(suffix=".yaml", prefix="sphinx_config_")[1])
        with open(temp_config_file, "w") as f:
            f.write(updated_content)

        return temp_config_file

    @staticmethod
    def _validate_path(path_: Path):
        if not path_.exists():
            raise FileNotFoundError(f"File does not exist: {path_}")


if __name__ == '__main__':
    config = PathConfig.from_yaml(
        Paths.SIMULATOR_CONFIG_DIR / "bunker-online-4k-config-test-front-small-offset-left.yaml"
    )

    print(config)
    print(f"Sphinx Command: {config.sphinx_command}")
    print(f"Firmware Command: {config.firmware_command}")
    print(f"Script Command: {config.script_path}")
    print(f"Script Args: {config.script_args}")
