import os

from dotenv import load_dotenv

from auto_follow.utils.path_manager import Paths


def get_auth_from_vault():
    vault_path = Paths.BASE_DIR / ".env"

    if not vault_path.exists():
        raise FileNotFoundError(f"Cannot read AUTH from .env file at path: {vault_path}")

    load_dotenv(dotenv_path=vault_path)
    return os.getenv("AUTH")


if __name__ == "__main__":
    auth = get_auth_from_vault()
    print(auth)
