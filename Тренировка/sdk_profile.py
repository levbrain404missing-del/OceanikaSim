"""Прочитать профиль установленного SDK. Подключение к оборудованию не требуется."""
from importlib.metadata import PackageNotFoundError, distribution
import json
import platform


def read_profile(config_path, sdk_version):
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    cameras = config.get("cameras") or {}
    return {
        "python": platform.python_version(),
        "sdk_version": sdk_version,
        "board_profile": config.get("board_name"),
        "camera_driver": cameras.get("driver"),
        "camera_types": sorted(key for key in cameras if key not in ("driver","recorder")),
        "servo_configured": bool(config.get("servo")),
    }


if __name__ == "__main__":
    try:
        sdk = distribution("pioneer_sdk2")
        report = read_profile(sdk.locate_file("pioneer_sdk2/board_config.json"), sdk.version)
    except PackageNotFoundError:
        raise SystemExit("В этом окружении отсутствует пакет pioneer_sdk2.")
    except (OSError, ValueError) as error:
        raise SystemExit(f"Ошибка чтения профиля SDK: {error}") from error
    print(json.dumps(report, indent=2, ensure_ascii=False))
