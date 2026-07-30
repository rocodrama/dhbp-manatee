from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from pathlib import Path


def _print_path_status(label: str, path: Path) -> None:
    print(f"{label}: {path}")
    print(f"  exists={path.exists()} is_dir={path.is_dir()} is_file={path.is_file()}")


def main() -> None:
    from main.fixed import data_adapter as adapter

    print("=== Manatee import diagnosis ===")
    print(f"python: {sys.executable}")
    print(f"cwd: {Path.cwd()}")
    print(f"MANATEE_API_ROOT: {os.getenv('MANATEE_API_ROOT')}")
    print(f"MANATEE_DATA_ROOT: {os.getenv('MANATEE_DATA_ROOT')}")
    print(f"resolved API_ROOT: {adapter.API_ROOT}")
    print(f"resolved DATA_ROOT: {adapter.DATA_ROOT}")
    _print_path_status("api.py", adapter.API_ROOT / "api.py")
    _print_path_status("app_state.py", adapter.API_ROOT / "app_state.py")
    _print_path_status("Manatee", adapter.DATA_ROOT / "Manatee")
    _print_path_status("Manatee/data", adapter.DATA_ROOT / "Manatee" / "data")

    print("\n--- import availability ---")
    for module_name in [
        "numpy",
        "pandas",
        "torch",
        "scipy",
        "sklearn",
        "fastapi",
        "pydantic",
        "langchain_ollama",
        "gradio",
    ]:
        spec = importlib.util.find_spec(module_name)
        print(f"{module_name}: {'OK' if spec else 'MISSING'}")

    print("\n--- import app_state ---")
    if str(adapter.API_ROOT) not in sys.path:
        sys.path.insert(0, str(adapter.API_ROOT))
    try:
        import app_state  # type: ignore

        print(f"app_state import: OK ({Path(app_state.__file__).resolve()})")
    except Exception:
        print("app_state import: FAILED")
        traceback.print_exc()

    print("\n--- import api and validate ManateeData ---")
    try:
        data = adapter.ManateeData.from_api_module()
        print("api import: OK")
        print(f"genes: {len(data.genes)}")
        print(f"tfs: {len(data.tfs)}")
        print(f"labels: {len(data.labels)}")
        print(f"x shape: {getattr(data.x, 'shape', None)}")
        print(f"trrust entries: {len(data.trrust)}")
    except Exception:
        print("api import: FAILED")
        traceback.print_exc()


if __name__ == "__main__":
    main()

