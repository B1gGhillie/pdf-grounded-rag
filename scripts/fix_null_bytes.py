"""Fix Python source files corrupted with null bytes or UTF-16 encoding."""

from __future__ import annotations

from pathlib import Path


def clean_file(path: Path) -> bool:
    raw = path.read_bytes()
    if not raw:
        path.write_text("", encoding="utf-8", newline="\n")
        return True

    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
        path.write_text(text, encoding="utf-8", newline="\n")
        return True

    if b"\x00" in raw:
        text = raw.replace(b"\x00", b"").decode("utf-8", errors="ignore")
        path.write_text(text, encoding="utf-8", newline="\n")
        return True

    return False


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    fixed: list[str] = []

    for path in root.rglob("*.py"):
        if ".venv" in path.parts:
            continue
        if clean_file(path):
            fixed.append(str(path.relative_to(root)))

    if fixed:
        print("Fixed files:")
        for item in fixed:
            print(f"  - {item}")
    else:
        print("No corrupted Python files found.")


if __name__ == "__main__":
    main()
