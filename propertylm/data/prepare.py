from __future__ import annotations
import shutil, zipfile
from pathlib import Path
import typer


def unpack(source: Path, destination: Path, clear: bool = False) -> int:
    source = source.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if clear and destination.exists():
        shutil.rmtree(destination)
        destination.mkdir(parents=True)
    copied = 0
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            for i, member in enumerate(archive.infolist()):
                if member.is_dir() or not member.filename.lower().endswith((".csv", ".parquet")):
                    continue
                target = destination / f"{i:05d}_{Path(member.filename).name}"
                with archive.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                copied += 1
    elif source.is_dir():
        for i, item in enumerate(source.rglob("*")):
            if item.suffix.lower() in {".csv", ".parquet"}:
                shutil.copy2(item, destination / f"{i:05d}_{item.name}")
                copied += 1
    else:
        shutil.copy2(source, destination / source.name)
        copied = 1
    typer.echo(f"Prepared {copied} files in {destination}")
    return copied
