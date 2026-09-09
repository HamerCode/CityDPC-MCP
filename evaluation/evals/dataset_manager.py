"""
Dataset Backup und Restore Manager für beide Dataset-Pfade.
"""

import hashlib
import shutil
from datetime import datetime
from pathlib import Path


def compute_file_hash(file_path: Path) -> str:
    """Berechnet SHA256 Hash einer Datei."""
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def create_dataset_backup(dataset_path: Path, backup_dir: Path) -> tuple[Path, str]:
    """
    Erstellt Backup des Datasets und gibt Pfad + Hash zurück.

    Returns:
        tuple[Path, str]: (backup_path, file_hash)
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{dataset_path.stem}_backup_{timestamp}{dataset_path.suffix}"
    backup_path = backup_dir / backup_name

    shutil.copy2(dataset_path, backup_path)
    file_hash = compute_file_hash(dataset_path)

    return backup_path, file_hash


def restore_dataset_from_backup(backup_path: Path, target_path: Path) -> None:
    """Stellt Dataset aus Backup wieder her."""
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup not found: {backup_path}")

    shutil.copy2(backup_path, target_path)


def verify_dataset_unchanged(dataset_path: Path, original_hash: str) -> bool:
    """Prüft ob Dataset sich geändert hat."""
    if not dataset_path.exists():
        return False

    current_hash = compute_file_hash(dataset_path)
    return current_hash == original_hash


def backup_datasets(dataset_paths: dict[str, Path], backup_dir: Path) -> dict[str, tuple[Path, str, Path]]:
    """
    Erstellt Backups aller Datasets.

    Args:
        dataset_paths: Dict mit {name: dataset_path}
        backup_dir: Verzeichnis für Backups

    Returns:
        dict: {name: (backup_path, file_hash, target_path)}
    """
    backups = {}

    for name, dataset_path in dataset_paths.items():
        if dataset_path.exists():
            backup_path, file_hash = create_dataset_backup(dataset_path, backup_dir)
            backups[name] = (backup_path, file_hash, dataset_path)

    return backups


def restore_datasets(backups: dict[str, tuple[Path, str, Path]]) -> dict[str, bool]:
    """
    Stellt alle Datasets aus Backups wieder her.

    Args:
        backups: Dict mit {name: (backup_path, original_hash, target_path)}

    Returns:
        dict: {name: was_modified}
    """
    results = {}

    for name, (backup_path, original_hash, target_path) in backups.items():
        # Prüfe ob geändert wurde
        was_modified = not verify_dataset_unchanged(target_path, original_hash)

        if was_modified:
            # Restore from backup
            restore_dataset_from_backup(backup_path, target_path)

        results[name] = was_modified

    return results


def cleanup_old_backups(backup_dir: Path, keep_last_n: int = 10) -> int:
    """
    Löscht alte Backups, behält nur die neuesten.

    Returns:
        int: Anzahl gelöschter Backups
    """
    if not backup_dir.exists():
        return 0

    backups = sorted(backup_dir.glob("*backup_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    to_delete = backups[keep_last_n:]

    for backup_file in to_delete:
        backup_file.unlink()

    return len(to_delete)
