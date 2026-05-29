"""Native, git-independent snapshot version control for ``sirenspec launch``.

Each snapshot is a versioned copy of the workflow YAML stored under ``.sirenspec/versions/``
beside the workflow file, with metadata (version, label, trigger, timestamp) recorded in an
``index.json``.  The studio uses this to let users iterate fearlessly: ``/snapshot`` saves a
labelled version, ``/diff`` compares versions, and a one-key rollback restores a previous
version — itself snapshotted first, so rollback is reversible.  Works in non-git directories
and is complementary to git, not a replacement.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sirenspec.exceptions import SnapshotError

# Triggers whose snapshots are always retained (never pruned), because they mark
# deliberate or safety-critical points: a manual save, an accepted /edit change, or the
# safety copy taken before a rollback (so every rollback stays reversible).
RETAINED_TRIGGERS: frozenset[str] = frozenset({"manual", "accept", "rollback"})

DEFAULT_KEEP_LAST = 20

VERSIONS_DIRNAME = ".sirenspec/versions"
INDEX_FILENAME = "index.json"


@dataclass(frozen=True)
class Snapshot:
    """Metadata for one stored snapshot.

    :param version: Monotonic 1-based version number.
    :param label: Optional human label (empty for unlabelled auto snapshots).
    :param trigger: What created it — ``"manual"``, ``"auto"``, ``"test"``, ``"run"``,
        ``"accept"``, or ``"rollback"``.
    :param timestamp: ISO-8601 creation time (UTC).
    :param filename: The snapshot file's name within the versions directory.
    """

    version: int
    label: str
    trigger: str
    timestamp: str
    filename: str

    @property
    def ref(self) -> str:
        """Return the canonical reference string for this snapshot.

        :returns: The version reference, e.g. ``"v3"``.
        """
        return f"v{self.version}"

    def to_dict(self) -> dict[str, object]:
        """Serialise this snapshot's metadata for the index file.

        :returns: A JSON-serialisable metadata dict.
        """
        return {
            "version": self.version,
            "label": self.label,
            "trigger": self.trigger,
            "timestamp": self.timestamp,
            "filename": self.filename,
        }


def snapshot_from_dict(raw: dict[str, object]) -> Snapshot:
    """Reconstruct a :class:`Snapshot` from an index record.

    :param raw: A metadata dict previously produced by :meth:`Snapshot.to_dict`.
    :returns: The reconstructed :class:`Snapshot`.
    """
    return Snapshot(
        version=int(raw["version"]),  # type: ignore[arg-type]
        label=str(raw.get("label", "")),
        trigger=str(raw.get("trigger", "manual")),
        timestamp=str(raw.get("timestamp", "")),
        filename=str(raw["filename"]),
    )


class SnapshotStore:
    """Filesystem-backed snapshot store for a single workflow file.

    :param workflow_path: Path to the working workflow YAML file.
    :param keep_last: Maximum number of prunable (non-retained) snapshots to keep.
    """

    def __init__(self, workflow_path: Path, keep_last: int = DEFAULT_KEEP_LAST) -> None:
        self.workflow_path = workflow_path
        self.keep_last = keep_last
        self.versions_dir = workflow_path.parent / VERSIONS_DIRNAME
        self.index_path = self.versions_dir / INDEX_FILENAME

    def list(self) -> list[Snapshot]:
        """Return all snapshots ordered by ascending version.

        :raises SnapshotError: If the index file exists but cannot be parsed.
        :returns: The list of stored :class:`Snapshot` records.
        """
        if not self.index_path.exists():
            return []
        try:
            records = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise SnapshotError(f"Could not read snapshot index: {exc}") from exc
        return sorted((snapshot_from_dict(r) for r in records), key=lambda s: s.version)

    def latest(self) -> Snapshot | None:
        """Return the most recent snapshot, or ``None`` when none exist.

        :returns: The highest-version :class:`Snapshot`, or ``None``.
        """
        snapshots = self.list()
        return snapshots[-1] if snapshots else None

    def latest_label(self) -> str:
        """Return the status-bar label for the active snapshot.

        :returns: The latest snapshot ref (e.g. ``"v3"``), or ``"—"`` when none exist.
        """
        latest = self.latest()
        return latest.ref if latest is not None else "—"

    def next_version(self) -> int:
        """Return the version number the next snapshot will receive.

        :returns: One greater than the highest existing version, or ``1``.
        """
        latest = self.latest()
        return latest.version + 1 if latest is not None else 1

    def create(self, label: str = "", trigger: str = "manual") -> Snapshot:
        """Capture the current working file as a new snapshot.

        :param label: Optional human label for the snapshot.
        :param trigger: What initiated the snapshot (e.g. ``"manual"``, ``"auto"``).
        :raises SnapshotError: If the working file is missing or cannot be copied.
        :returns: The newly created :class:`Snapshot`.
        """
        if not self.workflow_path.exists():
            raise SnapshotError(f"Cannot snapshot missing workflow file: {self.workflow_path}")
        try:
            content = self.workflow_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SnapshotError(f"Could not read workflow file: {exc}") from exc

        version = self.next_version()
        snapshot = Snapshot(
            version=version,
            label=label.strip(),
            trigger=trigger,
            timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
            filename=f"v{version}.yaml",
        )
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        (self.versions_dir / snapshot.filename).write_text(content, encoding="utf-8")

        records = self.list()
        records.append(snapshot)
        self.write_index(records)
        self.prune()
        return snapshot

    def write_index(self, snapshots: list[Snapshot]) -> None:
        """Persist *snapshots* to the index file.

        :param snapshots: The full list of snapshot records to write.
        """
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        payload = [s.to_dict() for s in sorted(snapshots, key=lambda s: s.version)]
        self.index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def resolve(self, reference: str) -> Snapshot:
        """Resolve a snapshot by version ref (``"v2"`` / ``"2"``) or exact label.

        :param reference: A version reference or label.
        :raises SnapshotError: If no snapshot matches *reference*.
        :returns: The matching :class:`Snapshot`.
        """
        needle = reference.strip()
        snapshots = self.list()
        version_token = needle[1:] if needle.startswith("v") else needle
        if version_token.isdigit():
            target = int(version_token)
            for snapshot in snapshots:
                if snapshot.version == target:
                    return snapshot
        for snapshot in snapshots:
            if snapshot.label and snapshot.label == needle:
                return snapshot
        raise SnapshotError(f"No snapshot matches '{reference}'.")

    def read_content(self, snapshot: Snapshot) -> str:
        """Read the stored YAML content of *snapshot*.

        :param snapshot: The snapshot to read.
        :raises SnapshotError: If the snapshot file is missing or unreadable.
        :returns: The snapshot's YAML text.
        """
        path = self.versions_dir / snapshot.filename
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SnapshotError(f"Could not read snapshot {snapshot.ref}: {exc}") from exc

    def working_content(self) -> str:
        """Read the current working workflow file.

        :returns: The working file's text, or ``""`` if it is missing.
        """
        try:
            return self.workflow_path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def diff(self, snapshot: Snapshot, other: Snapshot | None = None) -> list[str]:
        """Produce a unified diff for *snapshot*.

        With *other* omitted, the diff is the working tree against *snapshot*.  With *other*
        supplied, it is *snapshot* against *other* (older → newer by version).

        :param snapshot: The base snapshot.
        :param other: Optional second snapshot to diff against instead of the working file.
        :returns: Unified-diff lines (without trailing newlines).
        """
        if other is None:
            from_text, from_label = self.read_content(snapshot), snapshot.ref
            to_text, to_label = self.working_content(), "working"
        else:
            older, newer = sorted((snapshot, other), key=lambda s: s.version)
            from_text, from_label = self.read_content(older), older.ref
            to_text, to_label = self.read_content(newer), newer.ref
        diff = difflib.unified_diff(
            from_text.splitlines(),
            to_text.splitlines(),
            fromfile=from_label,
            tofile=to_label,
            lineterm="",
        )
        return list(diff)

    def rollback(self, snapshot: Snapshot) -> Snapshot:
        """Restore *snapshot* to the working file, snapshotting the current state first.

        A safety snapshot of the current working file is taken (trigger ``"rollback"``)
        before the restore, so the rollback can itself be rolled back.

        :param snapshot: The snapshot to restore.
        :raises SnapshotError: If the restore cannot be written.
        :returns: The safety snapshot captured immediately before the restore.
        """
        safety = self.create(label=f"pre-rollback to {snapshot.ref}", trigger="rollback")
        content = self.read_content(snapshot)
        try:
            self.workflow_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise SnapshotError(f"Could not restore snapshot {snapshot.ref}: {exc}") from exc
        return safety

    def prune(self) -> None:
        """Enforce the retention policy.

        Snapshots whose trigger is in :data:`RETAINED_TRIGGERS` are always kept.  Of the
        remaining (prunable) snapshots, only the most recent ``keep_last`` are retained;
        older ones have their files and index records removed.
        """
        snapshots = self.list()
        prunable = [s for s in snapshots if s.trigger not in RETAINED_TRIGGERS]
        if len(prunable) <= self.keep_last:
            return
        to_remove = prunable[: len(prunable) - self.keep_last]
        remove_versions = {s.version for s in to_remove}
        for snapshot in to_remove:
            (self.versions_dir / snapshot.filename).unlink(missing_ok=True)
        kept = [s for s in snapshots if s.version not in remove_versions]
        self.write_index(kept)
