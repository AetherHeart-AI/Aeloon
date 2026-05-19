"""Path helpers for Aeloon runtime data."""

from __future__ import annotations

from pathlib import Path

from aeloon.core.config.loader import get_aeloon_home, get_config_path
from aeloon.utils.helpers import ensure_dir


def get_data_dir() -> Path:
    """Return the runtime data directory."""
    return ensure_dir(get_profile_root())


def get_runtime_subdir(name: str) -> Path:
    """Return a named runtime subdirectory."""
    return ensure_dir(get_data_dir() / name)


def get_media_dir(channel: str | None = None) -> Path:
    """Return the media directory."""
    base = get_runtime_subdir("media")
    return ensure_dir(base / channel) if channel else base


def get_cron_dir() -> Path:
    """Return the cron data directory."""
    return get_runtime_subdir("cron")


def get_logs_dir() -> Path:
    """Return the logs directory."""
    return get_runtime_subdir("logs")


def get_gateway_log_path() -> Path:
    """Return the gateway log file path."""
    return get_logs_dir() / "gateway.log"


def get_profile_name(config_path: Path | None = None) -> str | None:
    """Return the active named profile, or None for the default profile."""
    path = (config_path or get_config_path()).expanduser().resolve()
    home = get_aeloon_home().expanduser().resolve()
    try:
        relative_parent = path.parent.relative_to(home)
    except ValueError:
        if "profiles" in path.parent.parts:
            parts = list(path.parent.parts)
            index = parts.index("profiles")
            if len(parts) > index + 1:
                return parts[index + 1]
        return None
    if len(relative_parent.parts) >= 2 and relative_parent.parts[0] == "profiles":
        return relative_parent.parts[1]
    return None


def get_profile_root(
    profile: str | None = None,
    *,
    config_path: Path | None = None,
) -> Path:
    """Return the root directory for the active or named profile."""
    normalized_profile = _normalize_profile_name(profile)
    if normalized_profile is not None:
        return get_aeloon_home() / "profiles" / normalized_profile
    if config_path is not None:
        return config_path.expanduser().resolve().parent
    active_config_path = get_config_path().expanduser().resolve()
    home = get_aeloon_home().expanduser().resolve()
    try:
        relative_parent = active_config_path.parent.relative_to(home)
    except ValueError:
        return active_config_path.parent
    if len(relative_parent.parts) >= 2 and relative_parent.parts[0] == "profiles":
        return home / "profiles" / relative_parent.parts[1]
    return active_config_path.parent


def get_env_path(
    profile: str | None = None,
    *,
    config_path: Path | None = None,
) -> Path:
    """Return the profile-scoped .env file path."""
    return get_profile_root(profile, config_path=config_path) / ".env"


def get_prompt_memory_dir(
    profile: str | None = None,
    *,
    config_path: Path | None = None,
) -> Path:
    """Return the prompt-memory directory for the active profile."""
    return ensure_dir(get_profile_root(profile, config_path=config_path) / "memory")


def get_provider_state_dir(
    profile: str | None = None,
    *,
    config_path: Path | None = None,
) -> Path:
    """Return the provider-state directory for the active profile."""
    return ensure_dir(get_profile_root(profile, config_path=config_path) / "providers")


def get_archive_db_path(
    profile: str | None = None,
    *,
    config_path: Path | None = None,
) -> Path:
    """Return the archive database path for the active profile."""
    return get_profile_root(profile, config_path=config_path) / "archive.db"


def get_workspace_path(workspace: str | Path | None = None) -> Path:
    """Return the workspace path."""
    return Path(workspace).expanduser() if workspace else get_aeloon_home() / "workspace"


def get_cli_history_path() -> Path:
    """Return the CLI history path."""
    return get_aeloon_home() / "history" / "cli_history"


class ProjectStorageGateway:
    """Workspace-scoped storage roots for project-local generated artifacts."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.root = self.workspace / ".aeloon"

    def project_root(self, *, create: bool = True) -> Path:
        return ensure_dir(self.root) if create else self.root

    def project_cache_root(self, name: str | None = None, *, create: bool = True) -> Path:
        root = self.project_root(create=create) / "cache"
        if name:
            root = root / name
        return ensure_dir(root) if create else root

    def project_compiled_skills_root(self, *, create: bool = True) -> Path:
        root = self.project_cache_root(create=create) / "compiled_skills"
        return ensure_dir(root) if create else root

    def project_skills_root(self, *, create: bool = True) -> Path:
        root = self.project_root(create=create) / "skills"
        return ensure_dir(root) if create else root

    def project_workflows_root(self, *, create: bool = True) -> Path:
        root = self.project_root(create=create) / "workflows"
        return ensure_dir(root) if create else root


def get_storage_gateway(workspace: str | Path) -> ProjectStorageGateway:
    """Return the storage gateway for a project workspace."""
    return ProjectStorageGateway(workspace)


def get_bridge_install_dir() -> Path:
    """Return the WhatsApp bridge directory."""
    return get_aeloon_home() / "bridge"


def get_legacy_sessions_dir() -> Path:
    """Return the legacy sessions directory."""
    return get_aeloon_home() / "sessions"


def get_wechat_accounts_dir() -> Path:
    """Return the WeChat accounts directory."""
    return ensure_dir(get_aeloon_home() / "accounts" / "wechat")


def get_wechat_login_qr_dir() -> Path:
    """Return the WeChat login QR directory."""
    return ensure_dir(get_aeloon_home() / "media" / "wechat-login")


def _normalize_profile_name(profile: str | None) -> str | None:
    if profile is None:
        return None
    normalized = profile.strip()
    if not normalized or normalized == "default":
        return None
    return normalized
