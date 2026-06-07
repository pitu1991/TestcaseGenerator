"""Notifications (Phase D) — wired placeholder.

The conflict workflow needs to alert reviewers when a conflict is flagged. We wire
the call site now with a swappable interface; the default just logs. Replace
LogNotifier with an EmailNotifier / SlackNotifier subclass later — no call-site
change required. This is the 'have a placeholder, just wire it' decision."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class Notifier:
    """Interface. Methods are no-ops by default so subclasses override only what
    they need."""

    def conflict_created(self, conflict) -> None:  # noqa: D401
        pass


class NullNotifier(Notifier):
    """Does nothing. Use in tests or when notifications are undesired."""


class LogNotifier(Notifier):
    """Default placeholder: writes a line to the log. Swap for email/Slack later."""

    def conflict_created(self, conflict) -> None:
        log.info(
            "CONFLICT FLAGGED %s [%s] %s <-> %s (similarity=%.3f) — assign a reviewer",
            conflict.conflict_id, conflict.module or "-",
            conflict.source_a, conflict.source_b, conflict.similarity,
        )
