"""Human-in-the-loop conflict review UI (Phase D).

Deliberately built on the standard-library http.server so the platform stays
local and dependency-free (no FastAPI/uvicorn). It binds to localhost only and is
single-user — auth and a real web stack are deferred to the hosted phase.

All rendering and action logic lives in ReviewService (pure, unit-testable). The
HTTP handler is a thin adapter, so tests never bind a socket. Resolve/verdict are
injected callables, keeping this module decoupled from ingestion and governance."""
from __future__ import annotations

import html
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


def _esc(s) -> str:
    return html.escape("" if s is None else str(s))


_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Conflict Review</title>
<style>
 body{{font-family:system-ui,Arabic,sans-serif;margin:2rem;color:#1a1a1a}}
 table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ccc;padding:.5rem;text-align:left}}
 .a{{background:#eef}} .b{{background:#fee}} textarea{{width:100%}}
 .status-suspected{{color:#a60}} .status-confirmed{{color:#c00}} .status-resolved{{color:#080}}
</style></head><body>{body}</body></html>"""


class ReviewService:
    """Backing logic for the review UI. resolve_fn(conflict_id, decision, text,
    approver) and verdict_fn(conflict_id, is_conflict, rationale) are injected."""

    def __init__(self, governance, store, resolve_fn, verdict_fn):
        self.governance = governance
        self.store = store
        self.resolve_fn = resolve_fn
        self.verdict_fn = verdict_fn

    # --- rendering -----------------------------------------------------------
    def list_html(self) -> str:
        conflicts = self.governance.get_conflicts()
        if not conflicts:
            return _PAGE.format(body="<h1>Conflict Review</h1><p>No conflicts.</p>")
        rows = "".join(
            f"<tr><td><a href='/conflict/{_esc(c.conflict_id)}'>{_esc(c.conflict_id)}</a></td>"
            f"<td>{_esc(c.module)}</td>"
            f"<td class='status-{_esc(c.status)}'>{_esc(c.status)}</td>"
            f"<td>{_esc(c.source_a)}</td><td>{_esc(c.source_b)}</td>"
            f"<td>{c.similarity:.3f}</td></tr>"
            for c in conflicts
        )
        body = ("<h1>Conflict Review</h1><table>"
                "<tr><th>ID</th><th>Module</th><th>Status</th>"
                "<th>Source A</th><th>Source B</th><th>Similarity</th></tr>"
                f"{rows}</table>")
        return _PAGE.format(body=body)

    def detail_html(self, conflict_id: str) -> str:
        c = self.governance.get_conflict(conflict_id)
        if c is None:
            return _PAGE.format(body="<h1>Not found</h1><p><a href='/'>Back</a></p>")
        chunks = {ch.id: ch for ch in self.store.get_chunks_by_ids([c.chunk_a_id, c.chunk_b_id])}
        ta = _esc(chunks[c.chunk_a_id].text) if c.chunk_a_id in chunks else "(missing)"
        tb = _esc(chunks[c.chunk_b_id].text) if c.chunk_b_id in chunks else "(missing)"
        body = f"""
        <h1>Conflict {_esc(c.conflict_id)}</h1>
        <p>Status: <b class='status-{_esc(c.status)}'>{_esc(c.status)}</b> &mdash;
           module {_esc(c.module)} &mdash; similarity {c.similarity:.3f}</p>
        <table><tr>
          <td class='a'><b>Source A</b><br>{_esc(c.source_a)}<hr>{ta}</td>
          <td class='b'><b>Source B</b><br>{_esc(c.source_b)}<hr>{tb}</td>
        </tr></table>
        <h3>Adjudicate</h3>
        <form method='post' action='/verdict'>
          <input type='hidden' name='conflict_id' value='{_esc(c.conflict_id)}'>
          <input name='rationale' placeholder='rationale' size='60'>
          <button name='is_conflict' value='1'>Confirm conflict</button>
          <button name='is_conflict' value='0'>Dismiss (no conflict)</button>
        </form>
        <h3>Resolve (authoritative decision)</h3>
        <form method='post' action='/resolve'>
          <input type='hidden' name='conflict_id' value='{_esc(c.conflict_id)}'>
          <select name='decision'>
            <option value='source_a'>Source A correct</option>
            <option value='source_b'>Source B correct</option>
            <option value='merge'>Merge both</option>
            <option value='new_rule' selected>Create new rule</option>
          </select>
          <input name='approver' placeholder='approver' size='20'><br>
          <textarea name='resolution_text' rows='4'
            placeholder='Authoritative resolution text'></textarea><br>
          <button type='submit'>Approve resolution</button>
        </form>
        <p><a href='/'>Back to list</a></p>"""
        return _PAGE.format(body=body)

    # --- actions -------------------------------------------------------------
    def verdict(self, conflict_id: str, is_conflict: bool, rationale: str = "") -> None:
        self.verdict_fn(conflict_id, is_conflict, rationale)

    def resolve(self, conflict_id: str, decision: str, text: str, approver: str) -> None:
        self.resolve_fn(conflict_id, decision, text, approver)


def make_handler(service: ReviewService):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: str, status: int = 200):
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _redirect(self, location: str):
            self.send_response(303)
            self.send_header("Location", location)
            self.end_headers()

        def do_GET(self):  # noqa: N802
            path = urlparse(self.path).path
            if path == "/" or path == "":
                self._send(service.list_html())
            elif path.startswith("/conflict/"):
                self._send(service.detail_html(path.split("/conflict/", 1)[1]))
            else:
                self._send("<h1>404</h1>", status=404)

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(length).decode("utf-8"))
            cid = form.get("conflict_id", [""])[0]
            path = urlparse(self.path).path
            if path == "/verdict":
                service.verdict(cid, form.get("is_conflict", ["0"])[0] == "1",
                                form.get("rationale", [""])[0])
            elif path == "/resolve":
                service.resolve(cid, form.get("decision", ["new_rule"])[0],
                                form.get("resolution_text", [""])[0],
                                form.get("approver", [""])[0])
            self._redirect(f"/conflict/{cid}" if cid else "/")

        def log_message(self, *args):  # silence default stderr logging
            pass

    return Handler


def run_in_thread(service: ReviewService, host: str = "127.0.0.1", port: int = 8765):
    """Start the review UI on a daemon thread (localhost). Returns (httpd, thread)
    so callers can shut it down."""
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread
