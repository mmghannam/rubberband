"""Contains AnalyzeExternalView: hand a run/comparison off to LogAnalyzer."""

import asyncio
import json
import logging
import os
import re
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

from elasticsearch.dsl import Search
from tornado.httpclient import AsyncHTTPClient, HTTPRequest
from tornado.options import options
from tornado.web import HTTPError

from rubberband.constants import EXPORT_FILE_TYPES, RESULT_INDEX
from rubberband.models import TestSet

from .base import BaseHandler
from .result import load_testsets_files

logger = logging.getLogger(__name__)

# Streaming progress page shown while the /analyze handoff runs. The handler
# writes this once, flushes it so the spinner appears immediately, then streams
# <script> chunks that update the stage text as each step completes, and finally
# streams a window.location redirect. The page is self-contained (no CDN).
_PROGRESS_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Handing off to LogAnalyzer…</title>
<style>
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         background:#f5f7fa; display:flex; align-items:center; justify-content:center; min-height:100vh; }
  .card { background:#fff; border-radius:12px; box-shadow:0 4px 24px rgba(0,0,0,.08);
          padding:40px 48px; max-width:540px; text-align:center; }
  .spinner { width:48px; height:48px; border:5px solid #e6ecf3; border-top-color:#2eb3b3;
             border-radius:50%; margin:0 auto 20px; animation:spin .9s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  h1 { font-size:20px; color:#2c3e50; margin:0 0 6px; }
  p.stage { color:#5a6b7b; font-size:14px; margin:0; min-height:20px; }
  .steps { margin-top:20px; text-align:left; font-size:13px; }
  .steps div { padding:4px 0; color:#c8d1d9; }
  .steps .done { color:#2eb3b3; }
  .steps .active { color:#2c3e50; font-weight:600; }
  .steps .error { color:#c0392b; font-weight:600; }
</style>
</head>
<body>
<div class="card">
  <div class="spinner" id="spinner"></div>
  <h1>Handing off to LogAnalyzer</h1>
  <p class="stage" id="stage">Preparing…</p>
  <div class="steps" id="steps"></div>
</div>
<script>
  function setStage(msg){ document.getElementById('stage').textContent = msg; }
  function addStep(label, active){
    var el = document.createElement('div');
    el.textContent = label;
    el.className = active ? 'active' : 'done';
    document.getElementById('steps').appendChild(el);
  }
  function setFailed(msg){
    document.getElementById('spinner').style.display = 'none';
    setStage(msg);
    var e = document.createElement('div'); e.className = 'error'; e.textContent = msg;
    document.getElementById('steps').appendChild(e);
  }
</script>
"""


def _result_coverage(ts_ids):
    """Count parsed results per testset in Elasticsearch (one cheap agg query).

    Returns a ``{testset_id: count}`` dict. An empty dict means ES is
    unreachable or nothing matched, which callers treat as "no results".
    """
    try:
        search = Search(index=RESULT_INDEX).query("terms", testset_id=ts_ids)
        search.aggs.bucket("per_ts", "terms", field="testset_id", size=len(ts_ids))
        resp = search.execute()
        return {b.key: b.doc_count for b in resp.aggregations.per_ts.buckets}
    except Exception:  # noqa: BLE001 - coverage is advisory
        return {}


def _es_run_groups(ts_ids):
    """Group testsets into runs by their filename's setting.

    Mirrors LogAnalyzer's rubberband convention
    ``check.<testset>.<binary_timestamp>.<queue>.<setting>-s<seed>.out``: testsets
    that share a setting (differ only by seed) belong to the same run. Returns
    ``{run_name: {"settings": ..., "testset_ids": [...]}}``.
    """
    def _load(ts_id):
        try:
            return TestSet.get(id=ts_id)
        except Exception:  # noqa: BLE001 - skip unreadable testsets
            return None

    if len(ts_ids) == 1:
        testsets = [_load(ts_ids[0])]
    else:
        with ThreadPoolExecutor(max_workers=min(len(ts_ids), 8)) as executor:
            testsets = list(executor.map(_load, ts_ids))

    groups = {}
    for ts in testsets:
        if ts is None:
            continue
        stem = os.path.splitext(ts.filename or "")[0]
        stem = re.sub(r"-s\d+$", "", stem)  # strip the -s<seed> suffix
        parts = stem.split(".")
        if len(parts) >= 5 and parts[0] == "check":
            # drop the timestamp from the binary part, like LogAnalyzer does
            binary = parts[2].rsplit("_", 1)[0] if "_" in parts[2] else parts[2]
            name = ".".join([parts[0], parts[1], binary, parts[3], parts[-1]])
        else:
            name = stem
        settings = parts[-1] if parts else ""
        groups.setdefault(
            name, {"settings": settings, "testset_ids": []}
        )["testset_ids"].append(ts.meta.id)
    return groups


def _encode_multipart(fields, files):
    """
    Encode form fields and files as multipart/form-data.

    Parameters
    ----------
    fields : dict
        name -> string value
    files : list of tuple
        (field_name, filename, content_type, bytes)

    Returns
    -------
    (bytes, str)
        the encoded body and the matching Content-Type header value
    """
    boundary = uuid.uuid4().hex
    marker = ("--" + boundary).encode()
    lines = []
    for name, value in fields.items():
        lines += [
            marker,
            f'Content-Disposition: form-data; name="{name}"'.encode(),
            b"",
            value.encode(),
        ]
    for name, filename, content_type, data in files:
        lines += [
            marker,
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"'.encode(),
            f"Content-Type: {content_type}".encode(),
            b"",
            data,
        ]
    lines += [("--" + boundary + "--").encode(), b""]
    body = b"\r\n".join(lines)
    return body, f"multipart/form-data; boundary={boundary}"


class AnalyzeExternalView(BaseHandler):
    """Bundle one or more TestSets and hand them to a LogAnalyzer instance."""

    # --- streaming progress page helpers -----------------------------------
    async def _progress_start(self):
        """Start the staged progress page; the spinner renders immediately."""
        self.set_header("Content-Type", "text/html; charset=UTF-8")
        self.set_header("Cache-Control", "no-store")
        self.write(_PROGRESS_PAGE)
        await self.flush()

    async def _stage(self, label, message):
        """Mark a step done and stream the current-stage text."""
        self.write(
            "<script>addStep({}, false); setStage({});</script>\n".format(
                json.dumps(label), json.dumps(message)
            )
        )
        await self.flush()

    async def _progress_redirect(self, url):
        """Finish the progress page and send the browser to LogAnalyzer."""
        self.write(
            "<script>addStep('Opening in LogAnalyzer', true); "
            "setStage('Redirecting to LogAnalyzer…');</script>\n"
        )
        self.write("<script>window.location.replace({});</script>\n".format(json.dumps(url)))
        self.write("</body></html>")
        await self.flush()
        self.finish()

    async def _progress_fail(self, message):
        """Show an error on the progress page instead of a blank/half page."""
        logger.error("LogAnalyzer handoff failed: %s", message)
        self.write(
            "<script>setFailed({});</script>\n</body></html>".format(json.dumps(message))
        )
        await self.flush()
        self.finish()

    async def _handoff_from_es(self, ts_ids):
        """Ask LogAnalyzer to import the testsets' already-parsed ES results.

        Returns the LogAnalyzer redirect URL when the ES path was used, or None
        when the caller should fall back to the raw-log upload.
        """
        base = options.loganalyzer_url.rstrip("/")
        public_base = (
            options.loganalyzer_public_url or options.loganalyzer_url
        ).rstrip("/")
        groups = await asyncio.to_thread(_es_run_groups, ts_ids)
        if not groups:
            return None
        runs = [
            {"name": name, "settings": g["settings"], "testset_ids": g["testset_ids"]}
            for name, g in groups.items()
        ]
        await self._stage(
            "Importing from Elasticsearch",
            f"Found parsed results — importing {len(runs)} run(s) "
            f"from Elasticsearch (no re-parse)…",
        )
        body = json.dumps(
            {
                "es_url": options.elasticsearch_url,
                "description": "Imported from Rubberband",
                "runs": runs,
            }
        ).encode()
        request = HTTPRequest(
            url=f"{base}/api/import-from-es",
            method="POST",
            body=body,
            headers={"Content-Type": "application/json"},
            request_timeout=120,
        )
        try:
            response = await AsyncHTTPClient().fetch(request)
            payload = json.loads(response.body)
        except Exception as e:  # noqa: BLE001 - fall back to the raw upload
            logger.warning("import-from-es failed (%r); falling back to raw upload", e)
            return None

        created = payload.get("runs") or []
        if not created or payload.get("skipped"):
            # no/partial parsed results - let the raw-log upload handle it
            return None
        run_ids = [r["run_id"] for r in created if r.get("run_id")]
        if not run_ids:
            return None
        logger.info("LogAnalyzer ES handoff: runs %s", run_ids)
        if len(run_ids) == 2:
            return public_base + "/compare?runs=" + ",".join(run_ids)
        if len(run_ids) == 1:
            return "{}/instances/{}".format(public_base, run_ids[0])
        return public_base + "/"

    async def get(self, testsets):
        """
        Zip the raw logs of the given testsets, upload them to LogAnalyzer and
        redirect the user to the resulting LogAnalyzer run page.

        Parameters
        ----------
        testsets : str
            comma-separated TestSet ids
        """
        # internal URL for the server-to-server upload; public URL for the
        # browser redirect (behind a reverse proxy these differ)
        base = options.loganalyzer_url.rstrip("/")
        public_base = (
            options.loganalyzer_public_url or options.loganalyzer_url
        ).rstrip("/")
        if not base:
            raise HTTPError(404, reason="LogAnalyzer integration is not configured.")

        ts_ids = [t for t in testsets.split(",") if t]
        if not ts_ids:
            raise HTTPError(400, reason="No testsets given.")

        # Stream a staged progress page so the user sees what is happening while
        # the handoff runs (it can take several seconds for the ES import or the
        # raw-log upload), instead of a blank tab that looks stuck.
        await self._progress_start()
        await self._stage(
            "Contacting Elasticsearch", "Checking for already-parsed results…"
        )

        # Fast path: if every testset already has parsed results in Elasticsearch,
        # hand those to LogAnalyzer directly. This skips shipping the raw logs
        # (~116 MB for a C6520 comparison) and LogAnalyzer re-parsing them
        # (~6 s/run), which dominate the cold handoff. LogAnalyzer's
        # import-from-es endpoint builds runs from the already-parsed values
        # (verified equivalent to its own parse). If any testset lacks results,
        # fall back to the raw-log upload below.
        coverage = await asyncio.to_thread(_result_coverage, ts_ids)
        if coverage and all(coverage.get(t, 0) > 0 for t in ts_ids):
            redirect_url = await self._handoff_from_es(ts_ids)
            if redirect_url:
                await self._progress_redirect(redirect_url)
                return

        # Only the raw logs are needed here (results/settings are not), so use
        # the lightweight loader that skips the expensive result scan. It runs in
        # a worker thread (off the event loop) and fetches the runs concurrently.
        await self._stage(
            "Downloading raw logs",
            "No parsed results in Elasticsearch — downloading the raw logs…",
        )
        try:
            ts_list = await asyncio.to_thread(load_testsets_files, ts_ids)
        except Exception as e:  # noqa: BLE001 - show a helpful error on the page
            await self._progress_fail("Could not load the selected testsets: " + str(e))
            return

        # Build the same raw-log archive the download button produces. LogAnalyzer
        # detects the Rubberband filename convention and re-parses with its own
        # parser, so we hand over the raw logs, not Rubberband's parsed data.
        #
        # Use ZIP_STORED (no deflate): LogAnalyzer runs on the same machine, so
        # the larger archive costs nothing on the loopback link, and skipping
        # compression avoids the deflate CPU here and the inflate CPU in
        # LogAnalyzer. The download button below stays ZIP_DEFLATED because that
        # archive goes to the user's browser over the network.
        with BytesIO() as byteio:
            with zipfile.ZipFile(byteio, "w", zipfile.ZIP_STORED) as archive:
                for ts in ts_list:
                    for ftype in EXPORT_FILE_TYPES:
                        try:
                            archive.writestr(
                                f"{ts.meta.id}/{os.path.splitext(ts.filename)[0]}{ftype}",
                                ts.files[ftype.lstrip(".")].text,
                            )
                        except (AttributeError, KeyError):
                            # no file of this type for this testset
                            pass
            zip_bytes = byteio.getvalue()

        logger.info(
            "LogAnalyzer handoff: base=%s zip_bytes=%d for %s",
            base,
            len(zip_bytes),
            ",".join(ts_ids),
        )

        label = (", ".join(ts.filename for ts in ts_list))[:120] or "Rubberband run"
        await self._stage(
            "Uploading to LogAnalyzer",
            f"Uploading {len(ts_list)} raw log(s) to LogAnalyzer…",
        )
        body, content_type = _encode_multipart(
            fields={"name": label, "description": "Imported from Rubberband"},
            files=[("files", "rubberband.zip", "application/zip", zip_bytes)],
        )

        request = HTTPRequest(
            url=f"{base}/api/upload",
            method="POST",
            body=body,
            headers={"Content-Type": content_type},
            request_timeout=180,
        )
        try:
            response = await AsyncHTTPClient().fetch(request)
        except Exception as e:  # noqa: BLE001 - surface any transport/HTTP error
            logger.error("LogAnalyzer upload failed: %r", e)
            if getattr(e, "response", None) is not None:
                logger.error(
                    "LogAnalyzer upload response body: %s", e.response.body[:1000]
                )
            await self._progress_fail("Could not reach LogAnalyzer: " + str(e))
            return

        try:
            payload = json.loads(response.body)
        except ValueError:
            await self._progress_fail("Unexpected response from LogAnalyzer.")
            return

        # A single run lands directly on its instances page. When LogAnalyzer
        # splits a Rubberband comparison into several runs (it groups by setting),
        # it returns {"multiple_runs": true, "runs": [...]} with no top-level
        # run_id. For a two-run comparison we hand the run ids straight to
        # LogAnalyzer's /compare route, which skips the dashboard, creates (or
        # reuses a cached) comparison and lands the user on the comparison page.
        # LogAnalyzer currently only renders two-run comparisons, so anything
        # larger falls back to the dashboard where the fresh runs appear.
        if payload.get("run_id"):
            await self._progress_redirect(
                "{}/instances/{}".format(public_base, payload["run_id"])
            )
        elif payload.get("runs"):
            run_ids = [r["run_id"] for r in payload["runs"] if r.get("run_id")]
            if len(run_ids) == 2:
                await self._progress_redirect(
                    public_base + "/compare?runs=" + ",".join(run_ids)
                )
            else:
                await self._progress_redirect(public_base + "/")
        else:
            await self._progress_fail("Unexpected response from LogAnalyzer.")
