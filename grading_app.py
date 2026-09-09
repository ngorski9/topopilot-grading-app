#!/usr/bin/env python3
"""Local review interface for agent_experiments/results.

Run from this directory with: python3 grading_app.py
Then visit http://127.0.0.1:8080.  This intentionally has simple plaintext
local authentication, since it is designed only for a trusted local machine.
"""
from __future__ import annotations

import csv
import json
import mimetypes
import re
import threading
import webbrowser
from collections import defaultdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = ROOT
RESULTS = EXPERIMENTS_ROOT / "results"
TRIAL_DEFINITIONS = EXPERIMENTS_ROOT / "prompts" / "trials.json"
ACCOUNTS = ROOT / "grader_accounts.csv"
GRADES = ROOT / "grading_results.csv"
SUMMARY = ROOT / "grading_summary.csv"
BENCHMARKS = ROOT / "benchmark_scores.csv"
LOCK = threading.Lock()  # Deliberately serialize writes: one grader at a time.

TASK_NAMES = {
    "critical_points": "Critical point extraction",
    "contour_tree": "Contour tree extraction",
    "vector_critical_point_tracking": "Vector field critical point tracking",
    "scalar_critical_point_tracking": "Scalar field critical point tracking",
    "Morse-Smale segmentation": "Morse-Smale segmentation extraction",
    "hyperLIC": "Symmetric Tensor field visualization",
    "asymmetric_tensor": "Asymmetric tensor eigenvector partition",
    "persistence_diagram": "Persistence diagram extraction",
}

DATA_RUBRIC = [
    "Correctly computed",
    "Correctly computed, but edge cases are not handled correctly",
    "Computed with errors",
    "Not computed",
]
VIS_RUBRIC = [
    "Correctly visualized",
    "Visualized, but design choices impair interpretability",
    "Visualization requirements are only partially fulfilled",
    "Not visualized",
]
GRADE_FIELDS = ["trial_id", "grader", "data_grades", "vis_grades"]
BENCHMARK_FIELDS = ["trial_id", "benchmark_score", "benchmark_not_applicable"]


def csv_rows(path: Path):
    if not path.exists(): return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def has_grade_schema():
    if not GRADES.exists():
        return False
    with GRADES.open(newline="", encoding="utf-8") as f:
        return csv.DictReader(f).fieldnames == GRADE_FIELDS


def trial_requirements():
    """Return the data and visualization requirements keyed by trial name."""
    with TRIAL_DEFINITIONS.open(encoding="utf-8") as f:
        definitions = json.load(f)
    return {
        item["name"]: {
            "data_requirements": item.get("data_requirements", []),
            "vis_requirements": item.get("vis_requirements", []),
        }
        for item in definitions
    }


def parse_trial(folder: Path, metadata, requirements):
    name = folder.name
    match = re.match(r"^(.*)_(codex|claude)_run_(\d+)_prompt_(\d+)(_api)?$", name)
    if match:
        task, agent, run, prompt, api = match.groups()
        run_number = int(run)
        prompt_number = int(prompt)
        return {"id": name, "task": TASK_NAMES.get(task, task), "agent": agent, "run": run_number, "prompt": prompt_number, "api": bool(api), "metadata": metadata.get((agent, task, run_number, prompt_number, bool(api)), {}), **requirements.get(task, {"data_requirements": [], "vis_requirements": []})}
    return {"id": name, "task": name, "agent": "unknown", "run": 0, "prompt": 0, "api": False, "metadata": {}, "data_requirements": [], "vis_requirements": []}


def trials():
    meta = {}
    for row in csv_rows(RESULTS / "results.csv"):
        try:
            run = int(row.get("run", 0))
            prompt = int(row.get("prompt", 0))
        except ValueError:
            continue
        meta[(row.get("agent"), row.get("trial"), run, prompt, row.get("api") == "True")] = row
    requirements = trial_requirements()
    items = [parse_trial(p, meta, requirements) for p in RESULTS.iterdir() if p.is_dir() and not p.name.startswith(".")]
    return sorted(items, key=lambda t: (t["task"].lower(), t["prompt"], t["api"], t["agent"], t["run"]))


def safe_trial(trial_id):
    candidate = (RESULTS / trial_id).resolve()
    if candidate.parent != RESULTS.resolve() or not candidate.is_dir(): raise ValueError("Unknown trial")
    return candidate


HIDDEN_FILE_NAMES = {"agent_time.txt", "codex_stderr.txt"}
HIDDEN_FILE_SUFFIXES = {".vti", ".vtu", ".vtp", ".vtk", ".npy", ".npz", ".csv", ".json", ".jsonl", ".pvsm", ".pyc", ".log"}
HIDDEN_DIRECTORIES = {".topopilot", "claude_transcripts", "codex_transcripts"}


def visible_file(path: Path, trial: Path) -> bool:
    """Whether a trial artifact should be available in the file viewer."""
    relative = path.relative_to(trial)
    return (
        not any(part in HIDDEN_DIRECTORIES for part in relative.parts[:-1])
        and path.name not in HIDDEN_FILE_NAMES
        and path.suffix.lower() not in HIDDEN_FILE_SUFFIXES
    )


def benchmark_for(trial_id):
    row = next((r for r in csv_rows(BENCHMARKS) if r.get("trial_id") == trial_id), None)
    if row:
        return row
    return {"trial_id": trial_id, "benchmark_score": "", "benchmark_not_applicable": "0"}


def grades_from_row(row, field):
    try:
        value = json.loads(row.get(field, "[]"))
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def row_is_complete(row, trial):
    return (
        len(grades_from_row(row, "data_grades")) == len(trial["data_requirements"])
        and len(grades_from_row(row, "vis_grades")) == len(trial["vis_requirements"])
    )


def row_has_no_errors(row):
    return all(grade.get("rating") in (DATA_RUBRIC[0], VIS_RUBRIC[0]) for grade in grades_from_row(row, "data_grades") + grades_from_row(row, "vis_grades"))


def row_has_no_major_errors(row):
    allowed_ratings = DATA_RUBRIC[:2] + VIS_RUBRIC[:2]
    return all(grade.get("rating") in allowed_ratings for grade in grades_from_row(row, "data_grades") + grades_from_row(row, "vis_grades"))


def make_summary():
    grouped = defaultdict(list)
    for row in csv_rows(GRADES): grouped[row["trial_id"]].append(row)
    output = []
    for trial in trials():
        rows = grouped[trial["id"]]
        if not rows:
            continue
        count = len(rows)
        benchmark = benchmark_for(trial["id"])
        metadata = trial.get("metadata", {})
        output.append({"trial_id": trial["id"], "task": trial["task"], "agent": trial["agent"], "run": trial["run"], "prompt": trial["prompt"], "api": trial["api"], "input tokens": metadata.get("total input", ""), "output": metadata.get("total output", ""), "cost (usd)": metadata.get("cost (usd)", ""), "time": metadata.get("time", ""), "grader_count": count, "benchmark_score": benchmark.get("benchmark_score", ""), "benchmark_not_applicable": benchmark.get("benchmark_not_applicable") == "1", "percent_no_errors": round(100 * sum(row_has_no_errors(row) for row in rows) / count, 1), "percent_no_major_errors": round(100 * sum(row_has_no_major_errors(row) for row in rows) / count, 1)})
    fields = ["trial_id", "task", "agent", "run", "prompt", "api", "input tokens", "output", "cost (usd)", "time", "grader_count", "benchmark_score", "benchmark_not_applicable", "percent_no_errors", "percent_no_major_errors"]
    write_csv(SUMMARY, output, fields)


class App(SimpleHTTPRequestHandler):
    def log_message(self, format, *args): print("[grader]", format % args)
    def send_json(self, value, status=200):
        data = json.dumps(value).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def body(self):
        length = int(self.headers.get("Content-Length", 0)); return json.loads(self.rfile.read(length) or b"{}")
    def do_GET(self):
        parsed = urlparse(self.path); path = parsed.path
        try:
            if path == "/api/trials": return self.send_json(trials())
            if path == "/api/grades":
                query = parse_qs(parsed.query)
                trial_id = query.get("trial", [""])[0]
                rows = [r for r in csv_rows(GRADES) if r.get("trial_id") == trial_id]
                benchmark = benchmark_for(trial_id)
                for row in rows:
                    row["data_grades"] = grades_from_row(row, "data_grades")
                    row["vis_grades"] = grades_from_row(row, "vis_grades")
                    row["benchmark_score"] = benchmark.get("benchmark_score", "")
                    row["benchmark_not_applicable"] = benchmark.get("benchmark_not_applicable", "0")
                authenticated = any(r.get("username") == query.get("grader", [""])[0] and r.get("password") == query.get("password", [""])[0] for r in csv_rows(ACCOUNTS))
                # Avoid exposing login names to spectators.
                if not authenticated:
                    for i, row in enumerate(sorted(rows, key=lambda r: r.get("grader", "")), 1): row["grader"] = f"Grader {i}"
                return self.send_json(rows)
            if path == "/api/graded":
                query = parse_qs(parsed.query)
                grader, password = query.get("grader", [""])[0], query.get("password", [""])[0]
                authenticated = any(r.get("username") == grader and r.get("password") == password for r in csv_rows(ACCOUNTS))
                if not authenticated: return self.send_json({"error": "Log in again."}, 401)
                return self.send_json([r.get("trial_id") for r in csv_rows(GRADES) if r.get("grader") == grader])
            if path == "/api/files":
                trial = safe_trial(parse_qs(parsed.query).get("trial", [""])[0])
                return self.send_json([str(p.relative_to(trial)) for p in sorted(trial.rglob("*")) if p.is_file() and visible_file(p, trial)])
            if path == "/file":
                query = parse_qs(parsed.query); trial = safe_trial(query.get("trial", [""])[0]); rel = Path(unquote(query.get("path", [""])[0]))
                target = (trial / rel).resolve()
                if trial not in target.parents or not target.is_file() or not visible_file(target, trial): raise ValueError("Unknown file")
                mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                self.send_response(200); self.send_header("Content-Type", mime); self.send_header("Content-Length", str(target.stat().st_size)); self.end_headers()
                with target.open("rb") as f:
                    while chunk := f.read(65536): self.wfile.write(chunk)
                return
            if path == "/" or path == "/index.html":
                data = HTML_PATH.read_bytes(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            if path == "/static/style.css":
                data = (ROOT / "static" / "style.css").read_bytes(); self.send_response(200); self.send_header("Content-Type", "text/css; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            if path == "/static/app.js":
                data = (ROOT / "static" / "app.js").read_bytes(); self.send_response(200); self.send_header("Content-Type", "text/javascript; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            self.send_error(404)
        except (ValueError, OSError) as e: self.send_json({"error": str(e)}, 400)
    def do_POST(self):
        try:
            data = self.body()
            if self.path == "/api/login":
                valid = any(r.get("username") == data.get("username") and r.get("password") == data.get("password") for r in csv_rows(ACCOUNTS))
                return self.send_json({"ok": valid}, 200 if valid else 401)
            if self.path == "/api/grade":
                grader = data.get("grader", ""); trial_id = data.get("trial_id", "")
                if not any(r.get("username") == grader and r.get("password") == data.get("password") for r in csv_rows(ACCOUNTS)): return self.send_json({"error": "Log in again."}, 401)
                safe_trial(trial_id)
                trial = next((item for item in trials() if item["id"] == trial_id), None)
                if trial is None: raise ValueError("Unknown trial")
                def validate_grades(field, requirements, options):
                    values = data.get(field, [])
                    if not isinstance(values, list) or len(values) != len(requirements):
                        raise ValueError(f"Every {field.replace('_', ' ')} requirement must be graded.")
                    normalized = []
                    for requirement, grade in zip(requirements, values):
                        if not isinstance(grade, dict) or grade.get("rating") not in options:
                            raise ValueError(f"Select a rubric option for: {requirement}")
                        normalized.append({"rating": grade["rating"], "comment": str(grade.get("comment", ""))})
                    return normalized
                row = {
                    "trial_id": trial_id,
                    "grader": grader,
                    "data_grades": json.dumps(validate_grades("data_grades", trial["data_requirements"], DATA_RUBRIC)),
                    "vis_grades": json.dumps(validate_grades("vis_grades", trial["vis_requirements"], VIS_RUBRIC)),
                }
                with LOCK:
                    rows = [r for r in csv_rows(GRADES) if (r.get("trial_id"), r.get("grader")) != (trial_id, grader)]
                    rows.append(row)
                    write_csv(GRADES, rows, GRADE_FIELDS)
                    benchmarks = [r for r in csv_rows(BENCHMARKS) if r.get("trial_id") != trial_id]
                    benchmarks.append({"trial_id": trial_id, "benchmark_score": str(data.get("benchmark_score", "")), "benchmark_not_applicable": "1" if data.get("benchmark_not_applicable") in ("1", 1, True) else "0"})
                    write_csv(BENCHMARKS, benchmarks, BENCHMARK_FIELDS)
                    make_summary()
                return self.send_json({"ok": True})
            self.send_json({"error": "Not found"}, 404)
        except (ValueError, OSError, json.JSONDecodeError) as e: self.send_json({"error": str(e)}, 400)


HTML_PATH = ROOT / "index.html"

if __name__ == "__main__":
    if not ACCOUNTS.exists(): write_csv(ACCOUNTS, [{"username": "grader1", "password": "change-me"}], ["username", "password"])
    # This release intentionally starts a fresh per-requirement grade file.
    # Older checkbox-based grades cannot be mapped to the new rubric.
    if not has_grade_schema():
        write_csv(GRADES, [], GRADE_FIELDS)
    if not BENCHMARKS.exists(): write_csv(BENCHMARKS, [], BENCHMARK_FIELDS)
    make_summary()
    url = "http://127.0.0.1:8080"
    print(f"Trial grader running at {url}  (Ctrl-C to stop)")
    # Schedule this after startup so the browser does not race the server bind.
    threading.Timer(0.25, webbrowser.open, args=(url,)).start()
    ThreadingHTTPServer(("127.0.0.1", 8080), App).serve_forever()
