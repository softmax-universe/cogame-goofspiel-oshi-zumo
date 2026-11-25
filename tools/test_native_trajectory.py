"""Qualify native host+player round-trip: regular JSON, fenced JSON, retry, invalid/ambiguous object, provider failures, and missing credentials."""

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from websockets.sync.client import connect

binary, output, revision = sys.argv[1:4]


def choose_action(user):
    return {"bid": int(re.search(r"YOUR LEGAL BIDS: ([0-9]+)", user)[1])}


def run_mode(root: Path, mode: str, config: dict):
    folder = root / mode
    folder.mkdir(mode=0o700)
    calls = []
    use_stub = mode != "no_credentials"
    expect_abort = mode in ("invalid_ambiguous", "provider_failure", "provider_auth_failure", "no_credentials")

    class Fixture(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            assert self.path == "/v1/messages"
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            user = request["messages"][0]["content"]
            call_id = str(uuid.uuid4())
            if mode == "provider_failure":
                body = json.dumps({"error": {"type": "api_error", "message": "upstream 500"}}).encode()
                calls.append({"id": call_id, "request": request, "response": json.loads(body), "status": 500})
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-Softmax-Llm-Call-Id", call_id)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if mode == "provider_auth_failure":
                body = json.dumps({"error": {"type": "authentication_error", "message": "invalid api key"}}).encode()
                calls.append({"id": call_id, "request": request, "response": json.loads(body), "status": 401})
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-Softmax-Llm-Call-Id", call_id)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            action = choose_action(user)
            raw = json.dumps(action, separators=(",", ":"))
            if mode == "fenced":
                raw = "```json\n" + raw + "\n```"
            elif mode == "retry" and "previous reply was" not in user:
                raw = "Here is my bid: " + raw
            elif mode == "invalid_ambiguous":
                if "previous reply was" not in user:
                    raw = raw + " " + raw
                else:
                    raw = "Here is my bid: " + raw
            response = {
                "model": request["model"],
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": raw}],
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }
            calls.append({"id": call_id, "request": request, "response": response, "status": 200})
            payload = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("X-Softmax-Llm-Call-Id", call_id)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = None
    if use_stub:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
        threading.Thread(target=server.serve_forever, daemon=True).start()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    config_path = folder / "config.json"
    config_path.write_text(json.dumps(config))
    env = dict(os.environ)
    for key in (
        "COWORLD_LLM_ENDPOINT",
        "COWORLD_LLM_MODEL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEY_URI",
        "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
        "AWS_BEARER_TOKEN_BEDROCK",
    ):
        env.pop(key, None)
    env.update(
        {
            "COGAME_HOST": "127.0.0.1",
            "COGAME_PORT": str(port),
            "COGAME_CONFIG_URI": config_path.as_uri(),
            "COGAME_RESULTS_URI": (folder / "results.json").as_uri(),
            "COGAME_SAVE_REPLAY_URI": (folder / "replay.json").as_uri(),
            "COGAME_SAVE_TRAJECTORY_URI": (folder / "trajectory.jsonl").as_uri(),
            "COGAME_PLAYER_FAILURE_URI": (folder / "player_failure.json").as_uri(),
            "COWORLD_EPISODE_ID": "fixture-" + mode,
            "COWORLD_GAME_VERSION": "source-fixture",
            "COWORLD_SOURCE_REVISION": revision,
        }
    )
    if use_stub:
        env.update(
            {
                "COWORLD_LLM_ENDPOINT": f"http://127.0.0.1:{server.server_port}",
                "COWORLD_LLM_MODEL": "fixture/synthetic",
                "COWORLD_LLM_TEMPERATURE": "0",
            }
        )
    exit_code = None
    with (folder / "game.log").open("w") as log:
        process = subprocess.Popen([binary], env=env, stdout=log, stderr=subprocess.STDOUT)
        seats = []
        try:
            deadline = time.monotonic() + 10
            while True:
                probe = socket.socket()
                ready = probe.connect_ex(("127.0.0.1", port)) == 0
                probe.close()
                if ready:
                    break
                assert process.poll() is None, (folder / "game.log").read_text()
                assert time.monotonic() < deadline
                threading.Event().wait(0.02)
            for slot in range(len(config["players"])):
                seat = connect(f"ws://127.0.0.1:{port}/player?slot={slot}&token={slot}")
                seats.append(seat)
                seat.send(json.dumps({"type": "prompt", "prompt": "PRIVATE OPERATOR SENTINEL"}))
            deadline = time.monotonic() + 90
            if expect_abort:
                while process.poll() is None:
                    assert time.monotonic() < deadline, (folder / "game.log").read_text()
                    threading.Event().wait(0.05)
                exit_code = process.returncode
            else:
                while not all((folder / name).is_file() for name in ["results.json", "replay.json", "trajectory.jsonl"]):
                    assert process.poll() is None, (folder / "game.log").read_text()
                    assert time.monotonic() < deadline, (folder / "game.log").read_text()
                    threading.Event().wait(0.05)
        finally:
            for seat in seats:
                seat.close()
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
            if server is not None:
                server.shutdown()
                server.server_close()

    decisions = [json.loads(line) for line in (folder / "trajectory.jsonl").read_text().splitlines()]
    summary = decisions[-1]
    assert "PRIVATE OPERATOR SENTINEL" not in (folder / "game.log").read_text()
    assert "PRIVATE OPERATOR SENTINEL" in (folder / "trajectory.jsonl").read_text()
    assert (folder / "trajectory.jsonl").stat().st_mode & 0o777 == 0o600

    if expect_abort:
        assert exit_code != 0, f"expected non-zero exit for {mode}, got {exit_code}"
        assert not (folder / "results.json").exists(), f"aborted mode {mode} must not write results.json"
        assert not (folder / "replay.json").exists(), f"aborted mode {mode} must not write scored replay.json"
        assert (folder / "player_failure.json").is_file(), f"aborted mode {mode} must write player_failure.json"
        failure = json.loads((folder / "player_failure.json").read_text())
        assert failure["status"] == "failed", failure
        assert "scores" not in failure and "points" not in failure, failure
        assert summary["status"] == "failed", summary
        assert summary["participant_outcomes"] is None, summary
        assert "scores" not in summary["outcome"], summary
        expected_kind = "model" if mode == "invalid_ambiguous" else "infrastructure"
        expected_action_status = "rejected" if mode == "invalid_ambiguous" else "missing"
        assert failure["failure_kind"] == expected_kind, failure
        assert len(failure["failed_seats"]) == len(config["players"]), failure
        for decision in decisions[:-1]:
            assert decision["action_status"] == expected_action_status, decision
            assert decision["selected_attempt_id"] is None, decision
            assert decision["executed_action"] is None, decision
            assert decision["observation"]["failure_kind"] == expected_kind, decision
            if mode == "invalid_ambiguous":
                assert len(decision["attempts"]) == 2, decision
            elif mode == "provider_failure":
                assert len(decision["attempts"]) == 2, decision
            elif mode == "provider_auth_failure":
                assert len(decision["attempts"]) == 1, decision
            elif mode == "no_credentials":
                assert len(decision["attempts"]) == 0, decision
                assert "no LLM credentials" in decision["observation"]["failure_reason"]
        return {"mode": mode, "aborted": True, "failure_kind": expected_kind, "decisions": len(decisions) - 1, "calls": len(calls)}

    assert not (folder / "player_failure.json").exists()
    archive_path = folder / "provider-archive.jsonl"
    with archive_path.open("x") as archive:
        os.chmod(archive_path, 0o600)
        for call in calls:
            archive.write(
                json.dumps(
                    {
                        "platform_call_id": call["id"],
                        "caller_request": call["request"],
                        "provider_response": call["response"],
                        "response_status_code": call["status"],
                    }
                )
                + "\n"
            )
    results = json.loads((folder / "results.json").read_text())
    archived = {call["id"]: call for call in calls}
    for decision in decisions[:-1]:
        assert decision["attempts"], decision
        for attempt in decision["attempts"]:
            call = archived[attempt["platform_call_id"]]
            assert attempt["request"] == call["request"]
            assert attempt["raw_response"] == call["response"]
            assert attempt["prompt"][0]["content"] == call["request"]["system"]
            assert attempt["prompt"][1]["content"] == call["request"]["messages"][0]["content"]
        assert decision["action_status"] == "accepted"
        assert len(decision["attempts"]) == (2 if mode == "retry" else 1)
        assert decision["attempts"][-1]["parsed_action"] == decision["executed_action"]
    assert results["reason"] == "complete"
    assert results["fallbacks"] == [0, 0, 0, 0]
    assert summary["status"] == "completed"
    assert "PRIVATE OPERATOR SENTINEL" not in (folder / "replay.json").read_text()
    assert calls and all(
        "Budget pacing matters" in call["request"]["system"] for call in calls
    )
    report = subprocess.check_output(
        [
            sys.executable,
            "tools/export_native_posttrain.py",
            "--replay",
            str(folder / "replay.json"),
            "--trajectory",
            str(folder / "trajectory.jsonl"),
            "--game-log",
            str(folder / "game.log"),
            "--episode-id",
            "fixture-" + mode,
        ]
    )
    verified = json.loads(report)
    assert verified["full_schedule"] and verified["decisions"] == len(decisions) - 1
    return {"mode": mode, "aborted": False, "decisions": len(decisions) - 1, "calls": len(calls)}


root = Path(output).resolve()
root.mkdir(mode=0o700, parents=True, exist_ok=False)
config = {
    "seed": 17,
    "promptVariant": "mixed",
    "promptSeed": 4,
    "sampled": True,
    "turnDelayMs": 0,
    "player_connect_timeout_seconds": 3,
    "episodeTimeoutSeconds": 120,
    "maxOutputTokens": 900,
    "llmTimeoutSeconds": 5,
    "tokens": ["0", "1", "2", "3"],
    "players": [
        {"name": "fixture-0"},
        {"name": "fixture-1"},
        {"name": "fixture-2"},
        {"name": "fixture-3"},
    ],
    "mode": "goofspiel",
    "cards": 4,
    "maxRounds": 2,
    "batchSpacingSeconds": 1,
}
modes = [
    "accepted",
    "fenced",
    "retry",
    "invalid_ambiguous",
    "provider_failure",
    "provider_auth_failure",
    "no_credentials",
]
reports = [run_mode(root, mode, config) for mode in modes]
result = {"cohort": "synthetic native HTTP fixture; no checkpoint performance claim", "reports": reports}
(root / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
