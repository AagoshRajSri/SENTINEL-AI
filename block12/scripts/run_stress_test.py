import os
import subprocess
import sys
import time
import urllib.request
import json

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.dirname(_HERE)
PYTHON_EXE = r"C:\Users\aagos\AppData\Local\Python\bin\python.exe"
TARGET_PORT = 5000
SERVER_URL = f"http://localhost:{TARGET_PORT}"

def main():
    print("=" * 70)
    print("SENTINEL-AI: STARTING CONCURRENCY & STRESS TEST HARNESS")
    print("=" * 70)

    env = os.environ.copy()
    env["PORT"] = str(TARGET_PORT)

    # Start FastAPI server
    print(f"Launching Sentinel-AI server on {SERVER_URL}...")
    server = subprocess.Popen(
        [PYTHON_EXE, "app.py"],
        cwd=_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for server ready
        ready = False
        for attempt in range(30):
            try:
                with urllib.request.urlopen(f"{SERVER_URL}/metrics", timeout=2) as resp:
                    if resp.status == 200:
                        ready = True
                        print(f"Server is healthy and serving requests on port {TARGET_PORT}!")
                        break
            except Exception:
                time.sleep(1)

        if not ready:
            print("Server failed to start within 30 seconds.")
            out, err = server.communicate(timeout=5)
            print("STDOUT:", out.decode(errors="ignore"))
            print("STDERR:", err.decode(errors="ignore"))
            sys.exit(1)

        # Warm up request (allow up to 60s for cold model loading on Windows CPU)
        print("\nSending warm-up triage request to /classify (loading model into memory)...")
        req = urllib.request.Request(
            f"{SERVER_URL}/classify",
            data=json.dumps({"text": "Where is my delayed order?"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
            print(f"Warm-up response status: {resp.status}")
            print(f"Intent: {data.get('intent')} | Escalated: {data.get('should_escalate')}")
            print(f"Latency breakdown: {data.get('latency_breakdown')}")

        # Run k6 load test
        print("\nExecuting k6 load test (10 VUs for 30 seconds)...")
        k6_cmd = ["k6", "run", "load_test.js"]
        k6_proc = subprocess.run(k6_cmd, cwd=_ROOT, capture_output=True, text=True)
        print("\n" + "-" * 70)
        print("K6 TEST RESULTS:")
        print("-" * 70)
        print(k6_proc.stdout)
        if k6_proc.stderr:
            print("STDERR:", k6_proc.stderr)

        # Retrieve Prometheus metrics
        print("-" * 70)
        print("PROMETHEUS OBSERVABILITY RECEIPTS (/metrics):")
        print("-" * 70)
        with urllib.request.urlopen(f"{SERVER_URL}/metrics", timeout=5) as resp:
            metrics_text = resp.read().decode()
            for line in metrics_text.splitlines():
                if any(k in line for k in ["faiss_lookup_latency_ms", "llm_inference_latency_ms", "triage_request_latency_ms", "escalation_gate_tripped_total"]):
                    if not line.startswith("#"):
                        print(line)

    finally:
        print("\nShutting down server...")
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
        print("Server shutdown complete.")

if __name__ == "__main__":
    main()
