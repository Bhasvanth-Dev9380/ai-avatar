"""Upload avatar_agent.py to RunPod pod."""
import subprocess
import base64
import sys
import os
import time

SSH_KEY = r"C:\Users\Veerendar\.ssh\id_ed25519"
SSH_HOST = "root@157.157.221.29"
SSH_PORT = "29659"

def ssh_cmd(command: str, timeout: int = 60) -> str:
    result = subprocess.run(
        ["ssh", "-T", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
         "-i", SSH_KEY, "-p", SSH_PORT, SSH_HOST, command],
        capture_output=True, text=True, timeout=timeout
    )
    return result.stdout + result.stderr

local_path = r"C:\Users\Veerendar\Desktop\ai avatar\avatar_agent.py"
remote_path = "/workspace/ai-avatar/avatar_agent.py"

with open(local_path, "rb") as f:
    content = f.read()

b64 = base64.b64encode(content).decode("ascii")
chunk_size = 2000  # smaller chunks to avoid timeout
chunks = [b64[i:i+chunk_size] for i in range(0, len(b64), chunk_size)]

print(f"Uploading {local_path} ({len(content)} bytes, {len(chunks)} chunks)")

# Clear temp file first
ssh_cmd("rm -f /tmp/upload.b64 && echo CLEAR_OK")

for i, chunk in enumerate(chunks):
    op = ">" if i == 0 else ">>"
    retries = 3
    for attempt in range(retries):
        try:
            out = ssh_cmd(f"echo -n '{chunk}' {op} /tmp/upload.b64 && echo CHUNK_{i}_OK")
            if f"CHUNK_{i}_OK" in out:
                if i % 3 == 0 or i == len(chunks) - 1:
                    print(f"  chunk {i}/{len(chunks)-1} OK")
                break
            else:
                print(f"  chunk {i} attempt {attempt+1} bad response: {out[:100]}")
        except subprocess.TimeoutExpired:
            print(f"  chunk {i} attempt {attempt+1} timeout, retrying...")
            time.sleep(2)
    else:
        print(f"FAILED on chunk {i} after {retries} attempts")
        sys.exit(1)

# Decode
out = ssh_cmd(f"base64 -d /tmp/upload.b64 > {remote_path} && wc -c {remote_path} && echo DECODE_OK")
if "DECODE_OK" in out:
    print(f"SUCCESS: {out.strip()}")
else:
    print(f"DECODE FAILED: {out}")
    sys.exit(1)

# Verify
print("\nVerifying both files...")
out = ssh_cmd("ls -la /workspace/ai-avatar/musetalk_server_v2.py /workspace/ai-avatar/avatar_agent.py && head -5 /workspace/ai-avatar/musetalk_server_v2.py && echo '---' && head -5 /workspace/ai-avatar/avatar_agent.py && echo VERIFY_OK")
print(out)
