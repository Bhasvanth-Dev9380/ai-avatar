"""Upload files to RunPod pod via SSH using chunked base64 transfer."""
import subprocess
import base64
import sys
import os

SSH_KEY = r"C:\Users\Veerendar\.ssh\id_ed25519"
SSH_HOST = "root@157.157.221.29"
SSH_PORT = "29659"

def ssh_cmd(command: str) -> str:
    """Run a command on the pod via SSH."""
    result = subprocess.run(
        ["ssh", "-T", "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY, 
         "-p", SSH_PORT, SSH_HOST, command],
        capture_output=True, text=True, timeout=60
    )
    return result.stdout + result.stderr

def upload_file(local_path: str, remote_path: str):
    """Upload a file to the pod using chunked base64 via SSH echo commands."""
    with open(local_path, "rb") as f:
        content = f.read()
    
    b64 = base64.b64encode(content).decode("ascii")
    chunk_size = 3000  # safe for SSH command line
    chunks = [b64[i:i+chunk_size] for i in range(0, len(b64), chunk_size)]
    
    print(f"Uploading {local_path} ({len(content)} bytes, {len(chunks)} chunks) -> {remote_path}")
    
    # Write first chunk (truncate)
    out = ssh_cmd(f"echo -n '{chunks[0]}' > /tmp/upload.b64 && echo CHUNK_0_OK")
    if "CHUNK_0_OK" not in out:
        print(f"ERROR on chunk 0: {out}")
        return False
    print(f"  chunk 0/{len(chunks)-1} OK")
    
    # Append remaining chunks
    for i, chunk in enumerate(chunks[1:], 1):
        out = ssh_cmd(f"echo -n '{chunk}' >> /tmp/upload.b64 && echo CHUNK_{i}_OK")
        if f"CHUNK_{i}_OK" not in out:
            print(f"ERROR on chunk {i}: {out}")
            return False
        if i % 2 == 0 or i == len(chunks) - 1:
            print(f"  chunk {i}/{len(chunks)-1} OK")
    
    # Decode and move to destination
    out = ssh_cmd(f"base64 -d /tmp/upload.b64 > {remote_path} && wc -c {remote_path} && echo DECODE_OK")
    if "DECODE_OK" not in out:
        print(f"ERROR decoding: {out}")
        return False
    
    print(f"  => {out.strip()}")
    return True

if __name__ == "__main__":
    files = [
        (r"C:\Users\Veerendar\Desktop\ai avatar\musetalk_server_v2.py", "/workspace/ai-avatar/musetalk_server_v2.py"),
        (r"C:\Users\Veerendar\Desktop\ai avatar\avatar_agent.py", "/workspace/ai-avatar/avatar_agent.py"),
    ]
    
    for local, remote in files:
        if not os.path.exists(local):
            print(f"SKIP: {local} not found")
            continue
        ok = upload_file(local, remote)
        if not ok:
            print(f"FAILED: {local}")
            sys.exit(1)
        print()
    
    # Verify
    print("Verifying files on pod...")
    out = ssh_cmd("ls -la /workspace/ai-avatar/musetalk_server_v2.py /workspace/ai-avatar/avatar_agent.py && head -3 /workspace/ai-avatar/musetalk_server_v2.py && echo VERIFY_OK")
    print(out)
