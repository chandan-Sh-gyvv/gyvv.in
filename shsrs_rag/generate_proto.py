"""
generate_proto.py
=================
Run once to generate gRPC stubs.
Usage: python generate_proto.py
"""
import sys
import os
from pathlib import Path

proto_dir   = Path("proto")
out_dir     = Path("shsrs_rag/proto_gen")
proto_file  = proto_dir / "shsrs_rag.proto"

out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "__init__.py").touch()

# Try grpc_tools first
try:
    from grpc_tools import protoc
    ret = protoc.main([
        "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        str(proto_file),
    ])
    if ret == 0:
        print("✓ Stubs generated via grpc_tools")
        sys.exit(0)
    else:
        print(f"grpc_tools returned {ret}, trying subprocess...")
except Exception as e:
    print(f"grpc_tools failed: {e}, trying subprocess...")

# Fallback: subprocess with explicit python path
import subprocess
result = subprocess.run([
    sys.executable, "-m", "grpc_tools.protoc",
    f"-I{proto_dir}",
    f"--python_out={out_dir}",
    f"--grpc_python_out={out_dir}",
    str(proto_file),
], capture_output=True, text=True)

if result.returncode == 0:
    print("✓ Stubs generated via subprocess")
else:
    print(f"✗ Failed: {result.stderr}")
    print("\nManual fix: install an older grpcio-tools compatible with Python 3.13")
    print("pip install grpcio-tools==1.60.0")
    sys.exit(1)
