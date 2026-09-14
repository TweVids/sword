"""
Sword Coding Gym Server CLI.
Hosts local Docker execution and exposes a public zrok endpoint for Molab / Colab.

Usage:
  python serve_gym.py --port 8765 --share-zrok
  python serve_gym.py --port 8765 --dataset-path path/to/dataset.jsonl --share-zrok
"""

import argparse
import sys
import os
import time
import signal
import logging

import sword
from sword.gym.docker_gym import DockerCodingGym
from sword.gym.server import GymServer, ZrokTunnelManager
from sword.gym.adapters import load_swe_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sword.serve_gym")


def main():
    parser = argparse.ArgumentParser(description="Sword Docker Coding Gym Server & zrok Tunnel")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765)")
    parser.add_argument("--pool-size", type=int, default=2, help="Warm Docker container slots (default: 2)")
    parser.add_argument("--image", type=str, default="sword-coding-gym:latest", help="Docker testbed image")
    parser.add_argument("--dataset-path", type=str, default=None, help="Optional JSONL dataset to seed problem queue")
    parser.add_argument("--share-zrok", action="store_true", default=True, help="Enable public zrok tunnel (default: True)")
    parser.add_argument("--no-share-zrok", action="store_false", dest="share_zrok", help="Disable zrok tunnel")
    parser.add_argument("--endpoint-file", type=str, default="gym_endpoint.txt", help="File to write active public URL")
    parser.add_argument("--state-file", type=str, default="gym_state.json", help="Persistent JSON file to store queue & results for resumption")
    args = parser.parse_args()

    print("=" * 72)
    print(" 🐳 STARTING SWORD DOCKER CODING GYM SERVER")
    print(f" Testbed Image:  {args.image}")
    print(f" Container Pool: {args.pool_size} warm slots")
    print(f" Local Port:     {args.port}")
    print(f" State File:     {args.state_file}")
    print("=" * 72)

    # 1. Initialize Docker Coding Gym
    gym = DockerCodingGym(
        default_image=args.image,
        pool_size=args.pool_size,
    )
    if not gym.client:
        logger.error("❌ Docker daemon is not accessible! Make sure Docker Desktop is running.")
        sys.exit(1)

    # 2. Optionally load dataset instances
    instances = []
    if args.dataset_path:
        logger.info(f"Loading SWE dataset from {args.dataset_path}...")
        instances = load_swe_dataset(args.dataset_path)
        logger.info(f"Loaded {len(instances):,} instances from dataset.")

    # 3. Start HTTP Server with persistent state support
    server = GymServer(
        (args.host, args.port),
        gym=gym,
        instances=instances,
        state_file=args.state_file,
    )
    if os.path.exists(args.state_file):
        print(f"[*] Resumed gym state from {args.state_file}: {len(server.instances_queue)} queued, {len(server.completed_problems)} completed.")
    server_thread = None

    # 4. Start zrok tunnel if requested
    tunnel_mgr = None
    public_url = None
    if args.share_zrok:
        try:
            # Check for zrok2 or zrok binary
            zrok_bin = "zrok2"
            tunnel_mgr = ZrokTunnelManager(target_port=args.port, zrok_cmd=zrok_bin)
            public_url = tunnel_mgr.start()
            with open(args.endpoint_file, "w", encoding="utf-8") as f:
                f.write(public_url + "\n")
            print("\n" + "*" * 72)
            print(f" 🌐 PUBLIC ZROK ENDPOINT READY FOR MOLAB / COLAB:")
            print(f" 👉 {public_url}")
            print(f" 📁 Saved endpoint to: {args.endpoint_file}")
            print("*" * 72 + "\n")
        except Exception as e:
            logger.warning(f"⚠️ Could not establish zrok tunnel: {e}")
            logger.info("Serving only on local network.")

    def shutdown(sig, frame):
        print("\n[!] Shutting down Sword Gym Server...")
        if tunnel_mgr:
            tunnel_mgr.stop()
        server.shutdown()
        gym.close()
        print("[OK] Server shutdown complete.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"[*] Gym Server listening on http://{args.host}:{args.port}")
    print("[*] Ready to evaluate rollouts from Molab GRPO training loop. Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
