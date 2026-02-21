"""Container image build/start/stop for SWE-bench instances.

Uses podman or docker via the SWE-bench harness build chain. The container
runtime is configurable via environment variables:
  DOCKER_HOST   -- socket path (default: unix:///run/user/1000/podman/podman.sock)
  CONTAINER_CMD -- CLI binary name (default: podman)
"""

from __future__ import annotations

import logging
import os
import subprocess

log = logging.getLogger(__name__)

# Container socket for docker-py compatibility (used by swebench build system)
CONTAINER_SOCKET = os.environ.get(
    "DOCKER_HOST",
    "unix:///run/user/1000/podman/podman.sock",
)

# CLI command for container operations (podman or docker)
CONTAINER_CMD = os.environ.get("CONTAINER_CMD", "podman")


class ContainerExecutor:
    """Execute agent tools inside a running container."""

    def __init__(self, container_id: str, workdir: str = "/testbed"):
        self.cid = container_id
        self.workdir = workdir

    def __call__(self, name: str, args: dict) -> str:
        """Dispatch a tool call to the container."""
        if name == "read_file":
            path = args.get("path", "")
            return self._exec(f"cat {self.workdir}/{path}")

        elif name == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")
            full_path = f"{self.workdir}/{path}"
            self._exec(f"mkdir -p $(dirname {full_path})")
            self._exec(f"tee {full_path} > /dev/null", input=content)
            return f"OK -- wrote {len(content)} chars to {path}"

        elif name == "list_files":
            path = args.get("path", ".")
            return self._exec(f"ls {self.workdir}/{path}")

        elif name == "run_command":
            cmd = args.get("command", "")
            return self._exec(f"cd {self.workdir} && {cmd}", timeout=120)

        return f"Unknown tool: {name}"

    def _exec(self, cmd: str, *, input: str | None = None, timeout: int = 30) -> str:
        """Run a command inside the container."""
        wrapped = f"source /opt/miniconda3/bin/activate testbed && {cmd}"
        try:
            result = subprocess.run(
                [CONTAINER_CMD, "exec", "-i", self.cid, "bash", "-c", wrapped],
                input=input,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = (result.stdout + result.stderr).strip()
            if len(output) > 5000:
                output = output[:5000] + "\n...(truncated)"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: command timed out ({timeout}s)"
        except Exception as e:
            return f"Error: {e}"

    def get_patch(self) -> str:
        """Extract git diff of the agent's changes."""
        result = self._exec(f"cd {self.workdir} && git diff HEAD", timeout=10)
        if result == "(no output)":
            return ""
        return result


def build_instance_image(instance: dict) -> str:
    """Build the SWE-bench instance image. Returns image name."""
    import docker
    from swebench.harness.docker_build import (
        build_base_images,
        build_env_images,
        build_instance_images,
    )

    client = docker.DockerClient(base_url=CONTAINER_SOCKET)
    dataset = [instance]

    log.info("Building images for %s...", instance["instance_id"])
    build_base_images(client, dataset, instance_image_tag="latest", env_image_tag="latest")
    build_env_images(client, dataset, max_workers=1, instance_image_tag="latest", env_image_tag="latest")
    build_instance_images(client, dataset, max_workers=1, tag="latest", env_image_tag="latest")

    from swebench.harness.test_spec.test_spec import make_test_spec
    spec = make_test_spec(instance)
    image_name = spec.instance_image_key
    log.info("Image ready: %s", image_name)
    return image_name


def start_container(image: str) -> str:
    """Start a container from a locally-built image. Returns container ID."""
    log.info("Starting container from: %s (runtime: %s)", image, CONTAINER_CMD)
    run = subprocess.run(
        [CONTAINER_CMD, "run", "-d", "--rm", image, "sleep", "infinity"],
        capture_output=True, text=True, timeout=30,
    )
    if run.returncode != 0:
        raise RuntimeError(f"{CONTAINER_CMD} run failed: {run.stderr.strip()}")

    cid = run.stdout.strip()
    log.info("Container started: %s", cid[:12])
    return cid


def stop_container(cid: str) -> None:
    """Stop a running container."""
    log.info("Stopping container: %s", cid[:12])
    try:
        subprocess.run(
            [CONTAINER_CMD, "stop", "-t", "5", cid],
            capture_output=True, timeout=30,
        )
    except Exception as e:
        log.warning("Failed to stop container %s: %s", cid[:12], e)
