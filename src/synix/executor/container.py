"""Container-based tool executor for synix-bench.

Ported from hybrid-memory-bench/scripts/podman_exec.py with imports
adapted to the synix namespace. Executes agent tools inside a container
(podman or docker).

Container runtime is configurable via environment variables:
  DOCKER_HOST   -- socket path (default: unix:///run/user/1000/podman/podman.sock)
  CONTAINER_CMD -- CLI binary name (default: podman)

On RunPod (Docker-in-Docker):
  DOCKER_HOST=unix:///var/run/docker.sock CONTAINER_CMD=docker
"""

from __future__ import annotations

import logging
import os
import subprocess

from synix.core.errors import ContainerError

log = logging.getLogger(__name__)

# Container socket for docker-py compatibility (used by swebench build system)
CONTAINER_SOCKET = os.environ.get(
    "DOCKER_HOST",
    "unix:///run/user/1000/podman/podman.sock",
)

# CLI command for container operations (podman or docker)
CONTAINER_CMD = os.environ.get("CONTAINER_CMD", "podman")


class ContainerExecutor:
    """Execute agent tools inside a running container (podman or docker).

    Implements the ToolExecutor protocol defined in synix.executor.base.
    """

    def __init__(self, container_id: str, workdir: str = "/testbed") -> None:
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
            # Write via stdin, suppress tee echo
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
        """Run a command inside the container via podman/docker exec.

        All commands run inside the conda 'testbed' env so python/pytest
        resolve to the correct interpreter with all deps installed.
        """
        # Activate conda env before every command -- SWE-bench containers
        # install everything into conda env 'testbed'
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
            log.error("Container exec failed: %s", e)
            return f"Error: {e}"

    def get_patch(self) -> str:
        """Extract git diff of the agent's changes. Returns empty string if no changes."""
        result = self._exec(f"cd {self.workdir} && git diff HEAD", timeout=10)
        # _exec returns "(no output)" for empty stdout -- normalize to empty string
        if result == "(no output)":
            return ""
        return result


def build_instance_image(instance: dict) -> str:
    """Build the SWE-bench instance image via swebench + podman. Returns image name."""
    import docker
    from swebench.harness.docker_build import (
        build_base_images,
        build_env_images,
        build_instance_images,
    )

    client = docker.DockerClient(base_url=CONTAINER_SOCKET)
    dataset = [instance]

    # Build chain: base -> env -> instance (skips already-built layers)
    log.info("Building images for %s...", instance["instance_id"])
    build_base_images(client, dataset, instance_image_tag="latest", env_image_tag="latest")
    build_env_images(client, dataset, max_workers=1, instance_image_tag="latest", env_image_tag="latest")
    build_instance_images(client, dataset, max_workers=1, tag="latest", env_image_tag="latest")

    # Get the image name from the test spec
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
        raise ContainerError(f"{CONTAINER_CMD} run failed: {run.stderr.strip()}")

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
