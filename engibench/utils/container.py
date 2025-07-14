"""Abstraction over container runtimes."""

from collections.abc import Sequence
import os, sys
import subprocess
import re


def pull(image: str) -> None:
    """Pull an image using the selected runtime.

    Args:
        image: Container image to pull.
    """
    if RUNTIME is None:
        msg = "No container runtime found"
        raise FileNotFoundError(msg)

    RUNTIME.pull(image)


def run(
    command: list[str],
    image: str,
    mounts: Sequence[tuple[str, str]] = (),
    env: dict[str, str] | None = None,
    name: str | None = None,
) -> None:
    """Run a command in a container using the selected runtime.

    Args:
        command: Command (as a list of strings) to run inside the container.
        image: Container image to use.
        mounts: Pairs of host folder and destination folder inside the container.
        env: Mapping of environment variable names and values to set inside the container.
        name: Optional name for the container (not supported by all runtimes).
    """
    if RUNTIME is None:
        msg = "No container runtime found. Please ensure Docker, Podman, or Singularity is installed and running."
        raise FileNotFoundError(msg)

    try:
        result = RUNTIME.run(command, image, mounts, env, name)
        result.check_returncode()
    except subprocess.CalledProcessError as e:
        msg = f"Container command failed with exit code {e.returncode}:\nCommand: {' '.join(command)}\nOutput: {e.output.decode() if e.output else 'No output'}"
        raise RuntimeError(msg) from e


class ContainerRuntime:
    """Abstraction over container runtimes."""

    name: str
    executable: str

    @classmethod
    def is_available(cls) -> bool:
        """Check if the container runtime is installed and executable.

        Returns:
            `True` if the container runtime appears to be installed on the system and if required daemons are running,
            `false` otherwise.
        """
        try:
            return (
                subprocess.run(
                    [cls.executable, "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
                ).returncode
                == 0
            )
        except FileNotFoundError:
            return False

    @classmethod
    def pull(cls, image: str) -> None:
        """Pull an image.

        Args:
            image: Container image to pull.
        """
        raise NotImplementedError("Must be implemented by a subclass")

    @classmethod
    def run(
        cls,
        command: list[str],
        image: str,
        mounts: Sequence[tuple[str, str]] = (),
        env: dict[str, str] | None = None,
        name: str | None = None,
    ) -> subprocess.CompletedProcess:
        """Run a command in a container.

        Args:
            command: Command (as a list of strings) to run inside the container.
            image: Container image to use.
            mounts: Pairs of host folder and destination folder inside the container.
            env: Mapping of environment variable names and values to set inside the container.
            name: Optional name for the container (not supported by all runtimes).
        """
        raise NotImplementedError("Must be implemented by a subclass")


def runtime() -> type[ContainerRuntime] | None:
    """Determine the container runtime to use according to the environment variable `CONTAINER_RUNTIME`.

    If not set, check for availability.

    Returns:
        Class object of the first available container runtime or the container runtime selected by the
        `CONTAINER_RUNTIME` environment variable if set.
    """
    runtimes_by_name = {rt.name: rt for rt in RUNTIMES}
    rt_name = os.environ.get("CONTAINER_RUNTIME")
    rt = runtimes_by_name.get(rt_name) if rt_name is not None else None
    if rt is not None:
        return rt
    for rt in RUNTIMES:
        if rt.is_available():
            return rt
    return None


class Docker(ContainerRuntime):
    """Docker 🐋 runtime."""

    name = "docker"
    executable = "docker"

    @classmethod
    def is_available(cls) -> bool:
        """Check if the container runtime is installed and executable.

        Returns:
            `True` if the container runtime appears to be installed on the system and if required daemons are running,
            `false` otherwise.
        """
        try:
            return (
                subprocess.run(
                    [cls.executable, "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
                ).returncode
                == 0
            )
        except FileNotFoundError:
            return False

    @classmethod
    def pull(cls, image: str) -> None:
        """Pull an image.

        Args:
            image: Container image to pull.
        """
        subprocess.run([cls.executable, "pull", image], check=True)

    @classmethod
    def run(
        cls,
        command: list[str],
        image: str,
        mounts: Sequence[tuple[str, str]] = (),
        env: dict[str, str] | None = None,
        name: str | None = None,
    ) -> subprocess.CompletedProcess:
        """Run a command in a container.

        Args:
            command: Command (as a list of strings) to run inside the container.
            image: Container image to use.
            mounts: Pairs of host folder and destination folder inside the container.
            env: Mapping of environment variable names and values to set inside the container.
            name: Optional name for the container (not supported by all runtimes).
        """
        name_args = [] if name is None else ["--name", name]
        mount_args = (["--mount", f"type=bind,src={src},target={target}"] for src, target in mounts)
        env_args = (["--env", f"{var}={value}"] for var, value in (env or {}).items())

        return subprocess.run(
            [
                cls.executable,
                "run",
                "--rm",
                *name_args,
                *(arg for args in mount_args for arg in args),
                *(arg for args in env_args for arg in args),
                image,
                *command,
            ],
            check=False,
        )


class Podman(Docker):
    """Podman 🦭 runtime."""

    name = "podman"
    executable = "podman"

    @classmethod
    def is_available(cls) -> bool:
        """Check if the container runtime is installed and executable.

        Returns:
            `True` if the container runtime appears to be installed on the system and if required daemons are running,
            `false` otherwise.
        """
        # `podman info` seems to take some more time than `docker info`.
        # Just use `podman --help` here.
        try:
            return (
                subprocess.run(
                    [cls.executable, "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
                ).returncode
                == 0
            )
        except FileNotFoundError:
            return False


class Singularity(ContainerRuntime):
    """Singularity / Apptainer."""

    #name = "singularity"
    #executable = "singularity"

    name = "apptainer"
    executable = "apptainer"

    @classmethod
    def sif_filename(cls, image: str) -> str:
        """Pull an image.

        Args:
            image: Container image to pull.
        """
        # Convert to docker URI if needed
        if "://" not in image:
            docker_uri = "docker://" + image
        else:
            docker_uri = image
            # Extract just the image part if it's already a docker URI
            if docker_uri.startswith("docker://"):
                image = docker_uri[len("docker://"):]

        # Parse the image name to match Singularity's naming convention
        # For "mdolab/public:u22-gcc-ompi-stable", Singularity creates "public_u22-gcc-ompi-stable.sif"
        image_name = image.split("/")[-1] if "/" in image else image

        # Replace ":" with "_" in the image name
        sif_filename = image_name.replace(":", "_") + ".sif"

        return sif_filename

    def setup_unique_container_workspace(base_path, study_id):
        """Create unique directories for each container instance."""
        
        # Create a container-specific workspace
        container_workspace = os.path.join(base_path, f".container_workspaces/study_{study_id}")
        
        # Create subdirectories for common conflict points
        dirs_to_create = {
            'tmp': os.path.join(container_workspace, 'tmp'),
            'cache': os.path.join(container_workspace, 'cache'),
            'run': os.path.join(container_workspace, 'run'),
            'shm': os.path.join(container_workspace, 'shm'),
            'home_local': os.path.join(container_workspace, 'home/.local'),
        }
        
        for dir_name, dir_path in dirs_to_create.items():
            os.makedirs(dir_path, exist_ok=True)
            
        return container_workspace, dirs_to_create


    @classmethod
    def pull(cls, image: str) -> None:
        """Pull an image.

        Args:
            image: Container image to pull.
        """
        # Get sif filename
        sif_filename = sif_filename(image)

        # Check if the image already exists
        if os.path.exists(sif_filename):
            print(f"Image file already exists: {sif_filename} - skipping pull")
            return
        else:
            # Image doesn't exist, proceed with pull
            subprocess.run([cls.executable, "pull", docker_uri], check=True)

    @classmethod
    def run(
        cls,
        command: list[str],
        image: str,
        mounts: Sequence[tuple[str, str]] = (),
        env: dict[str, str] | None = None,
        _name: str | None = None,
    ) -> subprocess.CompletedProcess:
        """Run a command in a container.

        Args:
            command: Command (as a list of strings) to run inside the container.
            image: Container image to use.
            mounts: Pairs of host folder and destination folder inside the container.
            env: Mapping of environment variable names and values to set inside the container.
            name: Optional name for the container (not supported by all runtimes).
        """
        # Get sif filename
        sif_image = cls.sif_filename(image)

        # Fallback - try the docker base dir or bash command
        #study_rel_path = "/" + command[2].split("/engibench/")[1].rsplit("/", 1)[0]
        match = re.search(r'/engibench/(.*?/study_\d+)', command[2])
        if match:
            study_rel_path = "/" + match.group(1)
            study_id = study_rel_path.split('_')[-1]
        study_abs_path = mounts[0][0]+study_rel_path
        print(study_id)
        print(study_rel_path)
        print(study_abs_path)
        print(env)

        # HPC/Singularity containers require explicit /tmp mounting to prevent memory issues
        # and ensure application compatibility. This is container configuration, not insecure temp file creation.
        mounts.append((mounts[0][0], "/tmp")) # noqa: S108

        mount_args = (["--mount", f"type=bind,src={src},target={target}"] for src, target in mounts)
        #mount_args = (["--mount", f"type=bind,src={study_abs_path},target={target}"] for src, target in mounts)
        env_args = (["--env", f"{var}={value}"] for var, value in (env or {}).items())

        mount_args_list = list(mount_args)

        for args in mount_args_list:
            print(f"  {args}")
        sys.exit('Inside container.py run method')

        if "://" not in image:
            image = "docker://" + image
        return subprocess.run(
            [
                cls.executable,
                "run",
                "--compat",
                *(arg for args in mount_args for arg in args),
                *(arg for args in env_args for arg in args),
                sif_image,
                *command,
            ],
            check=False,
        )


RUNTIMES = [
    rt
    for rt in globals().values()
    if isinstance(rt, type) and issubclass(rt, ContainerRuntime) and rt is not ContainerRuntime
]


RUNTIME = runtime()
