# Running in containers

When running containers in your problems, `engibench` provides a container runtime abstraction, currently supporting
- 🐋 [docker](https://www.docker.com/)
- 🦭 [podman](https://podman.io/)
- [apptainer](https://apptainer.org/)

The following functions will use the first available container runtime in the order:
docker, podman, apptainer, or the runtime specified by the environment variable `CONTAINER_RUNTIME` if it is set:

```{eval-rst}
.. automethod:: engibench.utils.container::pull
```

```{eval-rst}
.. automethod:: engibench.utils.container::run
```

Alternatively the following runtimes can be used directly:

```{eval-rst}
.. autoclass:: engibench.utils.container::Docker
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: engibench.utils.container::Podman
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: engibench.utils.container::Apptainer
   :show-inheritance:
```

All above container runtimes share the same interface:

```{eval-rst}
.. autoclass:: engibench.utils.container::ContainerRuntime
   :members:
```
