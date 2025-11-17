# Airfoil

```{warning}
Currently, there is an [issue](https://github.com/IDEALLab/EngiBench/issues/180) when running this problem with [apptainer](../utils/container.md).
Depending on the apptainer configuration, there is a rather low storage limit for the working directory `/tmp/`
(see [the "Note" box about `--writable-tmpfs`](https://apptainer.org/docs/user/main/persistent_overlays.html#overview)).
When running the problem, this directory might get out of space.
This issue does not occur on [Euler](https://docs.hpc.ethz.ch/#what-is-euler).
On local machines, we recommend using [podman](https://podman.io/) or [docker](https://www.docker.com/) as the preferred runtime.
```

``` {problem} airfoil
```
