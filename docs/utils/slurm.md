# Parameter space discovery via slurm

This module allows submitting a python callback as a slurm job array where each
individual job runs the callback with a different set of parameters.

```{testsetup} slurm
import os
from engibench.utils import slurm
import fake_sbatch
from example_callback import callback

slurm_args = slurm.SlurmConfig(sbatch_executable=fake_sbatch.__file__)
```

**Step 1:** Define a callback and the parameter space

```{literalinclude} example_callback.py
```

```{testcode} slurm
parameter_space = [
    {"config": {"arg": 1.0}, "a": 1, "b": "1"},
    {"config": {"arg": 2.0}, "a": 2, "b": "2"},
]
```

**Step 2:** Specify arguments to be passed to slurm:

```py
from engibench.utils import slurm
slurm_args = slurm.SlurmConfig(runtime="1:00:00")
```

**Step 3:** Submit a slurm job array

```{testcode} slurm

job = slurm.sbatch_map(
    callback,
    args=parameter_space,
    slurm_args=slurm_args,
)
```

For example this would submit a slurm array with 2 elements running

```py
callback(config={"arg": 1.0}, a=1, b=1) # element 1
callback(config={"arg": 2.0}, a=2, b=2) # element 2
```

By default, [sbatch_map()](#engibench.utils.slurm.sbatch_map) submits the job array in the background. That means that the execution flow of the python script will continue while the jobs are running.
If the calling python scripts needs to load results from the jobs, `wait=True` can be passed to [sbatch_map()](#engibench.utils.slurm.sbatch_map). In this case the call to [sbatch_map()](#engibench.utils.slurm.sbatch_map) will block until all jobs have completed.

Results of a submitted job also can be either "reduced" or saved automatically:

**Step 4a:** Reduce results from multiple job array elements

```{testcode} slurm

reduced = job.reduce(sum, slurm_args=slurm_args) # compute the sum of all result values

reduced # <- Evaluate `reduced` and display the result in the next cell
```

```{testoutput}
3
```

**Step 4b:** Save all results in one [pickle](https://docs.python.org/3/library/pickle.html) archive

```py
job.save("results.pkl", slurm_args=slurm_args)
```

[job.save()](#engibench.utils.slurm.SubmittedJobArray.save) will submit another slurm job using `slurm_args` (can be chosen independently from the arguments passed to [sbatch_map()](#engibench.utils.slurm.sbatch_map)) which will wait until all parallel jobs submitted by [sbatch_map()](#engibench.utils.slurm.sbatch_map) have completed.


The module contains a convenience wrapper around [pickle.load()](https://docs.python.org/3/library/pickle.html#pickle.load) to load result archives:

```py
results = slurm.load_results("results.pkl")
```

```{note}
In contrast to [sbatch_map()](#engibench.utils.slurm.sbatch_map), [job.save()](#engibench.utils.slurm.SubmittedJobArray.save) and [job.reduce()](#engibench.utils.slurm.SubmittedJobArray.reduce) are blocking.
The only reason that [sbatch_map()](#engibench.utils.slurm.sbatch_map) is non-blocking is to make it possible to chain this call with [job.save()](#engibench.utils.slurm.SubmittedJobArray.save)/ [job.reduce()](#engibench.utils.slurm.SubmittedJobArray.reduce).
If however `wait=True` is passed to [sbatch_map()](#engibench.utils.slurm.sbatch_map), the call will be also blocking, in which case chaining with [job.save()](#engibench.utils.slurm.SubmittedJobArray.save) or [job.reduce()](#engibench.utils.slurm.SubmittedJobArray.reduce) is not possible. 
```

## Technical details

The actual slurm job array will run instances of [engibench/utils/slurm/run_job.py](https://github.com/IDEALLab/EngiBench/blob/main/engibench/utils/slurm.py). The data to be processed will be serialized by [sbatch_map()](#engibench.utils.slurm.sbatch_map) and deserialized by [run_job.py](https://github.com/IDEALLab/EngiBench/blob/main/engibench/utils/slurm.py) using [pickle](https://docs.python.org/3/library/pickle.html).
This also includes the callable itself.
The pickle archive will contain the name of the module where the callable is defined together with the name of the callable.
If the callable is a bound method of a class instance, the arguments needed to reconstruct the class instance are serialized as well.
By default pickle only manages to deserialize callables defined in python modules which are reachable via `PYTHONPATH`.
To also allow python modules not reachable via `PYTHONPATH`, `engibench.utils.slurm` has an enhanced serializing mechanism (`engibench.utils.slurm.MemorizeModule`).
```{note}
As deserializing works via module name, callable name and arguments, the callable must be defined at the toplevel of the module and cannot be a python object which is returned by a factory function.
```
The execution flow is depicted in the [following figure](#slurm-execution-flow).
The execution flow of [sbatch_map()](#engibench.utils.slurm.sbatch_map) + [save()](#engibench.utils.slurm.SubmittedJobArray.save) is a bit simpler - it does not need the final deserializing step from slurm back to the calling script.


```{figure} slurm-execution-flow.svg
:alt: slurm execution flow
:name: slurm-execution-flow

Execution flow of [sbatch_map()](#engibench.utils.slurm.sbatch_map) followed by [reduce()](#engibench.utils.slurm.SubmittedJobArray.reduce)
```

## API

```{eval-rst}
.. autofunction:: engibench.utils.slurm::sbatch_map
```

```{eval-rst}
.. autoclass:: engibench.utils.slurm::SubmittedJobArray
   :members:
```

To tweak the arguments passed to sbatch, the `config` argument can be passed to `submit()`:

```{eval-rst}
.. autoclass:: engibench.utils.slurm.SlurmConfig
   :members:
```
