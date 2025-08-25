## How to Contribute to EngiBench

EngiBench aims to simplify research and adoption of ML for engineering design. We warmly welcome contributions from the community. Here are some ways you can help:

### Reporting a Bug

* First, check if the bug has already been reported in our [GitHub issues](https://github.com/IDEALLab/EngiBench/issues).
* If not, [open a new issue](https://github.com/IDEALLab/EngiBench/issues/new) and include:
  * A clear **title and description** of the problem.
  * Relevant details: system info, Python version, EngiBench version.
  * A **code sample** or **executable test case** that reproduces the issue.

### Fixing a Bug

* Open a GitHub pull request (PR) with your patch.
* Ensure your PR description clearly explains the problem and your solution. Reference the related issue if applicable.
* Verify your code passes all required checks (see our [PR template](https://github.com/IDEALLab/EngiBench/blob/main/.github/PULL_REQUEST_TEMPLATE.md) for details).

### Documentation or Cosmetic Changes

* Open a PR for fixes such as typos, formatting, or minor documentation updates.
* If no code is changed, running the full test suite is not required (`ruff format` is still recommended).

### Adding a New Feature or Problem

Before starting development, **please reach out to us**:

* **Features**: Open an issue describing the feature, expected behavior, and proposed API. Keep in mind that features should scale to the general problem set; we do not include features tied to a specific simulator in the core API.
* **Problems**: Problems are carefully vetted before inclusion. You are welcome to use your own problem in your research, but we cannot maintain all submitted problems. You can still fork our repo and follow [this guide](https://engibench.ethz.ch/tutorials/new_problem/) to add your own problem. If you want to share an external problem, you can link it in our documentation by opening a PR that modifies [this file](https://github.com/IDEALLab/EngiBench/blob/main/docs/problems/external.md).
