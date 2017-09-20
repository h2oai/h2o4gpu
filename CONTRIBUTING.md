# Contributing guidelines

This document is a set of guidelines for contributors, nothing is set in stone so please just use common sense.

### Table of contents

[Code of conduct](#code-of-conduct)

[Other important resources](#other-important-documents)

[How to contribute](#how-to-contribute)

* [Reporting bugs](#reporting-bugs)
* [Enhancements and Feature requests](#enhancements-and-feature-requests)
* [Committing code](#committing-code)

[Guidelines and standards](#guidelines-and-standards)

* [General](#general)
* [Git commit messages](#git-commit-messages)
* [License](#license)
* [C style guidelines](#c-style-guidelines)
* [Python style guidelines](#python-style-guidelines)
* [R style guidelines](#r-style-guidelines)
* [Documentation style guidelines](#documentation-style-guidelines)
* [Sklearn Override](#sklearn-override)

## Code of conduct

First and foremost please read our [Code of Conduct](CODE_OF_CONDUCT.md).

## Other important documents

* For questions please reach out to us on our [Gitter channel](https://gitter.im/h2oai/h2o4gpu?utm_source=share-link&utm_medium=link&utm_campaign=share-link) or post questions on [StackOverflow](https://stackoverflow.com) with a `h2o4gpu` tag. Please do not use the issue tracker to ask questions.
* Roadmap can be found [in the readme file](README.md#plans-and-roadmap)
* For build instructions etc. please refer to our [developer](DEVELOPER.md) guide.

## How to contribute

There are plenty of ways to contribute, not only by submitting new code! Reporting bugs, adding/improving documentation and tests is just as important to us.

### Reporting bugs

* First, please make sure the bug was not already reported by searching on GitHub under [issues](https://github.com/h2oai/h2o4gpu/issues).
* Only when you are sure the bug has not yet been reported, [open a new issue](https://github.com/h2oai/h2o4gpu/issues/new).
* Please follow the [issue template](ISSUE_TEMPLATE.md) and provide as many details as possible. A clear, but concise, title, your environment, h2o4gpu version and a [MCVE](https://stackoverflow.com/help/mcve) will make everyone's life easier.

### Enhancements and Feature requests

We are open to new enhancements/feature requests!

* Just like with bugs, please check whether there is already an existing issue in the tracker.
* If not then you can reach out to us on [Gitter channel](https://gitter.im/h2oai/h2o4gpu?utm_source=share-link&utm_medium=link&utm_campaign=share-link) and start a conversation or
create a new issue with an appropriate label (`feature request` or `enhancement`).
* Use a descriptive, but concise, title.
* Start of by stating what would this improvement fix/enhance and why would it be useful.
* Provide as detailed description as possible (but do not write a novel).
* Preferably add a specific example.
* If necessary (or you just think it would be beneficial) add drawings, diagrams etc.

### Committing code

Should you want to improve the codebase, please submit a pull request. If you are new to GitHub check their [how-to](https://help.github.com/articles/using-pull-requests/) first.

For outside/new contributors we try to mark issues with an appropriate label so have a look at [issues with the "beginner friendly" label](https://github.com/h2oai/h2o4gpu/issues?q=is%3Aopen+is%3Aissue+label%3A%22beginner+friendly%22).
Please be sure to comment on an issue should you decide to give it a go so other developers know about it. All issue related discussion should take place in the issue's comment section.

Before submitting your PR for a [review](https://github.com/h2oai/h2o4gpu/pulls) please make sure your changes follow the standards described below, are well tested and all the previous tests are passing.

Check our [developer](DEVEL.md) guide for build and testing instructions.

### Guidelines and standards

#### General

* Test your code. Be it a bug fix, enhancement or a new feature please provide a set of tests showing 1) that it actually runs 2) does what it is supposed to be doing 3) does not break existing codebase
* Keep in mind performance - runtime speed, resource use and accuracy are important to us.
* API breaking changes/fixes will need an extended discussion.
* If you add new features or add a non trivial piece of code please do document it if needed.

#### Git commit messages

Clean and descriptive commit messages keep the maintainers happy so please take a minute to polish yours.

Preferable message structure (from: http://git-scm.com/book/ch5-2.html):

> ```
> Short (50 chars or less) summary of changes
> 
> More detailed explanatory text, if necessary.  Wrap it to about 72
> characters or so.  In some contexts, the first line is treated as the
> subject of an email and the rest of the text as the body.  The blank
> line separating the summary from the body is critical (unless you omit
> the body entirely); tools like rebase can get confused if you run the
> two together.
> 
> Further paragraphs come after blank lines.
> 
>   - Bullet points are okay, too
> 
>   - Typically a hyphen or asterisk is used for the bullet, preceded by a
>     single space, with blank lines in between, but conventions vary here
> ```

* Keep the initial line short (50 characters or less).
* Use imperative mode in the first line i.e. "Fix", "Add", "Change" instead of "Fixed", "Added", "Changed".
* Second line blank.
* Add issue/pull request number in the message.
* When fixing typos and documentation include [ci skip] in the commit message (to be implemented).
* Don't end the summary line with a period.
* If you are having trouble describing what a commit does, make sure it does not include several logical changes or bug fixes.
Should that be the case please split it up into several commits using `git add -p`.

#### License

Each file has to include a license at the top.

* [C/C++ license example](https://github.com/h2oai/h2o4gpu/blob/master/src/cpu/include/cgls.h#L1)
* [Python license example](https://github.com/h2oai/h2o4gpu/blob/master/src/interface_py/h2o4gpu/solvers/base.py#L1)
* [R license example](https://github.com/h2oai/h2o4gpu/blob/master/src/interface_r/h2o4gpu/R/h2o4gpuglm.R#L1)
* [Bash license example](https://github.com/h2oai/h2o4gpu/blob/master/scripts/gitshallow_submodules.sh#L1)

#### C style guidelines

For C/C++/CUDA code please follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

On Unix based systems you can use `clang-tidy` to make sure your changes are ok. To install it on Ubuntu 16.04, do:

```
apt-get install -y clang-tidy
```

To check a single file, run:

```
clang-format <my_cc_file> --style=google > /tmp/my_cc_file.cc
diff <my_cc_file> /tmp/my_cc_file.cc
```

#### Python style guidelines

For Python code please follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

You can use `pylint` to make sure your changes are ok. To install it and use H2O4GPU's style definition, run:

```
pip install pylint
wget -O /tmp/pylintrc https://raw.githubusercontent.com/h2oai/h2o4gpu/master/tools/pylintrc
```

The `pylintrc` file can be found in the repository under `tools/pylintrc`.

To check a single file, run:

```
pylint --rcfile=/tmp/pylintrc <my_python_file>.py
```

To check all python files, do:

```
cd src/interface_py ; make pylint
```

To auto-format to some python style standards, do:

```
make pyformat
```

#### Documentation style guidelines

TBA.

#### Sklearn Override

In order to override the sklearn API with new (GPU) functions, one:

* Adds string replacements in scripts/prepare_sklearn.sh so sklearn
  class is renamed to <origin name>Sklearn

* Adds the override as an echo that appends to __init__.py for whatever
packages or class is overriden.

* Add wrapper that choses between new class and sklearn class.
E.g. use sklearn if arguments passed suggest advanced feature h2o4gpu
cannot yet handle, or continue with h2o4gpu class if parameter being
passed is not important or is related to an inferior algorithm option.
Can follow class LogisticRegression and class KMeans as examples.

This exposes the simplest cases of how to override sklearn classes
with our own, and then how we write our own class in smart way is up
to us.  One can have a class that inherits the sklearn class, or one
can ignore the sklearn completely, or one can use the sklearn class as
a backup when functionality is missing.
