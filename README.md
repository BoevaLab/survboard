### Suggested setup for development

Create a `virtualenv`:

```sh
python -m venv venv
```

Activate it:

```sh
source venv/bin/activate
```

Install the package (including required packages) as "editable", meaning changes to your code do not require another installation to update.

```sh
pip install -e ".[dev]"
# or `pip install -e ".[vcs,dev]"`  # if you rely on other packages from github
```

To ensure exact versions:
```sh
pip install -r requirements.txt
pip install -r dev_requirements.txt
pip install -e .
```

### Experimental files
Have a look at the [bench](bench/README.md) to keep your work that is not part of a
package under version control.

#### Formatter `black`

When writing code, you should **not** have to worry about how to format it best.
When committing code, it should be formatted in one specific way that reduces meaningless diff changes.  
This is achieved by using an agreed upon formatter in the project. Here, [`black`](https://github.com/psf/black) is recommended (with fixed version).
You can [set up your IDE](https://black.readthedocs.io/en/stable/editor_integration.html) to format your code on save, or add a [pre-commit hook](https://black.readthedocs.io/en/stable/version_control_integration.html).