Writing Skills for an Extension
=================================================

An extension package can ship agent skills the same way it ships documentation. PyTerrier's docs build copies
each extension's ``pt_docs/`` folder into these pages; ``pyterrier-skills`` copies each extension's
``pt_skills/`` folder into the skill set that users install. The skill then ships with the version of the
code it describes.

**Step 1: Add a pt_skills folder to your package**

Put one folder per skill inside the package, next to ``pt_docs/``:

.. code-block:: text
   :caption: Package layout

   pyterrier_myext/
     __init__.py
     pt_docs/
       index.rst
     pt_skills/
       pyterrier-myext/
         SKILL.md
         references/      # optional: long API tables and examples
         scripts/         # optional

The folder name is the skill name. A ``SKILL.md`` starts with YAML frontmatter:

.. code-block:: markdown
   :caption: pt_skills/pyterrier-myext/SKILL.md

   ---
   name: pyterrier-myext
   description: Builds and searches MyExt indexes in PyTerrier. Use when the user wants ... Not for ...
   compatibility: pyterrier>=1.1 (checked with 1.1.2, pyterrier-myext 0.3.0)
   ---

   # MyExt in PyTerrier

   Assumes the `pyterrier-basics` skill.

   ```python run
   import pyterrier as pt
   ...
   ```

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field / rule
     - Requirement
   * - ``name``
     - equal to the folder name; lowercase letters, digits and single hyphens; at most 64 characters
   * - ``description``
     - at most 1024 characters, third person, written as *when* to use the skill, with a "Not for ..." clause
       where another skill is a near miss
   * - ``compatibility``
     - the versions the examples were checked against
   * - length
     - ``SKILL.md`` under 500 lines; move detail into ``references/`` and say when to read each file
   * - examples
     - give code fences that must run the info string ``python run`` (``python run slow`` for large
       downloads or a GPU); the ``pyterrier-skills`` test suite executes them. Plain ``python`` blocks must
       still parse
   * - other skills
     - refer to them by name (``the pyterrier-dense skill``), never by a path

Use a skill name that no other extension uses. If it matches a skill in ``pyterrier-skills`` (for example
``pyterrier-splade``), your copy replaces it for every user.

**Step 2: Make sure the folder is in your wheel**

``scikit-build-core`` and ``hatchling`` include every file under the package directory. With ``setuptools``,
declare it:

.. code-block:: toml
   :caption: pyproject.toml (setuptools only)

   [tool.setuptools.package-data]
   "*" = ["pt_docs/**/*", "pt_skills/**/*"]

**Step 3: List the package in extensions.txt**

Skills are collected from the same
`extensions.txt <https://github.com/terrier-org/pyterrier/blob/master/docs/extensions.txt>`__ the docs use.
If your package is already listed with its GitHub URL, there is nothing to do:

.. code-block:: text

   package_id # PackageName <https://github.com/owner/repo>

Each week the collector lists every repository in that file, copies each ``pt_skills/<name>/`` folder from
the default branch, and opens a pull request on ``pyterrier-skills``. Once it is merged, users get the skill
with ``npx skills update``. To try your skills before that, install them directly from your repository (see
:doc:`usage`).
