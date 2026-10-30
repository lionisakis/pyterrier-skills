Installing & Updating
=================================================

Install
-------------------------------------------------

Run the ``skills`` CLI from the project that should get the skills. It copies each skill into the agent's
folder (``.claude/skills/`` for Claude Code) and records the source in ``skills-lock.json``.

.. code-block:: console
   :caption: Install all skills

   $ npx skills add cmacdonald/pyterrier-skills --agent claude-code

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Effect
   * - ``--list``
     - show the skills without installing
   * - ``--skill pyterrier-basics --skill pyterrier-dense``
     - install only these skills (always include ``pyterrier-basics``)
   * - ``-g``
     - install for your user instead of the project
   * - ``--agent codex`` (or ``cursor``, ``gemini-cli``, ...)
     - install for another agent

Install from GitHub as above, not from a local checkout: ``npx skills update`` skips skills installed from a
local path.

Update
-------------------------------------------------

There is no registry: the GitHub repository is the release. ``check`` compares each installed skill with the
repository; ``update`` pulls the changed ones, including skills newly collected from extensions.

.. code-block:: console
   :caption: Check for and install updates

   $ npx skills check
   $ npx skills update -y

To update on every Claude Code session, add a ``SessionStart`` hook to the project's
``.claude/settings.json``:

.. code-block:: json
   :caption: .claude/settings.json

   {"hooks": {"SessionStart": [{"hooks": [{"type": "command",
     "command": "npx -y skills update -p -y >/dev/null 2>&1 || true"}]}]}}

One extension only
-------------------------------------------------

An extension that ships skills (see :doc:`authoring`) can also be installed on its own, straight from its
``pt_skills/`` folder on GitHub. Updates work the same way.

.. code-block:: console
   :caption: Install the skills of one extension

   $ npx skills add https://github.com/<owner>/<repo>/tree/main/<package>/pt_skills --agent claude-code

Troubleshooting
-------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Symptom
     - Cause
     - Fix
   * - ``npx skills check`` says "Local path, cannot be checked automatically"
     - installed from a checkout
     - reinstall with ``npx skills add cmacdonald/pyterrier-skills``
   * - install fails with an authentication error
     - no access to the GitHub repository
     - log in with ``gh auth login`` or set up git credentials for github.com
   * - the agent ignores the skills
     - installed for a different agent or scope
     - ``npx skills list`` shows where they are; reinstall with the right ``--agent`` or ``-g``
