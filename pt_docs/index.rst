Agent Skills for PyTerrier
=================================================

`pyterrier-skills <https://github.com/cmacdonald/pyterrier-skills>`__ are agent skills that teach coding
agents (Claude Code, Codex, Cursor, Gemini CLI and others) to build PyTerrier 1.x pipelines and experiments:
retrieval, fusion, query and document expansion, reranking, RAG, caching and evaluation. They follow the open
`Agent Skills <https://agentskills.io/specification>`__ format: each skill is a ``SKILL.md`` with tested
examples, a list of common mistakes and a troubleshooting table.

You can install the skills into a project with the
`skills CLI <https://github.com/vercel-labs/skills>`__ (Node 22.20+):

.. code-block:: console
   :caption: Install the PyTerrier skills for Claude Code

   $ npx skills add cmacdonald/pyterrier-skills --agent claude-code

Extensions can ship their own skills in a ``pt_skills/`` folder, next to ``pt_docs/``. They are collected
into this set each week, so one install covers PyTerrier and its extensions.

.. toctree::
   :caption: Contents
   :maxdepth: 1

   Skills <skills>
   Installing & Updating <usage>
   Writing Skills for an Extension <authoring>
