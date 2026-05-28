---
name: engineering-plan-ontology
description: Use this skill when generating, rewriting, or maintaining an engineering plan that should guide autonomous agents. It captures the ontology shared by strong plans across backend, mobile, rendering, and presentation repos: product thesis, anti-thesis, quality bar, fixed direction, code-informed baseline, hard and domain-specific contracts, architecture boundaries, phase-gated execution, durable artifacts, and verification-backed progress notes.
---

# Engineering Plan Ontology

Use this skill when the user wants an engineering plan that is durable,
agent-operable, concise, and strong enough to drive implementation across many
turns.

The model to follow is not a loose roadmap. It is an executable constitution for
the repo.

## What this skill is for

- Writing a new `engineering-plan.md` or `ENGINEERING_PLAN.md`
- Rewriting a vague plan into a sharper operational document
- Updating a plan after tasks complete
- Determining what must be clarified before a real plan can be written
- Distilling product direction before implementation starts
- Converting architectural intuitions into explicit dependency rules
- Turning agent handoff lessons into ordered work with done criteria

## Core ontology

These plans share one deeper structure even when the domains differ.

1. They define why the project is worth doing.
2. They define what the project must not become.
3. They define the quality bar or success criteria that make the work worth
   shipping.
4. They choose a direction and freeze it.
5. They define architecture boundaries before task execution.
6. They define a hard contract that implementation is not allowed to violate.
7. They state the current baseline so future work is anchored in reality.
8. They separate in-scope from out-of-scope work.
9. They define an ordered execution model for agents.
10. They make progress legible through checklists, done criteria, and
    completion notes.

This is the key pattern:

- thesis
- anti-thesis
- quality bar
- chosen direction
- constraints
- current state
- dependency rules
- execution order
- verification-backed progress

If one of those is missing, the plan will usually drift.

## Conciseness rule

A strong plan is dense, not long.

- Keep only information that changes implementation behavior.
- Remove generic software advice.
- Prefer one strong rule over three overlapping paragraphs.
- Keep tasks short enough to scan and strong enough to execute.
- Put detail into artifacts only when it must survive across handoffs.

## First steps

1. Read `AGENTS.md` and preserve repo-specific constraints.
2. Read the current plan, if one exists.
3. Identify whether the user wants:
   - a new plan
   - a rewrite of an existing plan
   - a progress update on an existing plan
   - extraction of ontology only
4. Inspect enough repo context to avoid writing fiction:
   - current stack
   - existing directories
   - concrete entry points, seams, and divergence points
   - tests and tooling
   - active artifacts or docs
5. Before writing or rewriting the task list, determine whether these are clear
   enough:
   - product thesis
   - anti-thesis or non-targets
   - quality bar or success criteria
   - hard contract
   - domain-specific contracts
   - current baseline
   - dependency boundaries
   - verification model
   - durable artifact location
6. If any of those are missing, it is the agent's responsibility to close the
   gap before drafting tasks:
   - infer from repo evidence when possible
   - identify contradictions explicitly
   - ask the user only for decisions that cannot be recovered safely
7. Write the plan as a decision document first and a task list second.

## Generation responsibility

When generating a new plan, do not jump straight to checkboxes.

The agent must think ahead like the future executor of the plan.

- Identify conceptual gaps.
- Identify motivational gaps.
- Identify missing constraints.
- Identify missing success criteria.
- Identify places where task order is unclear or unsafe.
- Identify where artifacts are needed to avoid rediscovery.

The plan is not ready until objectives, constraints, and task order are clear
enough that another agent could execute the first task without reopening the
strategy.

## Required plan shape

Use these sections as the default scaffold, not as a rigid template.

The ontology matters more than the exact headings.

- merge sections when one repo expresses the concept more clearly that way
- rename sections when the repo already has stronger local language
- omit sections that would be empty or fake
- add domain-specific sections when the repo needs sharper contracts
- preserve the plan's existing shape if it already answers the right questions
  well

Do not force cosmetic heading churn onto a strong existing plan just to match a
canonical outline.

### Context

State the real situation, not the ideal one.

- what exists now
- why the current structure is inadequate
- what external product or research pressure matters
- what must continue working during the transition

### Product Thesis

Say what makes the project worth continuing.

- name the durable value proposition
- say what the software should be good at
- make the intended quality bar explicit

### Anti-Thesis or Explicit Non-Targets

Say what would make the project not worth continuing.

- name tempting but wrong directions
- name local maxima that produce weak software
- say what success must not be confused with

### Quality Bar or Success Criteria

State the bar that decides whether the project is actually good enough.

- define what counts as real success, not just structural validity
- name the output quality threshold or parity threshold explicitly
- say what must fail fast, degrade gracefully, or remain respectable
- make clear what "works" is not allowed to mean

### User Profile or Workflow

Only include this when user workflows materially shape architecture.

- who the primary user is
- what the main workflow loop is
- what must feel reliable or fast

### Objective

Write one hard objective sentence.

Good objectives describe the end state and the constraint:

- modular while preserving contract
- production-minded while staying Expo Go compatible
- one canonical composition contract across outputs
- flagship Beamer quality over multi-backend sprawl

### Direction Chosen

Freeze the strategic choices so agents do not reopen them casually.

- rank backends, platforms, or implementation priorities
- say what comes first and what must wait
- make tradeoffs explicit

### Current Baseline or Code-Informed Baseline

Describe what already exists and must be preserved or used.

- tooling
- source layout
- concrete files, functions, classes, routes, commands, or modules that already
  own the seam
- known runtime forks, divergence points, or duplicate sources of truth
- tests
- adapters
- artifacts
- existing workflows

Do not let future agents rediscover the same facts.

When the repo already has meaningful code, this section should be code-informed,
not only architectural. Name the actual implementation seams that matter:

- canonical entry points
- current orchestration functions
- public CLI or API contract points
- divergence points that create parity or maintenance risk
- existing automated protection and known gaps

### In Scope

List what work this phase includes.

### Out Of Scope

List what work this phase explicitly excludes.

This is not filler. It protects the repo from drift.

### Hard Contract or Guardrails

State what must remain stable.

Examples:

- public HTTP contract
- visual parity contract
- repo-owned metadata as source of truth
- flagship backend quality gate
- no compatibility shims by default

These rules must read as non-negotiable.

### Domain-Specific Contracts

Add one or more dedicated contract sections when the repo has domain boundaries
that are too important to hide inside generic guardrails.

Typical examples:

- repository contract
- markdown contract
- cloud and failure policy
- AI policy
- output rules
- projection-report seam
- API cutover rule

Use this when the repo needs a precise statement of behavior, ownership, or
failure semantics for one specific area.

Each contract should say:

- what the source of truth is
- what may vary and what must stay stable
- what failure or fallback behavior is allowed
- what must be validated before data crosses the boundary
- what other sections or tasks are downstream of that contract

### Dependency Rules

This is one of the most important sections.

- define named layers or areas
- state allowed import directions
- state forbidden dependencies explicitly
- keep transport, SDK, persistence, UI, and domain boundaries concrete

Good dependency rules prevent architecture from becoming slogan-only.

### Target Architecture or Source Layout

Include this when shape matters operationally.

- named modules
- layer responsibilities
- target folder layout
- projection of how the repo should look after migration

### Working Method For Agents

Tell agents how to act inside the plan.

Typical rules:

- work on the first unchecked task
- do not mark tasks done without evidence
- split oversized tasks before implementation
- update the plan after completion
- prefer the repo's canonical language/tooling for new files

### Verification Gate

Add this when correctness depends on contract preservation.

A strong plan says what must be protected before runtime edits:

- identify the behavior being changed
- identify the tests that already protect it
- add missing tests before the code change when needed
- record waivers explicitly when protection is blocked

### Durable Artifacts

Name the folder where migration or planning artifacts live.

The plan should centralize durable notes instead of scattering them across the
repo.

The plan should also say what belongs there:

- inventories
- manifests
- parity notes
- coverage gaps
- task-specific logs
- contract maps
- decision records that must survive handoffs

When recurring artifacts matter, name the actual files, not only the folder.
Examples:

- `task-log.md`
- `open-decisions.md`
- `test-guardrails.md`
- `coverage-gaps.md`
- `projection-capability-matrix.md`
- `database-replacement-readiness.md`

### Phase Gates or Tranches

Add this when the work is not one flat checklist.

Use it to define:

- major phases that must happen in order
- entry criteria for a later phase
- what must be frozen before downstream tasks start
- which tranche currently has the highest execution priority

This prevents agents from starting runtime work before the safety or contract
baseline exists.

### Task Split Rules

Add explicit split rules when a task can easily become too broad.

Good split rules say when an agent must stop and create child tasks before
editing code. Examples:

- the task touches more than one seam or layer family
- the task changes both contract shape and runtime behavior
- the task needs more than one guardrail family
- the task spans compiler plus projector, transport plus persistence, or CLI
  plus internals

This keeps "first unchecked task" execution safe.

### Completion Note Format

If the repo relies on durable evidence, define the completion-note format
explicitly instead of leaving it implied.

Useful required fields:

- date completed
- what changed
- which files, modules, or seams were touched
- which artifacts were added or updated
- which verification commands passed
- residual waivers or follow-up left open

### Ordered Tasks

This section is the execution engine.

Each task should have:

- a checkbox
- a concrete outcome
- done criteria
- optional notes or substeps
- a completion note after it is finished

Tasks should be dependency-ordered, not merely topic-grouped.

Tasks should also be extensible:

- leave room to insert new subtasks without rewriting the whole plan
- split large tasks before execution, not after failure
- preserve stable numbering or headings when later tasks will cite them
- use artifacts to absorb detail that would otherwise bloat the main plan

When the repo requires it, the task engine should also encode:

- tranche or phase order
- exact gate conditions between phases
- named artifact outputs for specific tasks
- explicit automated guardrail families
- exact verification commands before a task can be closed

### Verification Commands

Add this section when the repo has a known command sequence that should be run
before closing tasks or before moving between phases.

Be exact. Name the real commands, not generic placeholders such as "run tests".

Examples:

- `uv run pytest tests/test_render.py`
- `bash scripts/run-tests.sh`
- `uv run pyright`
- `docker compose -f docker-compose.test.yml up`

## Writing rules

Follow these style rules when authoring or revising a plan.

- Write with decision authority, not brainstorming tone.
- Prefer short declarative bullets over long motivational prose.
- Be concrete about files, modules, routes, backends, or workflows.
- Name unstable interfaces explicitly.
- Use negative rules where ambiguity would be expensive.
- Do not hide strategic decisions inside task notes.
- Do not present optional architecture ideas as mandatory constraints.
- Do not rewrite a strong plan into a worse shape just to normalize headings.

## Task design rules

Good tasks are not chores. They are contract-preserving units of change.

- Order tasks so early tasks create safety for later tasks.
- Put test-independence and baseline capture early when migration risk is high.
- Put inventory, contract mapping, or artifact creation before large rewrites.
- Put test-protection or coverage work before runtime edits when behavior must
  stay stable.
- Use tranche boundaries or gates when later runtime work depends on earlier
  contract cleanup.
- Do not mix several large migrations into one checkbox.
- If a task is broad, split it before execution.
- Done criteria must be externally checkable.
- Completion notes should mention actual evidence, not effort.

The common safe pattern is:

1. ensure coverage or protection for the behavior
2. update the implementation
3. run verification
4. update the plan and artifacts

Do not invert that order casually.

Bad task:

- `Refactor architecture`

Good task:

- `Record the current admin endpoints used in tests`
- `Add backend-invariant PNG parity coverage`
- `Make tests independent from implementation details`

## Completion-note rules

A finished task should leave a short durable note.

Completion notes should include:

- date when useful
- what changed
- the main files, modules, or seams changed
- the artifact files added or updated when relevant
- the tests or exact verification commands that justify the checkbox
- residual waivers or follow-up when relevant

Do not write:

- `done`
- `implemented`

without evidence.

## Update policy

When maintaining an existing plan:

1. Preserve the document's strategic commitments unless the user explicitly
   changes direction.
2. Preserve effective local section shapes when they already express the
   ontology well.
3. Update the baseline when reality changes.
4. Mark completed tasks only after verification.
5. Add completion notes instead of silently flipping checkboxes.
6. If a new pattern affects many tasks, fix the plan globally rather than
   patching one local section.

When updating a plan after implementation work, preserve the same execution
discipline:

- tests or coverage first when behavior changes
- code update second
- verification third
- plan/task/artifact update last

## Heuristics from strong exemplars

The strongest engineering plans usually share these traits:

- They make the product thesis explicit before any architecture talk.
- They include a strong anti-thesis so agents know what not to optimize.
- They make the quality bar explicit instead of treating "it runs" as success.
- They freeze one direction instead of balancing every option.
- They treat the hard contract as a first-class section.
- They break out domain-specific contracts when one generic guardrail section
  would hide critical behavior.
- They distinguish current baseline from target architecture.
- They anchor the baseline in real code seams instead of abstract diagrams
  alone.
- They use phases, gates, and split rules instead of one flat undifferentiated
  checklist.
- They define exact verification commands when correctness depends on a known
  command contract.
- They define dependency rules in named layers.
- They tell agents exactly how to pick the next task.
- They convert progress into evidence, not optimism.
- They preserve ontology even when the surface heading structure differs across
  repos.

## Minimal authoring workflow

1. Gather the repo reality.
2. State the thesis and anti-thesis.
3. Define the quality bar or success criteria.
4. Freeze the chosen direction.
5. Define objective, scope, and hard constraints.
6. Define dependency rules and target architecture.
7. Capture the code-informed baseline and durable artifact locations.
8. Define phases, gates, split rules, and verification commands when the repo
   needs them.
9. Write the ordered task list with done criteria.
10. Re-read the plan as if a new agent had to execute it without extra context,
    without assuming canonical headings.

## Mini-outlines

Use these as compact archetypes, not as copy-paste templates.

### 1. Contract-preserving backend migration

- `Context`: deployed system, public API must stay stable, logic is entangled.
- `Product Thesis`: keep the existing product working while making the backend
  modular and replaceable.
- `Anti-Thesis`: do not treat a framework rewrite or database swap as the first
  goal.
- `Quality Bar`: preserved endpoints keep wire behavior and important side
  effects; modular code becomes easier to test and change.
- `Hard Contract`: route paths, auth mode, status codes, response shapes, and
  observable side effects stay stable unless a task explicitly changes them.
- `Code-Informed Baseline`: current routes, middlewares, models, test harness,
  startup behavior, and known coupling points.
- `Dependency Rules`: domain, application, persistence, transport, and external
  modules each have explicit import boundaries.
- `Durable Artifacts`: route inventory, keep/delete manifest, coverage gaps,
  test-protection log, feature and side-effect maps.
- `Phase Gates`: baseline and contract coverage first, source skeleton second,
  low-risk migrations before high-risk flows, upgrades only after protection
  exists.
- `Task Pattern`: protect behavior -> extract one seam -> verify -> update plan
  and artifacts.
- `Verification Commands`: exact typecheck, test, and container or live-server
  commands required before closing tasks.

### 2. Mobile product plan with replaceable adapters

- `Context`: Android-first app, product workflow matters as much as code
  structure, first version should avoid premature native complexity.
- `Product Thesis`: the phone is a strong capture and reading surface; durable
  data lives in repository-backed Markdown plus metadata.
- `Anti-Thesis`: do not build a generic AI chat wrapper or a workflow that
  requires power-user tooling before the core product works.
- `Quality Bar`: the primary read, capture, edit, search, and move workflows
  feel reliable on a physical phone; AI stays optional and replaceable.
- `Domain-Specific Contracts`: repository contract, Markdown contract, cloud
  and failure policy, AI policy, secret-handling rules.
- `Direction Chosen`: Expo Go first, repository-aware session load from the
  start, native builds only when a real capability gap justifies them.
- `Code-Informed Baseline`: current Expo scaffold, module skeleton, provider
  wiring, tests, validation scripts, and existing adapters.
- `Dependency Rules`: domain and application stay free of UI, router, storage,
  and network client details; infrastructure implements ports.
- `Target Layout`: modules split by feature with domain/application/
  infrastructure/presentation boundaries.
- `Ordered Tasks`: scaffold and tooling, contracts and schemas, application
  services, first adapters, core UI, capture/AI flows, native capability work
  only later.
- `Verification Commands`: exact unit, app, emulator, and build checks.

### 3. Renderer or compiler cutover with parity requirements

- `Context`: multiple runtime paths define output today, causing fragile parity
  and duplicated rendering logic.
- `Product Thesis`: one canonical compiled representation should define output,
  with backend-specific projectors deriving from it.
- `Anti-Thesis`: do not keep several layout truths alive or fork renderer logic
  per output target.
- `Quality Bar`: canonical raster output is backend-invariant where supported;
  editable outputs are projections, not layout authorities.
- `Hard Contract`: output parity rules, fallback rules, unsupported-feature
  behavior, and cutover policy are explicit.
- `Code-Informed Baseline`: current compiler entry points, shared render seam,
  projector functions, CLI contract points, and divergence paths.
- `Domain-Specific Contracts`: output rules, projector-report seam, cutover
  rule, template-package or IR contract.
- `Phase Gates`: freeze shared render contract first, then compiler contracts,
  then thin projectors, then parity corpus and cutover.
- `Durable Artifacts`: open decisions, test guardrails, parity waivers, golden
  corpus, capability matrix, task log.
- `Task Split Rules`: split when a task touches compiler plus projector, CLI
  plus runtime internals, or more than one guardrail family.
- `Verification Commands`: exact focused tests, full test suite, format/lint,
  and parity or raster-diff commands.

## Anti-pattern appendix

These are common ways a document looks like an engineering plan while failing
to guide execution well.

### 1. The vague roadmap

Bad pattern:

- "Improve architecture"
- "Make the app production-ready"
- "Support more backends"

Why it fails:

- no hard contract
- no frozen direction
- no quality bar
- no next executable step

Stronger replacement:

- name the exact contract to preserve
- name the chosen direction and rejected alternatives
- define the output or behavior bar
- turn the first real seam into a concrete task with done criteria

### 2. The task dump without strategy

Bad pattern:

- dozens of checkboxes with no thesis, no anti-thesis, and no task ordering

Why it fails:

- agents cannot tell which work is actually important
- early tasks may not create safety for later risky edits
- checkboxes drift into local chores instead of product-shaping work

Stronger replacement:

- write the decision document first
- separate in-scope from out-of-scope
- add phase gates when later work depends on earlier contract cleanup
- order tasks so protection, inventory, and baseline capture come before major
  rewrites

### 3. The architecture slogan plan

Bad pattern:

- "Use clean architecture"
- "Adopt DDD"
- "Modularize the codebase"

Why it fails:

- boundaries stay abstract
- real seams, imports, and ownership rules remain undefined
- future agents still have to rediscover the repo

Stronger replacement:

- name the actual layers or modules
- state allowed and forbidden dependency directions
- anchor the baseline in concrete files, functions, routes, or CLI entry points
- name current divergence points or duplicate sources of truth

### 4. The fake verification plan

Bad pattern:

- "Run tests"
- "Verify everything still works"

Why it fails:

- nobody knows which checks are mandatory
- tasks get marked done on confidence rather than evidence
- protection is often added after runtime changes instead of before

Stronger replacement:

- define the verification gate before risky edits
- name exact commands
- require completion notes to cite the evidence
- record waivers explicitly when test-first protection is blocked

### 5. The template-fetish rewrite

Bad pattern:

- forcing a strong existing plan into a canonical heading set just for
  uniformity

Why it fails:

- creates churn without improving execution
- can erase repo-specific language that made the plan clear
- mistakes surface structure for ontology

Stronger replacement:

- preserve the existing shape when it already expresses the right concepts well
- normalize missing ontology, not cosmetic headings
- add domain-specific sections only when they sharpen execution

## Output standard

A good engineering plan should let a competent agent answer these questions
without asking the user again:

- What is this project really trying to become?
- What directions are explicitly rejected?
- What quality bar decides whether the output is actually good enough?
- What must stay stable while work happens?
- What concrete files, functions, commands, or seams define the current
  baseline?
- What domain-specific contracts define source of truth, failure behavior, or
  output semantics?
- What code boundaries matter?
- What phase gates, split rules, artifact files, and exact verification
  commands govern execution?
- What should be done next?
- What evidence is required before a task can be called complete?
