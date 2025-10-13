<!--
Sync Impact Report - Constitution v1.0.0
========================================

Version Change: NONE → 1.0.0 (Initial constitution creation)

Rationale: First formal constitution establishing Engineering Excellence (EE-style)
decision-making methodology and engineering principles. This is a MAJOR version as
it establishes the baseline governance structure focused on structured reasoning,
code quality, testing, UX, and performance standards.

New Principles (4 core areas):
- NEW: EE-Style Decision Mode (Structured Reasoning)
- NEW: Code Quality Standards
- NEW: Testing Standards
- NEW: User Experience Consistency
- NEW: Performance Requirements

Added Sections:
- Core Principles (EE Decision Mode + 4 Engineering Principles)
- EE Reflection Template
- Decision Rubric
- Guardrails & Anti-Patterns
- Evaluation Check
- Governance

Templates Requiring Updates:
✅ plan-template.md - Constitution check updated with EE principles
✅ spec-template.md - Aligned with EE requirements
✅ tasks-template.md - Aligned with testing and observability standards

Follow-up TODOs:
- None (initial constitution complete)

Date: 2025-10-12
-->

# Engineering Excellence Constitution

## Core Principles

### I. EE-Style Decision Mode (NON-NEGOTIABLE)

All significant technical decisions MUST follow structured reasoning that explicitly
compares at least TWO plausible actions, predicts outcomes for each, and provides
a justified expert choice.

**Output Schema (Enforced):**

Return two top-level fields ONLY:

- **Reasoning**: 3–8 sentences that:
  - (a) States the current state/signal
  - (b) Lists Action A, Action B, (C optional) with expected outcomes/evidence
  - (c) Cites risks/assumptions
  - (d) Explains why the expert action is best

- **Action**: A single imperative line describing the chosen expert action

**Formatting Requirements:**
- Start Reasoning with "Reasoning:" and end with a one-line summary
- Start Action with "Action:" and write one precise command/next-step

**Rationale:** Structured decision-making prevents hasty choices, documents rationale
for future maintainers, and ensures engineering excellence through systematic
evaluation of alternatives and predicted outcomes.

### II. Code Quality Standards (NON-NEGOTIABLE)

All code MUST meet high-quality standards for maintainability, clarity, and robustness.

**Requirements:**
- Small, cohesive modules with explicit boundaries and no hidden state
- Document "purpose, not effect" at file headers
- TODOs MUST include owner names
- APIs MUST be stable and versioned
- Inputs MUST be validated
- Errors MUST be typed and actionable

**Rationale:** Code quality directly impacts maintainability, debugging speed, and
system reliability. Explicit boundaries and documented purposes reduce cognitive
load for future developers.

### III. Testing Standards (NON-NEGOTIABLE)

All features MUST have comprehensive test coverage with deterministic behavior.

**Requirements:**
- Tests MUST co-evolve with code
- Require at least one unit test + one integration test per feature
- Use deterministic seeds/fixtures—flaky tests are failing tests
- Cover critical paths: parsing, I/O, concurrency, recovery, timeouts
- Tests MUST be written with acceptance criteria

**Rationale:** Deterministic tests enable reliable CI/CD. Comprehensive coverage
across unit and integration levels ensures both component correctness and system
behavior validation.

### IV. User Experience Consistency

All user-facing features MUST maintain consistent UX patterns and accessibility
standards.

**Requirements:**
- Follow existing design tokens, spacing, and interaction patterns
- Keyboard and screen-reader friendly with visible focus states
- Error messages MUST be actionable
- Same intent MUST produce same pattern across all surfaces
- Meet WCAG AA+ accessibility standards

**Rationale:** Consistent UX reduces cognitive load, improves accessibility for all
users, and builds trust through predictable interactions.

### V. Performance Requirements

All features MUST meet defined performance budgets and avoid common anti-patterns.

**Requirements:**
- Set explicit budgets: P95 latency, memory usage, and payload sizes per surface
- Avoid N+1 queries and other performance anti-patterns
- Stream or paginate large datasets
- Cache with clear TTL policies
- Add tracing and metrics around hot paths
- Guard against thundering herd with rate limiting

**Rationale:** Performance directly impacts user experience and system scalability.
Explicit budgets prevent performance regressions and ensure the system remains
responsive under load.

## EE Reflection Template

**The agent MUST internally follow this template for all significant decisions:**

```
State: <succinct description of the situation>

Alternatives:
- Action A: <what> → Expected Outcome: <good/bad, evidence>
- Action B: <what> → Expected Outcome: <good/bad, evidence>
- (Optional) Action C: <what> → Expected Outcome: <good/bad, evidence>

Analysis: Compare outcomes, risks, reversibility, testability, UX/perf impact.

Conclusion: Therefore, the best action is <Expert Action> because <primary reasons>.
```

## Decision Rubric

**When choosing between alternatives, prefer actions that:**

1. Satisfy acceptance criteria with the fewest moving parts
2. Reduce error surface area and prevent regressions
3. Preserve UX consistency and accessibility (WCAG AA+)
4. Improve testability and observability (deterministic, measurable)
5. Offer the best latency/memory/performance trade-offs under load
6. Are reversible and incremental (safe rollout, kill-switch ready)

## Guardrails & Anti-Patterns

**DO NOT:**
- Output code without tests or acceptance notes
- Pick actions whose outcomes are unexamined or unverifiable
- Add speculative features—privilege smallest change that satisfies acceptance criteria
- Use vague language—be precise and verifiable with metrics

**DO:**
- If information is missing, choose the action that acquires it fastest
- Document all assumptions explicitly
- Consider rollback and recovery scenarios
- Include metrics and budgets in all performance-related decisions

## Evaluation Check

**Self-grade decisions 0–1 (internal use; do not print the score):**

Score across four dimensions:
- **(a)** Alternatives compared (at least 2 with predicted outcomes)
- **(b)** Evidence and outcomes provided
- **(c)** Principle fit (code quality, tests, UX, performance)
- **(d)** Reversibility and safety considered

**If score < 0.7, refine the decision once before proceeding.**

## Style & Length Guidelines

**For all decision documentation:**
- Be concise: Reasoning ≤ 8 sentences; Action = one line
- Use precise, verifiable language
- Include any metrics or budgets referenced
- Avoid jargon unless necessary; prefer clarity

## Governance

### Amendment Procedure

1. Propose amendment via pull request to `.specify/memory/constitution.md`
2. Include rationale, impact analysis, and migration plan (if breaking)
3. Update affected templates (plan, spec, tasks) in same PR
4. Require 2 reviewer approvals for MAJOR changes, 1 for MINOR/PATCH
5. Announce amendment to all contributors before merge

### Versioning Policy

- **MAJOR**: Breaking changes to principles, removed requirements, new non-negotiable rules
- **MINOR**: New principles, expanded guidance, new recommended (non-mandatory) practices
- **PATCH**: Clarifications, typo fixes, reformatting without semantic changes

### Compliance Review

- All PRs MUST include constitution compliance check in description
- Constitution violations MUST be explicitly justified with:
  - **Why Needed**: Specific problem being solved
  - **Simpler Alternative Rejected Because**: Why standard approach insufficient
  - **Mitigation Plan**: How to minimize complexity or restore compliance later
- Reviewers MUST challenge unjustified violations

### Constitution Precedence

- This constitution supersedes all other development practices
- In case of conflict between this document and code comments/docs, constitution wins
- EE Decision Framework applies to all significant technical decisions

**Version**: 1.0.0 | **Ratified**: 2025-10-12 | **Last Amended**: 2025-10-12
