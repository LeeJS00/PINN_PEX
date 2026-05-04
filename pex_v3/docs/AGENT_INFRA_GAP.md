# Agent Infrastructure Gap — Phase C Discovery

_Date: 2026-05-02_
_Status: gap identified, workaround in place, structural fix recommended_

## Critical finding

The 8 specialist agents defined in `.claude/agents/*.md` are **not directly
invocable** through the Agent tool's `subagent_type` parameter. Available
types in this session are:

```
claude-code-guide, codex:codex-rescue, Explore, general-purpose, Plan,
statusline-setup
```

Custom agent definitions in `.claude/agents/` are NOT registered as
selectable subagent types. Attempting `subagent_type: "pex-data-engineer"`
returns:

```
Agent type 'pex-data-engineer' not found.
Available agents: claude-code-guide, codex:codex-rescue, ...
```

This means the Phase C role validation (per `PHASE_C_AGENT_VALIDATION_PLAN.md`)
cannot directly use the named agents.

## Implication

The 8-agent roster I defined is currently:
- ✅ Documented (markdown role definitions exist)
- ✅ Discoverable via filesystem
- ❌ Not invocable as `subagent_type`
- ❌ Not auto-registered with the Claude Code agent system

This is the kind of gap that critical-analysis discipline (CLAUDE.md
mandate) is supposed to catch BEFORE building infrastructure. I missed it
when defining the agents — assumed the markdown files would be auto-picked-up.

## Workaround used in this session

For Phase C Round 1, four roles were validated by invoking
`subagent_type: "general-purpose"` and **embedding the role definition in
the prompt**:

```
You are being invoked as the **<role-name>** specialist agent. Read your
role definition at /home/jslee/projects/PINNPEX/.claude/agents/<role>.md
FIRST and operate strictly within that role.

[domain-specific task]
```

This works for one-off invocations but loses:
- Tool restrictions (the agent definition's `tools:` list isn't enforced)
- Model override (the `model: opus` line in role md isn't honored)
- Auto-discoverability (caller must manually pass the path)
- Role caching (each invocation re-reads the file)

## Two paths forward

### Path A — Keep role markdown as documentation; use general-purpose wrapper

Status: **active** (this session's Phase C uses this pattern).

Pros:
- Zero infrastructure work required.
- Roles still serve as documentation for what the specialist should do.
- Each invocation is auditable (role md content visible in agent prompt).

Cons:
- Tool restrictions / model overrides not enforced.
- Caller must remember the role path.
- Drift risk: role md and prompt-embedded version can diverge over time.

### Path B — Make custom agents real subagent types

Status: **research needed**. Investigate how Claude Code registers custom
subagent types. Options:
1. Configuration in `.claude/settings.json` to auto-register agents.
2. Plugin manifest declaring agents.
3. (Maybe not supported in current Claude Code; would require feature
   request.)

Action: query Claude Code docs to confirm whether custom subagent types
are supported, or whether the markdown-in-`.claude/agents/` pattern is
just naming convention without runtime hookup.

## Recommendation

For this project (Strategy v3), **stay on Path A**:
- Path B is an infrastructure investment of unknown size.
- Path A delivers the value (specialist perspective via role-embedded
  prompts) immediately.
- The role md files are themselves valuable as the "operational manual"
  for each domain — they survive any infrastructure change.

When Phase 1 model code starts, every invocation of a specialist follows
the Path A pattern. The lead session (this Claude) is the coordinator;
specialists are domain-specific consults via general-purpose + prompt.

## Memory update

Save a memory entry noting this gap so future sessions don't waste
turns trying `subagent_type: "<custom>"` directly.
