# Stories

Stories are work packets. They turn product intent into bounded implementation
and validation work.

Implemented story packets currently cover:

- `US-001`: POC contract and professor data scaffold.
- `US-002`: public dataset registry and license audit.
- `US-003`: v1 training smoke baseline.
- `US-004`: public dataset manifest builder.
- `US-005`: NASA MAT derived WAV conversion.

Candidate follow-up stories are tracked in `docs/stories/backlog.md`.

## Normal Story

Use `docs/templates/story.md` for normal feature work.

Suggested path:

```text
docs/stories/epics/E01-domain-name/US-001-short-story-title.md
```

## High-Risk Story

Use `docs/templates/high-risk-story/` when the feature intake classifies work as
high-risk.

Suggested path:

```text
docs/stories/epics/E02-risky-domain/US-012-risky-story-title/
  execplan.md
  overview.md
  design.md
  validation.md
```

## Status Flow

```text
planned -> in_progress -> implemented
                  |
                  v
               changed
                  |
                  v
               retired
```
