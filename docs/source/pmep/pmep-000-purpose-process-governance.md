# PMEP 0: How we decide larger changes

| | |
|---|---|
| **Status** | Draft |
| **Type** | Process |
| **Author** | Daniel Krakowczyk |
| **Created** | 2026-09-08 |

## TL;DR

Changes that many other things depend on (the data model, operation contracts, on-disk
formats, this process) require a PMEP: a design document in `docs/source/pmep/`,
proposed and decided in a pull request. Acceptance needs a window of 14 days, 2 approvals
from active core contributors other than the author, and no unresolved objection. The lead
maintainer may override objections but not the approval floor. Core contributors are listed
in `GOVERNANCE.md`. Everything else keeps the usual issue and PR workflow.

Several changes to the data model and the operation layer are upcoming. This page states how
they get decided, so that the reasoning is written down once and can be cited later instead of
reconstructed from issue threads.

## What needs a proposal

A **PMEP** (pymovements Enhancement Proposal) is required only for changes that many other
things depend on:

- the data model: containers, slots, how recordings, events, stimuli and measures are represented
- contracts every operation or pipeline follows: operation contract, registry, column declarations, log format
- on-disk formats and layouts that published datasets depend on
- this process

Adding an attribute to an existing class needs a PMEP only if readers, writers or operations
must account for it, not if it is local to the class.

Everything else, including breaking changes to individual functions or classes, is an issue,
a pull request, a changelog entry, and the usual five-minor-release deprecation window.
When in doubt, open an issue. It graduates to a PMEP if the discussion shows it needs one.

Work already in review or on an agreed issue when this process is adopted is grandfathered:
no PMEP is required for it retroactively.

## Where they live

`docs/source/pmep/pmep-NNN-short-title.md`, opened as a pull request and rendered in the
documentation. The number is assigned when the pull request opens, so the proposal can be
cited by number during discussion. The pull request stays open until the decision falls and
merges carrying the final status. Rejected and withdrawn PMEPs are merged too and stay in the
repository as records of what was considered and why not.

Status is one of: Draft · Accepted · Final (implemented) · Rejected · Withdrawn · Superseded.

## Who decides

**Core contributors** are listed in `GOVERNANCE.md` and have the right to vote. Anyone may
comment. The entry
criterion, additions to the list and the emeritus rule are defined in `GOVERNANCE.md`.

**The lead maintainer decides objections.** An unresolved objection may be overridden. The
override is written into the PMEP with reasoning and answers the objection on its merits.
The approval requirement below is a hard floor and cannot be overridden.

## How a PMEP is accepted

Votes are cast as GitHub reviews on the pull request: an approving review counts as an
approval, a request for changes as an objection. The acceptance window starts when the author
marks the pull request ready for review.

1. Open at least 14 days since the last substantive revision.
2. At least 2 approvals from active core contributors other than the author(s).
3. No unresolved objection.

An objection must state a technical reason concrete enough to be addressed. It does not have
to propose the fix. An objection without such a reason does not block.

An objection is resolved when withdrawn, when addressed in the text and not renewed within
7 days, or when overridden by the lead maintainer. After 30 days without quorum the author
pings the active core contributors. Nothing changes state automatically.

A PMEP may be discussed in a scheduled design review meeting. The meeting does not shorten
the 14 day window, votes still count only as GitHub reviews on the pull request, and
decisions or objections from the meeting are written into the PMEP within a week.

A substantive change to an *Accepted* PMEP reopens a 7 day window and needs 1 approval.
Editorial changes need none.

## What a PMEP must contain

In this order:

- **TL;DR** (at most 150 words)
- **What it looks like** (code, before and after)
- **Resulting signatures** (written out, with schema versions for file formats)
- Motivation
- Specification
- Rationale, including rejected alternatives
- Backwards compatibility (deprecation version, computed removal version)
- Implementing issues

This structure is for Standards PMEPs. Process PMEPs adapt it and keep what applies.

The first three are mandatory before voting opens. A reviewer should be able to form a view in
five minutes from them alone. `docs/source/pmep/TEMPLATE.md` has them stubbed.

## Relationship to issues and PRs

A PMEP specifies. Issues carry it out and link back. PRs implementing an accepted PMEP cite it,
and may be declined for diverging from it without reopening the design. If implementation shows
the spec is wrong, the PMEP is revised.

## Adoption

This PMEP is adopted under its own rule: 14 days, 2 approvals from active core contributors
other than the author, no unresolved objection. The initial core contributor list in
`GOVERNANCE.md` is accepted with it.
