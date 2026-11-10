<!--
Title: start with a type prefix and a colon, as in "fix: ..." or "feat: ...".
Prefixes: feat, fix, docs, dataset, depr, perf, build, chore, ci, refactor, test, release.
A scope in parentheses is optional, as in "feat(events): ...".
Add "!" after prefix and scope for a breaking change, as in "feat!: ..." or "feat(events)!: ...".
-->

<!--
If it applies: open the body with a "## 💥 Breaking Change" and/or a "## ⚠️ Deprecation" section.
Definitions:
https://pymovements.readthedocs.io/en/latest/contributing/contributing-code.html#pull-request-body
-->

## Description

<!-- What does this change do, and why? Write "Resolves #N" if it fixes an issue. -->

<!-- If a generative AI tool did part of the work, name the tool and what it did here. It helps the reviewer. -->

## Changes

<!-- A short bullet list that maps the diff, one bullet per change. -->

## Testing

<!-- How did you verify the change? Name the tests you added or ran and the checks you made. -->

<!--
If it applies: add "## 🚨 Merge Danger" and "## Dependencies" after Testing, in that order.
Definitions:
https://pymovements.readthedocs.io/en/latest/contributing/contributing-code.html#pull-request-body
-->

## Checklist

- [ ] If I used generative AI, I was in the loop, I checked the output with care, and I can explain every change in my own words (see [Use of Generative AI](https://pymovements.readthedocs.io/en/latest/contributing/contributing-code.html#use-of-generative-ai))
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have checked my code and corrected any misspellings
- [ ] First-time contributors: I added myself to `CITATION.cff`
