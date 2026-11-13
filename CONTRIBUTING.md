# Contributing to Chainfix

Keep changes focused and describe the resulting behavior and relevant validation
in the pull request. Refer to pull requests by their GitHub number or link. Name
planning stages by their purpose, such as runtime baseline, real scalars, complex
scalars, arrays, buffer reuse, and reductions.

## Pull request titles and merging

Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for PR
titles and the resulting squash commits:

```text
type(optional-scope)!: concise description
```

Both the scope and `!` are optional. Use `!` when the change breaks compatibility,
including changes to supported Python versions. Explain the break and any
migration steps in the description, ending with a `BREAKING CHANGE:` footer.

Allowed types:

| Type | Purpose |
| --- | --- |
| `feat` | Add user-facing functionality |
| `fix` | Correct a bug |
| `docs` | Change documentation |
| `test` | Add or improve tests |
| `ci` | Change CI workflows or checks |
| `build` | Change packaging, build tools, dependencies, or supported runtimes |
| `refactor` | Restructure code without changing its behavior |

Scopes are optional. When used, choose one of `context`, `binary`, `complex`,
`decimal`, `arrays`, or `deps`. Extend this list and the workflow together through
normal review when a new area warrants it.

Examples:

```text
build!: require Python 3.12+
fix(context): preserve default contexts when installing a copy
feat(binary): add arithmetic operators and math policies
feat(complex): add CFixb
feat(arrays): add fixed-point array multiplication
ci: validate pull request titles
```

Use squash merging. The repository permits only squash merges and defaults the
squash commit to the PR title and description. Review that message before merging,
including any compatibility notes. The required `Validate PR title` check runs
when a PR is opened, reopened, edited, or updated with new commits. It validates
the title's syntax, type, and optional scope; reviewers check the accuracy of the
type, breaking-change marker, and description.

Intermediate branch commits do not need to follow this convention. Apply the
policy prospectively; do not rewrite existing history. For a change spanning
multiple types, select the dominant purpose for the title and explain the rest
in the description. Prefer separate PRs for independently reviewable changes.

## Releases before 1.0

Releases remain manual. The version lives in `src/chainfix/__init__.py`; publishing
a GitHub release triggers the existing package-publishing workflow. Merging a PR
does not itself publish a release.

For versions below 1.0, determine the next version from all changes since the last
release, applying the highest applicable bump once:

| Changes included in the release | Bump | Example |
| --- | --- | --- |
| Any breaking change (`!`), regardless of type | Minor | 0.1.2 → 0.2.0 |
| A new feature (`feat`) without a break | Minor | 0.1.2 → 0.2.0 |
| Bug fixes (`fix`) without either of the above | Patch | 0.1.2 → 0.1.3 |
| Other changes without a break | No automatic release requirement | Maintainer decides |

Mark breaking changes even before 1.0. Multiple breaking or feature PRs can belong
to one release; do not increment the version for each PR. The planned runtime
baseline and first arithmetic release target 0.2.0 at the agreed release boundary.
Include relevant fixes in the release notes even when their squash commit has
another primary type.

If release automation is introduced later, preserve these rules. For Release
Please, this means `bump-minor-pre-major: true` and
`bump-patch-for-minor-pre-major: false`. Automation and the versioning policy for
1.0 onward require a separate change.

## Testing

Follow the installation and testing instructions in [README.md](README.md).
Include the commands run, results, and any limitations in your PR description.
