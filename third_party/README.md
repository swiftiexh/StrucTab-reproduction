# Third-party dependencies

## StrucTab official repository

This reproduction references, but does not redistribute, the official StrucTab repository:

- Repository: <https://github.com/VirtualLUOUCAS/StrucTab>
- Tested commit: `5d2935c9c400e524d53c9cf87656142ba38f09f9`
- Upstream license: see the official repository's `LICENSE`

Clone it into the project root before running the official TEDS/Reward integration:

```bash
git clone https://github.com/VirtualLUOUCAS/StrucTab.git StrucTab
git -C StrucTab checkout 5d2935c9c400e524d53c9cf87656142ba38f09f9
```

The reproduction loads the following official components at runtime:

- Uni-TabRL validity and 1D Probe primitives;
- TEDS/TEDS-S implementation;
- HTML rendering template;
- official three-section parsing semantics.

## Local source fix

The official training-data script originally counted physical `td`/`th` tags when deriving the
column count. This under-counts cells with `colspan` and does not combine them with active
`rowspan` occupancy. Apply the recorded patch if you want to reproduce that source-level fix:

```bash
git -C StrucTab apply ../patches/structab_logical_column_count.patch
```

The patch changes only `code/training_data/get_sequencial_reasoning.py`. Related regression tests
are in `reproduction/tests/test_official_training_data.py`.

