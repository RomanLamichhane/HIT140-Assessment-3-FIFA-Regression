# Data Sources and Processing Notes

## Raw dataset used in this repository

The repository now contains the **original raw match CSV supplied by the group** at:

`Shared_Data/raw/fbref_matches_raw.csv`

The file has 104 FIFA World Cup 2026 matches and the columns `Date`, `Round`, `Home`, `Away`, and `Score`. The repository naming and project documentation identify FBref as the working source used for this raw match file. The assessment brief also lists FIFA Official Statistics, The Stats Don't Lie, and FBref as permitted data sources.

For assessment submission, the group should describe only the sources it actually used and should not claim that every permitted source was used if it was not.

SHA-256 of the supplied raw CSV used to build this package:

`927671a8f7ec831574038441b636296b5cb8178580e2fa83ce90dd5a5ce417ae`

## Reproducible cleaning and preparation performed in Python

`build_datasets.py` performs these steps:

- checks that the required columns `Date`, `Round`, `Home`, `Away`, and `Score` exist;
- checks that exactly 104 raw matches are present;
- converts `Date` to a pandas datetime value;
- standardises score separators (hyphen, en dash, em dash) before parsing home and away goals;
- creates unique match IDs (`M001` to `M104`);
- processes matches chronologically;
- calculates rolling and recent-form predictors from previously completed matches only;
- creates the 104-row LR2.1 modelling dataset;
- creates the 208-row LR2.2 team-match modelling dataset;
- exports `Shared_Data/processed/team_match_history.csv` for reproducibility.

The generated LR2.1 and LR2.2 files reproduce the previously supplied modelling datasets exactly.

## Leakage control

The current fixture is added to each team's history only **after** its pre-match explanatory variables are created. Therefore, current-match goals are targets only and cannot leak into the predictors.

For a team's first tournament match, tournament-history predictors use a neutral zero baseline because no prior tournament match exists. This is a documented limitation: zero is a reproducible placeholder, not a direct measure of pre-tournament team strength.

## Potential future improvement

A complete and independently verified pre-tournament FIFA ranking variable could add team-strength information unavailable from tournament history alone. It is not used in the submitted eight-predictor specifications because it was not part of the supplied raw match file and any added variable would need to preserve the assignment's exactly-eight-predictor constraint.
