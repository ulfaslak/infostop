# Changelog

## Unreleased

### Labels on crowded stops change once: new `num_trials` parameter, default 50

`Infostop` and `SpatialInfomap` have a new parameter, `num_trials`. Infomap now clusters the network 50 times from different random starts and keeps the partition with the shortest codelength. Before, it ran once.

- **Why.** With a single run, the labels of crowded stops depended on the Infomap version and on the operating system. Crowded stops are stops that overlap within `r2`. On 300 crowded test traces, about 1 in 6 got different labels:
  - between Infomap 1.0.6 and 2.13;
  - between macOS and Linux on the same Infomap 2.x version.

  With 50 runs, Infomap 1.0.6, 2.3, 2.13 and 2.16, on macOS and on Linux, agreed on 299 or 300 of the 300.
- **What changes for you.** Labels of clearly separated stops do not change, and points labelled `-1` (moving) never change.
  - On 60 realistic multi-day traces of single people and 60 of 10-person populations, the labels were the same as those infostop 0.1.9 gives with Infomap 1.0.6.
  - On the 300 crowded traces, 20 got different labels than infostop 0.1.9 gave. Each time the new partition has a shorter codelength, so it is a better solution by Infomap's own measure.
- **The old behaviour.** `Infostop(num_trials=1)` and `SpatialInfomap(num_trials=1)` run Infomap once, as before. With 1 run, labels on crowded stops again depend on the Infomap version and the operating system.
- **Cost.** The clustering step runs 50 times:
  - 33,000 points (10 people over 3 days): `fit_predict` takes 0.36 s instead of 0.05 s;
  - 340,000 points (100 people): 4.0 s instead of 0.6 s.
