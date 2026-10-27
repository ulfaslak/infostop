# Changelog

## Unreleased

### Labels on crowded stops change once: new `num_trials` parameter, default 50

`Infostop` and `SpatialInfomap` have a new parameter, `num_trials`. Infomap now clusters the network 50 times from different random starts and keeps the partition with the shortest codelength. Before, it ran once. The measurements behind the numbers below are in [PR #79](https://github.com/ulfaslak/infostop/pull/79).

- **Why.** With a single run, the labels of crowded stops depended on the Infomap version and on the operating system. Crowded stops are stops that overlap within `r2`. In our tests, about 1 in 6 crowded traces got different labels between Infomap 1.0.6 and 2.13, and between macOS and Linux. With 50 runs they agreed.
- **What changes for you.**
  - Labels of clearly separated stops do not change, and points labelled `-1` (moving) never change.
  - On some crowded stops, labels change once: in our tests, 20 of 300 crowded traces got different labels than infostop 0.1.9 gave with Infomap 1.0.6. Each time the new partition has a shorter codelength, so it is a better solution by Infomap's own measure. On realistic multi-day traces we saw no change.
- **Label numbers.** Stops that Infomap groups are numbered by flow, largest first. Infomap 2.9 to 2.15 number them in a different order when they run more than one trial, so infostop renumbers them by flow. Where Infomap already numbers by flow, nothing changes. On Infomap 2.9 to 2.15, stops with exactly the same flow may still be numbered in a different order.
- **The old behaviour.** `Infostop(num_trials=1)` and `SpatialInfomap(num_trials=1)` run Infomap once, as before. With 1 run, labels on crowded stops again depend on the Infomap version and the operating system.
- **Cost.** The clustering step runs 50 times. In our tests, `fit_predict` on 33,000 points took 0.36 s instead of 0.05 s, and on 340,000 points 4.0 s instead of 0.6 s.
