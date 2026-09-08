---
name: longwalk-sublength-recheck
description: Redo the HRP5P_LongWalk RPE sublength sweep (1/2/4/5/10 m) now that the covariance tuning is settled
metadata:
  type: project
---

Arnaud asked (2026-08-30) that the HRP5P_LongWalk sublength sweep over {1, 2, 4, 5, 10} m be run
again once the tuning was finalised, and treated it as an important standing note. The first sweep
picked 10 m, which is what `Projects/HRP5P_LongWalk/projectConfig.yaml` now sets.

Still outstanding as of 2026-08-31. The trigger has now arrived: `contact_process_position_xy` was
changed from 2.5e-7 to 1e-9 and the full final candidate was installed into
`/home/arnaud/devel/src/mc_rtc_configs/observers/MCKineticsObserver.yaml` (backup suffix
`.pre-anchor1e-9-20260831-173007`). That moved LongWalk's translation ratio 1.376 -> 1.218, so the
sublength that best represents the sequence may no longer be 10 m.

Note when redoing it: LongWalk is no longer a holdout — it entered the selection when the anchor
was tuned on it, so nothing in the 13-dataset set is held out any more.
