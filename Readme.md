## Quality Estimation
A video quality estimation module. The module uses one of the lastest state-of-the-art video quality estimation model, CONVIQT[1]. The code in this repository is mostly copied from the original repository with a few improvements.

---

### Download:
- Download the weights necessary to run the quality estimator
```
mkdir models
wget -L https://utexas.box.com/shared/static/rhpa8nkcfzpvdguo97n2d5dbn4qb03z8.tar -O models/CONTRIQUE_checkpoint25.tar -q --show-progress
wget -L https://utexas.box.com/shared/static/7s8348b0imqe27qkgq8lojfc2od1631a.tar -O models/CONVIQT_checkpoint10.tar -q --show-progress
wget -L https://github.com/pavancm/CONVIQT/raw/main/models/LIVE_ETRI.save -O models/LIVE_ETRI.save -q --show-progress
wget -L https://github.com/pavancm/CONVIQT/raw/main/models/LIVE_YT_HFR.save -O models/LIVE_YT_HFR.save -q --show-progress
wget -L https://github.com/pavancm/CONVIQT/raw/main/models/YouTube_UGC.save -O models/YouTube_UGC.save -q --show-progress
```

---

### Working:
- We estimate the quality of a video by splitting it video segments of length 1 or 2 seconds.
- We calculate quality of each video segment over time.

Example Code to get scores:
```
from estimate_quality import estimate_quality

F = estimate_quality()
F.get_scores(video_path)
```

---

### References:
[1] Madhusudana, Pavan C., et al. "Conviqt: Contrastive video quality estimator." IEEE Transactions on Image Processing (2023).
[2] https://github.com/pavancm/CONVIQT
