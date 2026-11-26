# Step-detector (CNN) training

```
python train_step.py --quick   # smoke test
python train_step.py           # full run: 25,000 steps, ~86 min on one GPU
```
Records are simulated on the fly by `sim2_torch.py` (2-3 traps, coupling, common drive, boxcar,
drift, 1/f noise). Output: `out/stepnet.pt`, `out/eval.json`, `out/train_log.txt`.
The trained weights used in the paper are in `../models/stepnet.pt`.
