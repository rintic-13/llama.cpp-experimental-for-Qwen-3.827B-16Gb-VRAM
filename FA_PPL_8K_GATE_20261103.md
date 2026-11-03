# PPL gate at full 8K context - PASSED (thin margin), 2026-11-03

The frozen quality anchors (gold 7.1378, turbo4/turbo3 7.1719-7.1751, turbo5/turbo3 7.1868) were all
measured at `-c 4096`, where the 4097-token corpus overflows the window and the tail tokens are scored
in a truncated context. This gate re-runs the turbo-vs-f16 comparison at `-c 8192`, where every token
sees the full window, to confirm that the `-ctk turbo4 -ctv turbo3` production choice stays inside the
|dPPL| <= 0.05 tolerance at longer depth.

## Protocol

Identical to the frozen 4K protocol, only `-c 4096` -> `-c 8192`:

```
llama-perplexity.exe -m Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf -f ppl_eval\wiki.test.quick.txt \
    -c 8192 --chunks 1 -b 1 -ub 1 -ngl 99 -fa on -t 7 -s 42 -ctk <K> -ctv <V>
```

4097 tokens, 1 pass, seed 42. Raw `ppl_eval\ppl8k_*.log` files stay out of git (`.gitignore` `*.log`);
the final lines are reproduced below.

## Results (c=8192)

| row | K/V | PPL | delta vs gold |
|---|---|---|---|
| P8-3 gold | f16 / f16 | **7.2415 +/- 0.27601** | - |
| P8-1 production | turbo4 / turbo3 (scratch prefill) | **7.2900 +/- 0.27873** | **+0.0485 PASS** (limit 0.05) |
| P8-2 | turbo5 / turbo3 | 7.2863 +/- 0.27848 | +0.0448 PASS |
| P8-4 | turbo4 / turbo3 NATIVE vec | 7.2817 +/- 0.27806 | +0.0402 |

Side results:
- Native-vs-scratch bridge P8-4 - P8-1 = **-0.0083**, same magnitude as the 4K reference (+0.0083):
  the f16-scratch numerics and the in-kernel vec decode differ by a constant, not growing with depth -
  consistent with the closed S3 conclusion.
- Memory live confirmation of the closed S3: context 265 MiB at c=8192 with turbo4/turbo3 vs 661 MiB
  with f16 KV; compute buffer 32.3 MiB (scratch path) vs 6.5 MiB (native path); 3.9 GiB free on the
  16 GB card. No multi-GB f16 scratch anywhere.

## Verdict and consequences

- **PASSED, but the margin is thin**: dPPL grew from +0.0341 at c=4096 to +0.0485 at c=8192 -
  roughly +0.014 per window doubling. The tolerance is nearly consumed at 8K; a deeper honest test
  (>= 8192 scored tokens, e.g. `ppl_eval\wiki.ab.mixed_800k.txt`) would likely cross 0.05 at 16K+.
  This is quantization physics, not an implementation bug.
- These four numbers become the **frozen 8K anchors**. Any future change touching turbo-KV numerics
  (dither design, centroid tables beyond the already-gated S7/S8, KV type changes) must re-run the
  P8-1 / P8-3 pair and keep |dPPL| <= 0.05 against this pair, not only against the 4K anchors.
- If deeper-context quality ever becomes the binding constraint, the lever is the KV type (turbo5 K
  costs about +0.004 here, i.e. it is already near its own limit) or the dither design - not the
  attention kernels: S1-S8 are all closed.
