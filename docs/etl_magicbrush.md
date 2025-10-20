# MagicBrush (InstructPix2Pix) ETL Plan

## Goal

Convert MagicBrush / InstructPix2Pix edit triples into EE trajectories with deterministic guardrails for visual consistency, enabling artistry agents to learn edit strategies and feed ACE with reliable corrections.

## Source Layout

Each record provides:

- `source_image`: path/URL/bytes of the original image
- `edit_instruction`: natural-language description of the desired edit
- `target_image`: resulting image after the edit
- Optional tags (e.g., “remove object”, “change color”)

## Target Schema (per JSONL line)

```json
{
  "task_id": "mb-<hash>",
  "state": {
    "image_uri": "s3://.../source.png",
    "instruction": "Replace the sky with a sunset",
    "metadata": {
      "width": 512,
      "height": 512,
      "channels": 3
    }
  },
  "action": {
    "edit_prompt": "Replace the sky with a vibrant orange sunset"
  },
  "next_state": {
    "image_uri": "s3://.../target.png",
    "checksum": "<sha256>",
    "metrics": {
      "lpips": 0.18,
      "ssim": 0.82,
      "dominant_colors": ["#f8721d", "#1d3f8c"]
    }
  },
  "ground_truth": "lpips<=0.25 && ssim>=0.75",
  "guardrail": {
    "instructions": "Verify perceptual distance <= 0.25 and structural similarity >= 0.75. Clamp answer to 'pass' or 'fail'.",
    "value": "pass",
    "format": "string"
  }
}
```

### Notes

- Store images in accessible object storage or embed base64 (if size permits). Include dimensions so the environment can pre-validate inputs.
- `action.edit_prompt` mirrors the instruction; future refinements may include parameter tweaks (strength, guidance scale).
- `next_state.metrics` collect deterministic observations used by the guardrail.

## Deterministic Guardrails

1. Load `source_image` and `target_image`.
2. Compute perceptual metrics (LPIPS, SSIM) and optionally color histogram divergence.
3. Compare against thresholds (`lpips <= 0.25`, `ssim >= 0.75`, tweak per domain).
4. Optionally assert resolution/format invariants (same size, no alpha channel changes).
5. Output canonical `pass` or `fail`, with detailed logs for ACE.

## ETL Steps

1. **Extract**: iterate over MagicBrush records, download or reference images in storage.
2. **Transform**:
   - Upload images to your bucket (or encode as URIs) and compute hashes.
   - Run metric calculators (LPIPS/SSIM); store in `next_state.metrics`.
   - Assemble JSONL records with guardrail thresholds.
3. **Load**: write to `benchmarks/magicbrush.jsonl`.
4. **Scaffold**: `python scripts/scaffold_domain.py magicbrush --from-benchmark benchmarks/magicbrush.jsonl`.
5. **Verify**: run `python scripts/run_benchmark.py benchmarks/magicbrush.jsonl --domain magicbrush --offline` to confirm guardrail coverage; online runs require access to the image store and metric calculators.

## ACE Integration

- Guardrail corrections capture cases where edits drift too far (“LPIPS > 0.25”). ACE can store reusable heuristics (“Keep SSIM ≥ 0.75 when swapping skies”) and feed them back into future edit prompts.
