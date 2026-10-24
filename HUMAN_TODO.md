# HUMAN_TODO

Status legend: `awaiting-human` = blocked on you; `done` = completed (write the date).

## Batch 0 — Shopping list and setup  `awaiting-human`
1. An FDM printer (any common model; 0.4 mm nozzle). Network/CLI control is a bonus. Tell me the model and nozzle by adding a line to ENVIRONMENT.md under "Printer".
2. Digital calipers (0.01 mm resolution).
3. PLA (1 kg) and PETG (1 kg), 1.75 mm.
4. A sheet of matte white printer paper and access to a laser printer (for the scale card, Batch 1).

## Batch 1 — Print the scale card  `awaiting-human` (card generated in M1)
- Print `capture/scale_card_A4.pdf` at **100% / "actual size"** (no "fit to page").
- Check the printed 100 mm ruler on the card with calipers. Write the measured length in `data/real/card_check.txt`, e.g. `ruler_100mm_measured: 99.85`.
- Tape the card flat to stiff cardboard.

## Batch 2 — Collect parts  `awaiting-human`
Collect **20–100** small household plastic parts for which you have a *working original*:
- Good types: clips, knobs, end caps, brackets, battery covers, hinges, spacers, shelf pegs, cable clips, rack clips.
- Collect the **mating context** where you can: the shaft, tube, rack wire, housing or hole the part attaches to.
- For about a third of them, print a duplicate later (Batch 5) and snap it, to make broken/intact pairs.
- **Do not collect** any of the following; the pipeline refuses them:
  - parts that bear a person's load;
  - vehicle, brake or steering parts;
  - child-safety items (car seats, cribs, gates);
  - medical devices;
  - gas, fuel or pressure parts;
  - mains-electrical enclosures;
  - parts used near heat sources.
- Give each part an ID `p001`, `p002`, …: write it on masking tape and stick it on the part.

## Batch 3 — Photograph and measure each part  `awaiting-human` (guide generated in M1)
Follow `capture/CAPTURE_GUIDE.md`, which lists 5–12 photos per part plus 2–4 photos of the mating location. Put the files in `data/real/<part_id>/photos/`; they are git-ignored and never committed. Fill in `data/real/<part_id>/measurements.yaml` from the template the guide links.

## Batch 4 — Print and test  `awaiting-human` (queued by the agent in PRINT_QUEUE.md)
For each row in PRINT_QUEUE.md:
1. Print the file with the listed slicer profile.
2. Try the part in place.
3. Record the result with the 30-second fit form: `python -m spare.feedback.fit_form --part <id> --attempt <n>`.
