# Capture guide (about 3 minutes per part)

You need the printed scale card (`capture/scale_card_A4.pdf` at 100 % scale, ruler checked), a phone and calipers.

## Set-up
- Put the card flat on a table, out of direct sun. Diffuse daylight or two lamps from different sides work best.
- Place the part on the **grey stage** in the middle of the card, resting on its largest flat face if it has one.
- Use the phone's main (1×) camera. Do not use portrait mode, zoom or filters. Turn HDR off if you can.
- Clean the lens.

## Photos of the part (5–12 per part)
Keep the **whole grey stage plus at least two rows of markers** in every frame. Fill as much of the frame as you can while doing that.

| # | View | How |
|---|---|---|
| 1–2 | Top-down | Hold the phone flat, about 20 cm above the part. Take one photo, then rotate yourself 90° and take another. |
| 3–6 | Oblique | Hold the phone 40–65° above the table. Go around the part and take one photo every 90°. |
| 7–8 | Low | Hold the phone 25–35° above the table, from two opposite sides. The card markers must still be readable. |
| 9–12 | Optional | Repeat the oblique shots with the part flipped onto another face. Put these in a sub-folder `flip1/`. |

Tap the screen on the part to focus. Every photo must be sharp; run the check below.

## Photos of the mating location (2–4)
These are photos of where the part attaches: the shaft, the hole in a housing, the rack wire, the slot.
- If the mating item can be taken off (for example a shaft or a wire), put it on the card and photograph it like a part.
- If it cannot be taken off, hold the card flat right next to it so that the card and the feature are in the same plane. Take a straight-on photo and one at 45°.
- Put these in `data/real/<part_id>/mating/`.

## Caliper measurements (most important)
Photos give the shape and the feature types. Fit-critical sizes need about 0.05 mm accuracy, which photos cannot give: the synthetic benchmark measures a median 1.2 mm error on hole diameters (`results/m1/metrology_synth.json`). Measure these with calipers:
- every hole, bore, pin or shaft diameter, on **both** the part and the mating item;
- D-shaft across-flat;
- slot and wire widths;
- wall or beam thickness of snap hooks;
- overall height.

Copy `data/real/_template/measurements.yaml` to `data/real/<part_id>/measurements.yaml` and fill it in.

## Check before you leave the part
```
.venv/bin/python -m spare.capture.quality data/real/<part_id>/photos
```
Retake any photo that the check flags: blurry, glare, or card not visible enough.

## Privacy
Photos are git-ignored and never committed. Keep faces, documents and screens out of the frame.
