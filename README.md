# Original RoboSense M1: pedestrian, cyclist and animal detection

## Latest project review â€” November15,2026

**[Complete project handoff, current source and evidence](reviews/2026-11-15/HANDOFF.md)** Â· [Review questions](reviews/2026-11-15/REVIEW-REQUEST.md)

The current app is0.9.17-ground-repeat. The package distinguishes deployed behavior, bounded tests, rejected changes and unresolved limitations. The four cases below are older diagnostic examples, not current-release benchmark outputs.

[Download the complete November15 review bundle](https://github.com/TwelveInchPianist/odyssey-lidar-samples/releases/download/review-2026-11-15/project-review-2026-11-15.zip).

## Original sample cases â€” October29,2026

Four real recording excerpts for technical review of an experimental **lidar-based pedestrian, cyclist and animal detector with audible alerts, particularly at night**. This is a custom pipeline, not an Autoware or OpenPCDet deployment, and it does not control the vehicle.

**Review request:** What is realistically achievable with this sensor configuration? Which existing perception methods fit, and what experiment would separate sensor limitations from software mistakes? Counterexamples and reasons to abandon an approach are welcome.

| Case | Observed result | Evidence |
|---|---|---|
| [01: Distant cyclist](cases/01-cyclist/README.md) | Daylight logged track begins about 95 m forward of vehicle origin. Earlier 115Ã¢â‚¬â€œ131 m point associations remain candidates. | [Data ZIP](https://github.com/TwelveInchPianist/odyssey-lidar-samples/releases/download/review-2026-10-29/01-cyclist-data.zip), [narrow preview](https://github.com/TwelveInchPianist/odyssey-lidar-samples/releases/download/review-2026-10-29/01-cyclist-narrow-preview.mp4), [wide preview](https://github.com/TwelveInchPianist/odyssey-lidar-samples/releases/download/review-2026-10-29/01-cyclist-wide-preview.mp4) |
| [02: Nighttime animal](cases/02-night-animal/README.md) | A small animal is visible in video; current detector/tracker produces no confirmed animal advisory. Its lidar returns have not been isolated reliably. | [Data ZIP](https://github.com/TwelveInchPianist/odyssey-lidar-samples/releases/download/review-2026-10-29/02-night-animal-data.zip), [narrow preview](https://github.com/TwelveInchPianist/odyssey-lidar-samples/releases/download/review-2026-10-29/02-night-animal-narrow-preview.mp4), [wide preview](https://github.com/TwelveInchPianist/odyssey-lidar-samples/releases/download/review-2026-10-29/02-night-animal-wide-preview.mp4) |
| [03: False animal](cases/03-false-animal/README.md) | Palm foliage produces a temporally confirmed camera animal advisory. | [Data ZIP](https://github.com/TwelveInchPianist/odyssey-lidar-samples/releases/download/review-2026-10-29/03-false-animal-data.zip), [wide preview](https://github.com/TwelveInchPianist/odyssey-lidar-samples/releases/download/review-2026-10-29/03-false-animal-wide-preview.mp4) |
| [04: Generic notices](cases/04-generic-notices/README.md) | Lidar-only offline replay produces 36 generic proposals in one minute, while not selecting the cyclist. | [Data ZIP](https://github.com/TwelveInchPianist/odyssey-lidar-samples/releases/download/review-2026-10-29/04-generic-notices-data.zip), camera source is case01 |

Read the [technical brief](TECHNICAL-BRIEF.md), [data format and timing notes](DATA-FORMAT.md), and [questions](QUESTIONS.md). Download one case at a time; the previews are small, and ZIPs contain original native camera segments and selected complete raw lidar scans.

These are selected diagnostic examples, not a labeled benchmark or an estimate of overall recall. Preview videos are resized and reencoded; original HEVC files are unchanged. Point-cloud previews and overlays are diagnostic. No geometric alignment is represented as ground truth. No raw data has been altered to make an object easier to detect.
