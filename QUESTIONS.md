# Questions for independent review

1. For an original M1 in this mounting, which claims about **presence**, **tracking**, and **classification** at50/100/150/200 m are physically and empirically reasonable? What evidence would be required for each, especially at night?
2. Which current method or implementation would you start from: temporal geometric detection, a lidar object detector, semantic segmentation, occupancy, or another representation? What changes would its training sensor and labels require for this M1?
3. What would you inspect first to separate ground/model/alignment mistakes from insufficient target returns? Are the selected raw clouds normal for the original M1?
4. How would you retain weak observations over time without incorrectly merging a new person/animal into an already-announced identity? Which current-image or current-point evidence is indispensable?
5. What defensible sensitivity/nuisance evaluation can be built from ordinary drives with no crashes, staged hazards or surveyed targets? Which claims must remain unknown?
6. What is the smallest experiment that could disprove your proposed approach, and what work would you stop doing here?

Please distinguish firsthand experience, results on another sensor, general hypotheses and directly reproducible findings in these excerpts. Relevant code, counterexamples and referrals are welcome.
