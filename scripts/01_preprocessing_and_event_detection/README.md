## Scripts

- `MoveCorr_AngVel_HeadVel_multiple_participants.ipynb` — Preprocesses the raw multi-participant eye-tracking data and computes core movement variables used throughout the project. This includes movement correction, angular gaze velocity, and head angular velocity measures that form the basis for later fixation and saccade classification.

- `add_objectcategorymerged.ipynb` — Adds merged object-category labels to the dataset so that object-based analyses can be performed on a cleaner and more interpretable set of categories.

- `context_extraction.ipynb` — Extracts contextual variables around gaze events, such as surrounding fixation and saccade properties, object transitions, novelty measures, distance, and head-movement context. These derived variables are later used in the Short/Long and Fast/Slow contextual analyses.

- `merge_participants_add_shortlong_classification.ipynb` — Merges participant-level data into a shared analysis dataset and adds the duration-based Short/Long saccade classification.


