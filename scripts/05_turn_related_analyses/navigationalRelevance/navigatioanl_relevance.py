import pandas as pd
import numpy as np

# ----------------------- Config -----------------------
data_path = 'C:/Users/ameer/Downloads/combined_dataframe.csv'   # <-- replace with your file path
window_seconds = 10
mad_z_thresh = 3.5           # outlier threshold for fixation durations
collider_col = 'Collider_CategoricalN'
min_windows_for_landmark = 5     # how often an object must appear to be considered
landmark_quantile = 0.75         # top 25% by overall mean dwell -> candidate "navigational"
save_csv = 'nav_significance_dwell_fix_per_object.csv'  # set to None to skip saving
# ------------------------------------------------------

# Load data
df = pd.read_csv(data_path)

# Choose time column
time_col = 'total_time' if 'total_time' in df.columns else 'Continuous_Time'
if time_col not in df.columns:
    raise ValueError("No usable time column found. Expected 'total_time' or 'Continuous_Time'.")

# Basic column checks
required_cols = [
    'SubjectID', time_col, 'events', 'length', 'names',
    'entry_nr', 'street_id_within_participant', 'isNewTurn'
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if collider_col not in df.columns:
    raise ValueError(f"Required category column '{collider_col}' not found in the data.")

# Sort for time-consistent windowing
df = df.sort_values(['SubjectID', time_col]).reset_index(drop=True)

# Keep only fixation events (events==2) and remove outliers via MAD
gaze = df[df['events'] == 2].copy()
if len(gaze) == 0:
    raise ValueError("No fixation rows found (events == 2).")

mad = np.nanmedian(np.abs(gaze['length'] - np.nanmedian(gaze['length']))) * 1.4826
scale = mad if mad > 0 else 1.0
gaze['gaze_mad_z'] = np.abs(gaze['length'] - np.nanmedian(gaze['length'])) / scale
gaze_no_out = gaze[gaze['gaze_mad_z'] <= mad_z_thresh].copy()

def per_window_object_stats(entry_nr_target: int) -> pd.DataFrame:
    """
    For a given entry number (1 or 2), build 10s windows after each entry anchor and
    return per-window per-object stats:
      - total_dwell: sum of fixation durations in that window for the object
      - fixation_count: number of fixations in that window for the object
      - mean_fixation_this_window: mean fixation duration in that window for the object
    """
    rows = []
    for subj, df_s in df.groupby('SubjectID', sort=False):
        gz = gaze_no_out[gaze_no_out['SubjectID'] == subj]
        if gz.empty:
            continue
        for street_id, df_ss in df_s.groupby('street_id_within_participant', sort=False):
            if pd.isna(street_id):
                continue
            if entry_nr_target == 1:
                anchors = df_ss[(df_ss['entry_nr'] == 1) & (df_ss['isNewTurn'] == True)]
            else:
                anchors = df_ss[(df_ss['entry_nr'] == 2)]
            if anchors.empty:
                continue

            anchor_time = anchors[time_col].min()
            in_win = (gz[time_col] >= anchor_time) & (gz[time_col] <= anchor_time + window_seconds)
            win = gz.loc[in_win, ['names', 'length', time_col]]
            if win.empty:
                continue

            per_obj = (win
                       .groupby('names', as_index=False)
                       .agg(total_dwell=('length', 'sum'),
                            fixation_count=('length', 'count'),
                            mean_fixation_this_window=('length', 'mean')))
            per_obj['SubjectID'] = subj
            per_obj['street_id'] = street_id
            per_obj['entry_nr'] = entry_nr_target
            per_obj['anchor_time'] = anchor_time
            rows.append(per_obj)

    if not rows:
        return pd.DataFrame(columns=[
            'names','total_dwell','fixation_count','mean_fixation_this_window',
            'SubjectID','street_id','entry_nr','anchor_time'
        ])
    return pd.concat(rows, ignore_index=True)

# Per-window stats for first and second entry
win_first = per_window_object_stats(entry_nr_target=1)
win_second = per_window_object_stats(entry_nr_target=2)

print(f"Windows with data - First entry:  {win_first[['SubjectID','street_id','anchor_time']].drop_duplicates().shape[0]}")
print(f"Windows with data - Second entry: {win_second[['SubjectID','street_id','anchor_time']].drop_duplicates().shape[0]}")

def per_object_aggregates(win_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-window object stats into per-object metrics:
      - mean_dwell_per_window, median_dwell_per_window
      - mean_fix_duration (overall)
      - total_fixations
      - windows_seen: number of windows in which the object appeared
    """
    if win_df.empty:
        return pd.DataFrame(columns=[
            'mean_dwell_per_window','median_dwell_per_window',
            'mean_fix_duration','total_fixations','windows_seen'
        ])
    agg = (win_df
           .groupby('names')
           .agg(mean_dwell_per_window=('total_dwell', 'mean'),
                median_dwell_per_window=('total_dwell', 'median'),
                mean_fix_duration=('mean_fixation_this_window', 'mean'),
                total_fixations=('fixation_count', 'sum'),
                windows_seen=('names', 'count')))
    return agg

obj_first = per_object_aggregates(win_first)
obj_second = per_object_aggregates(win_second)

print("\nPer-object (first entry) — head:")
print(obj_first.head().to_string())
print("\nPer-object (second entry) — head:")
print(obj_second.head().to_string())

# ---------- Add categories and combine first & second entry ----------

# Map objects to Collider_CategoricalN (mode across all occurrences)
cat_series = (df[['names', collider_col]]
              .dropna()
              .groupby('names')[collider_col]
              .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]))

all_names = sorted(set(obj_first.index).union(set(obj_second.index)))
table = pd.DataFrame(index=all_names)

# Join first-entry stats
table = table.join(obj_first.add_suffix('_first'))
# Join second-entry stats
table = table.join(obj_second.add_suffix('_second'))
# Add category
table = table.join(cat_series.rename(collider_col))

# Convenience columns
table['overall_mean_dwell'] = table[['mean_dwell_per_window_first',
                                     'mean_dwell_per_window_second']].mean(axis=1)
table['delta_mean_dwell'] = (table['mean_dwell_per_window_second'] -
                             table['mean_dwell_per_window_first'])
table['delta_mean_fix_dur'] = (table['mean_fix_duration_second'] -
                               table['mean_fix_duration_first'])

# Flag candidate "navigationally significant" objects:
#   - appear in at least min_windows_for_landmark windows (across both entries)
#   - in the top landmark_quantile (e.g., 75th percentile) of overall_mean_dwell
table['total_windows_both'] = (table['windows_seen_first'].fillna(0) +
                               table['windows_seen_second'].fillna(0))
dwell_threshold = table['overall_mean_dwell'].quantile(landmark_quantile)
table['nav_candidate'] = (
    (table['overall_mean_dwell'] >= dwell_threshold) &
    (table['total_windows_both'] >= min_windows_for_landmark)
)

# Sort by overall mean dwell (descending)
table = table.sort_values('overall_mean_dwell', ascending=False)

# Round numerical columns for cleaner viewing
num_cols = table.select_dtypes(include=[np.number]).columns
table[num_cols] = table[num_cols].round(3)

print("\nCombined per-object table (first vs second entry) with nav_candidate flag:")
print(table.head(30).to_string())

print(f"\nNumber of candidate 'navigational' objects (by dwell criteria): "
      f"{table['nav_candidate'].sum()}")

print("\nTop candidate navigational objects:")
print(table[table['nav_candidate']].head(20).to_string())

# Save to CSV if desired
if save_csv:
    table.to_csv(save_csv, index_label='object_name')
    print(f"\nSaved full table to: {save_csv}")

from scipy.stats import ttest_rel
import numpy as np

# ----------------------------------------------------------
# Helper: run paired t-test and compute Cohen's d (paired)
# ----------------------------------------------------------
def paired_ttest(x, y, label):
    """
    x, y: pandas Series or 1D arrays of same length
    label: description string for printing
    """
    # Drop NaNs pairwise
    paired = np.array([x.values, y.values], dtype=float).T
    mask = np.isfinite(paired).all(axis=1)
    x_clean = paired[mask, 0]
    y_clean = paired[mask, 1]

    print(f"\n--- {label} ---")
    print(f"Number of paired objects: {len(x_clean)}")

    if len(x_clean) < 2:
        print("Not enough data for a t-test.")
        return

    # Paired t-test
    t_stat, p_val = ttest_rel(x_clean, y_clean)
    diff = y_clean - x_clean
    d = diff.mean() / diff.std(ddof=1)  # Cohen's d for paired samples

    print(f"Mean first entry : {x_clean.mean():.3f}")
    print(f"Mean second entry: {y_clean.mean():.3f}")
    print(f"Mean difference (second - first): {diff.mean():.3f}")
    print(f"t = {t_stat:.3f}, p = {p_val:.5f}, Cohen's d = {d:.3f}")


# ----------------------------------------------------------
# 1) Dwell times per object (mean dwell per window)
# ----------------------------------------------------------
paired_ttest(
    table['mean_dwell_per_window_first'],
    table['mean_dwell_per_window_second'],
    label="Mean dwell time per object (per-window mean) – First vs Second entry"
)

# ----------------------------------------------------------
# 2) Fixation durations per object (mean fixation duration)
# ----------------------------------------------------------
if {'mean_fix_duration_first', 'mean_fix_duration_second'}.issubset(table.columns):
    paired_ttest(
        table['mean_fix_duration_first'],
        table['mean_fix_duration_second'],
        label="Mean fixation duration per object – First vs Second entry"
    )
else:
    print("\nFixation duration columns not found in 'table'.")

# ----------------------------------------------------------
# 3) Total number of fixations per object
# ----------------------------------------------------------
if {'total_fixations_first', 'total_fixations_second'}.issubset(table.columns):
    paired_ttest(
        table['total_fixations_first'],
        table['total_fixations_second'],
        label="Total number of fixations per object – First vs Second entry"
    )
else:
    print("\nTotal fixation count columns not found in 'table'.")