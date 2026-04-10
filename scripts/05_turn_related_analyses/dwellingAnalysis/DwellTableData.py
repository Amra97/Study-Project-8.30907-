import pandas as pd
import numpy as np

# ----------------------- Config -----------------------
data_path = 'C:/Users/ameer/Downloads/combined_dataframe.csv'   # <-- replace with your file path
window_seconds = 10
mad_z_thresh = 3.5           # outlier threshold for fixation durations
collider_col = 'Collider_CategoricalN'
save_csv = 'per_object_dwell_first_vs_second_with_category.csv'  # set to None to skip saving
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
    return per-window per-object dwell stats for 'names':
      - total_dwell (sum of fixation lengths in the window)
    Entry selection:
      - entry_nr==1: (entry_nr==1) & (isNewTurn==True)
      - entry_nr==2: (entry_nr==2) only (do not rely on isNewTurn)
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
                       .agg(total_dwell=('length', 'sum')))
            per_obj['SubjectID'] = subj
            per_obj['street_id'] = street_id
            per_obj['entry_nr'] = entry_nr_target
            per_obj['anchor_time'] = anchor_time
            rows.append(per_obj)
    if not rows:
        return pd.DataFrame(columns=['names','total_dwell','SubjectID','street_id','entry_nr','anchor_time'])
    return pd.concat(rows, ignore_index=True)

# Collect per-window stats separately for first and second entries
win_first = per_window_object_stats(entry_nr_target=1)
win_second = per_window_object_stats(entry_nr_target=2)

print(f"Windows with data - First entry:  {win_first[['SubjectID','street_id','anchor_time']].drop_duplicates().shape[0]}")
print(f"Windows with data - Second entry: {win_second[['SubjectID','street_id','anchor_time']].drop_duplicates().shape[0]}")

def per_object_aggregates(win_df: pd.DataFrame) -> pd.DataFrame:
    """
    From per-window per-object stats, compute per-object aggregates across all windows:
      - median_dwell_per_window
      - mean_dwell_per_window
      - sum_dwell_across_windows
      - appearances (number of windows where the object appeared)
    """
    if win_df.empty:
        return pd.DataFrame(columns=[
            'median_dwell_per_window','mean_dwell_per_window',
            'sum_dwell_across_windows','appearances'
        ])
    agg = (win_df
           .groupby('names')
           .agg(median_dwell_per_window=('total_dwell', 'median'),
                mean_dwell_per_window=('total_dwell', 'mean'),
                sum_dwell_across_windows=('total_dwell', 'sum'),
                appearances=('names', 'count')))
    return agg

obj_first = per_object_aggregates(win_first)
obj_second = per_object_aggregates(win_second)

# Build name -> category mapping (mode of Collider_CategoricalN)
cat_series = (df[['names', collider_col]]
              .dropna()
              .groupby('names')[collider_col]
              .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]))

# Merge into one table: First and Second entry side-by-side + category
all_names = sorted(set(obj_first.index).union(set(obj_second.index)))
table = pd.DataFrame(index=all_names)

# join aggregates (rename columns to indicate entry)
table = table.join(obj_first.rename(columns={
    'median_dwell_per_window': 'median_dwell_first',
    'mean_dwell_per_window': 'mean_dwell_first',
    'sum_dwell_across_windows': 'sum_dwell_first',
    'appearances': 'windows_first'
}))
table = table.join(obj_second.rename(columns={
    'median_dwell_per_window': 'median_dwell_second',
    'mean_dwell_per_window': 'mean_dwell_second',
    'sum_dwell_across_windows': 'sum_dwell_second',
    'appearances': 'windows_second'
}))

# add category
table = table.join(cat_series.rename(collider_col))

# optional: order columns
ordered_cols = [
    collider_col,
    'median_dwell_first', 'mean_dwell_first', 'sum_dwell_first', 'windows_first',
    'median_dwell_second', 'mean_dwell_second', 'sum_dwell_second', 'windows_second'
]
# keep only columns that exist (in case some sets are empty)
ordered_cols = [c for c in ordered_cols if c in table.columns]
table = table[ordered_cols]

# sort by the larger median dwell across entries (descending)
if {'median_dwell_first','median_dwell_second'}.issubset(table.columns):
    sort_key = table[['median_dwell_first','median_dwell_second']].max(axis=1)
    table = table.loc[sort_key.sort_values(ascending=False).index]

# round numeric columns for readability
num_cols = table.select_dtypes(include=[np.number]).columns
table[num_cols] = table[num_cols].round(3)

print("\nPer-object dwell table (first vs second entry) with categories:")
print(table.head(20).to_string())

# Save to CSV if desired
if save_csv:
    table.to_csv(save_csv, index_label='names')
    print(f"\nSaved table to: {save_csv}")