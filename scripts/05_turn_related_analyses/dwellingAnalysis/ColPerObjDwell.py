import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------- Config -----------------------
data_path = 'C:/Users/ameer/Downloads/combined_dataframe.csv'   # <-- replace with your file path
window_seconds = 10
top_n = 30                   # limit bars in plots for readability; set to None for all
mad_z_thresh = 3.5           # outlier threshold for fixation durations
collider_col = 'Collider_CategoricalN'  # exact column for object categories
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

# Ensure the Collider_CategoricalN column exists
if collider_col not in df.columns:
    raise ValueError(f"Required category column '{collider_col}' not found in the data.")
else:
    print(f"Using collider category column: {collider_col}")

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
      - median_dwell_per_window: median of total_dwell across windows
      - mean_dwell_per_window: mean of total_dwell across windows (reference)
      - appearances: number of windows where the object appeared
    """
    if win_df.empty:
        return pd.DataFrame(columns=['median_dwell_per_window','mean_dwell_per_window','appearances'])
    agg = (win_df
           .groupby('names')
           .agg(median_dwell_per_window=('total_dwell', 'median'),
                mean_dwell_per_window=('total_dwell', 'mean'),
                appearances=('names', 'count'))
           .sort_values('median_dwell_per_window', ascending=False))
    return agg

obj_first = per_object_aggregates(win_first)
obj_second = per_object_aggregates(win_second)

print("\nPer-object (first entry) — top 10 by median dwell:")
print(obj_first.head(10))
print("\nPer-object (second entry) — top 10 by median dwell:")
print(obj_second.head(10))

# Build name -> category mapping (mode of Collider_CategoricalN)
tmp = (df[['names', collider_col]]
       .dropna()
       .groupby('names')[collider_col]
       .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]))
name_to_cat = tmp.to_dict()

def plot_dwell_with_categories(
    series: pd.Series,
    title: str,
    ylabel: str,
    color='steelblue',
    top_n=None,
    label_mode='category_below',       # 'category_below' (default), 'category_above', or 'category_only'
    max_chars=22,                      # truncate long labels for readability
    fontsize=9,
    rotation=0                         # keep 0 degrees to improve readability for two-line labels
):
    s = series.dropna().sort_values(ascending=False)
    if top_n is not None and len(s) > top_n:
        s = s.head(top_n)

    # Construct two-line labels with category placed BELOW the object name (default)
    labels = []
    for name in s.index:
        cat = str(name_to_cat.get(name, '—'))
        # Optional truncation to keep labels readable
        nm = str(name)
        if max_chars and len(nm) > max_chars:
            nm = nm[:max_chars - 1] + '…'
        if max_chars and len(cat) > max_chars:
            cat = cat[:max_chars - 1] + '…'

        if label_mode == 'category_below':
            lbl = f"{nm}\n{cat}"            # category on the second line (below)
        elif label_mode == 'category_above':
            lbl = f"{cat}\n{nm}"            # category on the first line (above)
        elif label_mode == 'category_only':
            lbl = f"{cat}"
        else:
            lbl = f"{nm}\n{cat}"
        labels.append(lbl)

    # Make the figure wider as the number of bars grows, to avoid cramped text
    fig_w = max(12, 0.6 * len(s))
    plt.figure(figsize=(fig_w, 6))
    x = np.arange(len(s))
    plt.bar(x, s.values, color=color, edgecolor='none')
    plt.xticks(x, labels, rotation=rotation, ha='center', fontsize=fontsize)
    plt.xlabel(f'Object (names) and {collider_col}')
    plt.ylabel(ylabel)
    plt.title(title)
    # Add extra bottom margin for two-line labels
    plt.subplots_adjust(bottom=0.28)
    plt.grid(axis='y', linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.show()

# Plots for first entry (DWELL ONLY) with category below the object name for readability
if not obj_first.empty:
    plot_dwell_with_categories(
        obj_first['median_dwell_per_window'],
        'Median Dwell Time per Object (First Entry, 10s after turn)',
        'Median per-window dwell time (s)',
        color='#2ca02c',
        top_n=top_n,
        label_mode='category_below',   # category appears on the line below the object name
        fontsize=9,
        rotation=0
    )
else:
    print("No per-object data for first entry.")

# Plots for second entry (DWELL ONLY)
if not obj_second.empty:
    plot_dwell_with_categories(
        obj_second['median_dwell_per_window'],
        'Median Dwell Time per Object (Second Entry, 10s after turn)',
        'Median per-window dwell time (s)',
        color='#d62728',
        top_n=top_n,
        label_mode='category_below',
        fontsize=9,
        rotation=0
    )
else:
    print("No per-object data for second entry.")

# Optional: save results
# obj_first.to_csv('per_object_first_entry_dwell.csv')
# obj_second.to_csv('per_object_second_entry_dwell.csv')