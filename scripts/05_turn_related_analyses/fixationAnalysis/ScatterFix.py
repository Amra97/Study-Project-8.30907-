import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

# ----------------------- Config -----------------------
data_path = 'C:/Users/ameer/Downloads/combined_dataframe.csv'   # <-- replace with your file path
window_seconds = 10
mad_z_thresh = 3.5           # outlier threshold for fixation durations
collider_col = 'Collider_CategoricalN'
save_csv = 'per_object_fixation_first_vs_second_with_category.csv'  # set to None to skip saving
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

def per_window_object_fixation(entry_nr_target: int) -> pd.DataFrame:
    """
    For a given entry number (1 or 2), build 10s windows after each entry anchor and
    return per-window per-object fixation stats for 'names':
      - avg_fixation_this_window: mean fixation duration for the object within the window
      - fixation_count: number of fixations for the object within the window
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
                       .agg(avg_fixation_this_window=('length', 'mean'),
                            fixation_count=('length', 'count')))
            per_obj['SubjectID'] = subj
            per_obj['street_id'] = street_id
            per_obj['entry_nr'] = entry_nr_target
            per_obj['anchor_time'] = anchor_time
            rows.append(per_obj)

    if not rows:
        return pd.DataFrame(columns=['names','avg_fixation_this_window','fixation_count',
                                     'SubjectID','street_id','entry_nr','anchor_time'])
    return pd.concat(rows, ignore_index=True)

# Collect per-window fixation stats separately for first and second entries
win_first = per_window_object_fixation(entry_nr_target=1)
win_second = per_window_object_fixation(entry_nr_target=2)

print(f"Windows with data - First entry:  {win_first[['SubjectID','street_id','anchor_time']].drop_duplicates().shape[0]}")
print(f"Windows with data - Second entry: {win_second[['SubjectID','street_id','anchor_time']].drop_duplicates().shape[0]}")

def per_object_fixation_aggregates(win_df: pd.DataFrame) -> pd.DataFrame:
    """
    From per-window per-object fixation stats, compute per-object aggregates across windows:
      - mean_avg_fixation_per_window: mean of per-window average fixation durations
      - median_avg_fixation_per_window: median of per-window average fixation durations
      - appearances: number of windows where the object appeared
      - total_fixations: sum of fixation counts across windows
    """
    if win_df.empty:
        return pd.DataFrame(columns=['mean_avg_fixation_per_window',
                                     'median_avg_fixation_per_window',
                                     'appearances','total_fixations'])
    agg = (win_df
           .groupby('names')
           .agg(mean_avg_fixation_per_window=('avg_fixation_this_window', 'mean'),
                median_avg_fixation_per_window=('avg_fixation_this_window', 'median'),
                appearances=('names', 'count'),
                total_fixations=('fixation_count', 'sum')))
    return agg

obj_first = per_object_fixation_aggregates(win_first)
obj_second = per_object_fixation_aggregates(win_second)

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
    'mean_avg_fixation_per_window': 'mean_fix_first',
    'median_avg_fixation_per_window': 'median_fix_first',
    'appearances': 'windows_first',
    'total_fixations': 'total_fixations_first'
}))
table = table.join(obj_second.rename(columns={
    'mean_avg_fixation_per_window': 'mean_fix_second',
    'median_avg_fixation_per_window': 'median_fix_second',
    'appearances': 'windows_second',
    'total_fixations': 'total_fixations_second'
}))

# add category
table = table.join(cat_series.rename(collider_col))

# order columns (keep only those that exist)
ordered_cols = [
    collider_col,
    'mean_fix_first', 'median_fix_first', 'total_fixations_first', 'windows_first',
    'mean_fix_second', 'median_fix_second', 'total_fixations_second', 'windows_second'
]
ordered_cols = [c for c in ordered_cols if c in table.columns]
table = table[ordered_cols]

# sort by the larger mean fixation duration across entries (descending)
if {'mean_fix_first','mean_fix_second'}.issubset(table.columns):
    sort_key = table[['mean_fix_first','mean_fix_second']].max(axis=1)
    table = table.loc[sort_key.sort_values(ascending=False).index]

# round numeric columns for readability
num_cols = table.select_dtypes(include=[np.number]).columns
table[num_cols] = table[num_cols].round(3)

print("\nPer-object fixation table (first vs second entry) with categories:")
print(table.head(30).to_string())


# --- Build table with mean fixation time for first & second entry ---

# Make sure these exist from your previous code:
# obj_first, obj_second  (indexes = object names, column 'mean_avg_fixation_per_window')

first_mean = obj_first['mean_avg_fixation_per_window'].rename('mean_fix_first')
second_mean = obj_second['mean_avg_fixation_per_window'].rename('mean_fix_second')

# Combine into one DataFrame; keep only objects that appear in at least one entry
scatter_df = pd.concat([first_mean, second_mean], axis=1)

# If you want only objects that have data in BOTH visits, uncomment the next line:
# scatter_df = scatter_df.dropna(subset=['mean_fix_first', 'mean_fix_second'])

scatter_df = scatter_df.reset_index().rename(columns={'names': 'object'})

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# obj_first and obj_second already exist from your code
# (index = object name, column 'mean_avg_fixation_per_window')

# --- 1. Select top-10 objects by mean fixation in each visit ---

top10_first_names = (
    obj_first['mean_avg_fixation_per_window']
    .sort_values(ascending=False)
    .head(10)
    .index.tolist()
)

top10_second_names = (
    obj_second['mean_avg_fixation_per_window']
    .sort_values(ascending=False)
    .head(10)
    .index.tolist()
)

# Union of both sets of objects
selected_names = sorted(set(top10_first_names) | set(top10_second_names))

# --- 2. Build table with mean fixation times for those objects ---

first_mean = obj_first['mean_avg_fixation_per_window'].rename('mean_fix_first')
second_mean = obj_second['mean_avg_fixation_per_window'].rename('mean_fix_second')

scatter_df = pd.concat([first_mean, second_mean], axis=1)

# Keep only the selected objects
scatter_df = scatter_df.loc[scatter_df.index.intersection(selected_names)]

# Require data in BOTH entries for the scatter plot
scatter_df = scatter_df.dropna(subset=['mean_fix_first', 'mean_fix_second'])

scatter_df = scatter_df.reset_index().rename(columns={'names': 'object'})

# Optional: add Collider_CategoricalN category for colouring
if 'Collider_CategoricalN' in df.columns:
    cat_series = (
        df[['names', 'Collider_CategoricalN']]
        .dropna()
        .groupby('names')['Collider_CategoricalN']
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
    )
    scatter_df = scatter_df.merge(
        cat_series.rename('category'),
        left_on='object',
        right_index=True,
        how='left'
    )

print("\nScatter-plot data (top-10 first & second entry, first rows):")
print(scatter_df.head().to_string(index=False))

# --- 3. Scatter plot for selected objects ---

sns.set(style='whitegrid')
plt.figure(figsize=(7, 7))

if 'category' in scatter_df.columns:
    sns.scatterplot(
        data=scatter_df,
        x='mean_fix_first',
        y='mean_fix_second',
        hue='category',
        s=70,
        alpha=0.9
    )
else:
    sns.scatterplot(
        data=scatter_df,
        x='mean_fix_first',
        y='mean_fix_second',
        color='steelblue',
        s=70,
        alpha=0.9
    )

# y = x reference line
finite_vals = scatter_df[['mean_fix_first', 'mean_fix_second']].to_numpy()
finite_mask = np.isfinite(finite_vals).all(axis=1)
if finite_mask.any():
    max_val = float(np.nanmax(finite_vals[finite_mask]))
    plt.plot([0, max_val], [0, max_val], 'r--', label='y = x')

plt.xlabel('Mean fixation duration (First entry, s)')
plt.ylabel('Mean fixation duration (Second entry, s)')
plt.title('Top-10 Objects by Mean Fixation: First vs Second Entry')
plt.legend(loc='best', fontsize=8)
plt.tight_layout()
plt.show()