import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
      - median_dwell_per_window
      - mean_dwell_per_window
      - appearances
    """
    if win_df.empty:
        return pd.DataFrame(columns=['median_dwell_per_window','mean_dwell_per_window','appearances'])
    agg = (win_df
           .groupby('names')
           .agg(median_dwell_per_window=('total_dwell', 'median'),
                mean_dwell_per_window=('total_dwell', 'mean'),
                appearances=('names', 'count')))
    return agg

obj_first = per_object_aggregates(win_first)
obj_second = per_object_aggregates(win_second)

print("\nPer-object (first entry) — top 10 by median dwell:")
print(obj_first.sort_values('median_dwell_per_window', ascending=False).head(10))
print("\nPer-object (second entry) — top 10 by median dwell:")
print(obj_second.sort_values('median_dwell_per_window', ascending=False).head(10))

# ----------------------- NEW: category-level dwell & scatter -----------------------

# 1. Map each object name to its Collider_CategoricalN category (mode across rows)
name_to_cat = (
    df[['names', collider_col]]
    .dropna()
    .groupby('names')[collider_col]
    .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
    .to_dict()
)

# Add category to object-level tables
obj_first_cat = obj_first.copy()
obj_first_cat['category'] = obj_first_cat.index.map(name_to_cat)

obj_second_cat = obj_second.copy()
obj_second_cat['category'] = obj_second_cat.index.map(name_to_cat)

# Drop objects with unknown category (NaN)
obj_first_cat = obj_first_cat.dropna(subset=['category'])
obj_second_cat = obj_second_cat.dropna(subset=['category'])

# 2. Aggregate dwell per CATEGORY for first and second visits
#    (using mean of the object-level mean_dwell_per_window as a simple summary)
cat_first = (
    obj_first_cat
    .groupby('category')
    .agg(mean_dwell_first=('mean_dwell_per_window', 'mean'),
         median_dwell_first=('median_dwell_per_window', 'median'),
         n_objects_first=('mean_dwell_per_window', 'size'))
)

cat_second = (
    obj_second_cat
    .groupby('category')
    .agg(mean_dwell_second=('mean_dwell_per_window', 'mean'),
         median_dwell_second=('median_dwell_per_window', 'median'),
         n_objects_second=('mean_dwell_per_window', 'size'))
)

# 3. Combine first & second visit category stats into one table for scatter
all_cats = sorted(set(cat_first.index) | set(cat_second.index))
cat_table = pd.DataFrame(index=all_cats)
cat_table = cat_table.join(cat_first[['mean_dwell_first']])
cat_table = cat_table.join(cat_second[['mean_dwell_second']])

# Option: keep only categories with data in both visits
cat_table_plot = cat_table.dropna(subset=['mean_dwell_first', 'mean_dwell_second'])

print("\nCategory-level dwell table used for scatter plot:")
print(cat_table_plot.round(3).to_string())

# 4. Scatter plot: dwelling time per category, first vs second visit
sns.set(style='whitegrid')

plt.figure(figsize=(7, 7))
plt.scatter(
    cat_table_plot['mean_dwell_first'],
    cat_table_plot['mean_dwell_second'],
    s=80,
    alpha=0.9,
    color='steelblue'
)

# Label each point with the category name
for cat, row in cat_table_plot.iterrows():
    plt.text(
        row['mean_dwell_first'],
        row['mean_dwell_second'],
        str(cat),
        fontsize=8,
        ha='left',
        va='bottom'
    )

# y = x reference line
vals = cat_table_plot[['mean_dwell_first', 'mean_dwell_second']].to_numpy()
finite = np.isfinite(vals).all(axis=1)
if finite.any():
    max_val = float(np.nanmax(vals[finite]))
    plt.plot([0, max_val], [0, max_val], 'r--', label='y = x')

plt.xlabel('Mean dwell time per category (First entry, s)')
plt.ylabel('Mean dwell time per category (Second entry, s)')
plt.title('Category-wise Mean Dwell Time: First vs Second Entry')
plt.legend(loc='best', fontsize=8)
plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------- Build object-level table for scatter ----------------

# Use mean dwell per window; change to 'median_dwell_per_window' if you prefer
first_mean = obj_first['mean_dwell_per_window'].rename('mean_dwell_first')
second_mean = obj_second['mean_dwell_per_window'].rename('mean_dwell_second')

# Combine into one DataFrame, indexed by object name
obj_scatter = pd.concat([first_mean, second_mean], axis=1)

# Keep only objects that have dwell data in BOTH visits
obj_scatter = obj_scatter.dropna(subset=['mean_dwell_first', 'mean_dwell_second'])

# Add category information from your existing name_to_cat mapping
obj_scatter['category'] = obj_scatter.index.map(name_to_cat)

# Move index into a column for easier plotting
obj_scatter = obj_scatter.reset_index().rename(columns={'names': 'object'})

print("\nObject-level dwell table used for scatter plot (first rows):")
print(obj_scatter.head(20).to_string(index=False))

# ---------------- Scatter plot: dwell per object (first vs second) ----------------

sns.set(style='whitegrid')
plt.figure(figsize=(7, 7))

# If some objects have no category, they'll appear as NaN in 'category' and get their own colour
sns.scatterplot(
    data=obj_scatter,
    x='mean_dwell_first',
    y='mean_dwell_second',
    hue='category',      # remove 'hue' if you don't want colouring by category
    s=60,
    alpha=0.9
)

# Add y=x reference line
vals = obj_scatter[['mean_dwell_first', 'mean_dwell_second']].to_numpy()
finite = np.isfinite(vals).all(axis=1)
if finite.any():
    max_val = float(np.nanmax(vals[finite]))
    plt.plot([0, max_val], [0, max_val], 'r--', label='y = x')

plt.xlabel('Mean dwell time per object (First entry, s)')
plt.ylabel('Mean dwell time per object (Second entry, s)')
plt.title('Object-wise Mean Dwell Time: First vs Second Entry')
plt.legend(loc='best', fontsize=8)
plt.tight_layout()
plt.show()