import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------- Config -----------------------
data_path = 'C:/Users/ameer/Downloads/combined_dataframe.csv'   # <-- replace with your file path
window_seconds = 10
mad_z_thresh = 3.5           # outlier threshold for fixation durations (MAD z)
top_n = None                 # e.g., 30 to limit number of objects in the plot; None = show all
# ------------------------------------------------------

# Load data
df = pd.read_csv(data_path)

# Choose time column
time_col = 'total_time' if 'total_time' in df.columns else 'Continuous_Time'
if time_col not in df.columns:
    raise ValueError("No usable time column found. Expected 'total_time' or 'Continuous_Time'.")

# Resolve Collider_Categorical column (case-insensitive)
cols_lower = {c.lower(): c for c in df.columns}
if 'collider_categorical' in cols_lower:
    object_col = cols_lower['collider_categorical']
elif 'collider_categoricaln' in cols_lower:
    object_col = cols_lower['collider_categoricaln']
else:
    raise ValueError("No 'Collider_Categorical' (or 'Collider_CategoricalN') column found.")

# Required columns
required_cols = ['SubjectID', time_col, 'events', 'length', object_col,
                 'entry_nr', 'street_id_within_participant', 'isNewTurn']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Sort for time-consistent windowing
df = df.sort_values(['SubjectID', time_col]).reset_index(drop=True)

# Keep only fixation events (events==2); length is fixation duration
gaze = df[df['events'] == 2].copy()
if len(gaze) == 0:
    raise ValueError("No fixation rows found (events == 2).")

# Remove outlier fixations via MAD (robust, like your example)
mad = np.nanmedian(np.abs(gaze['length'] - np.nanmedian(gaze['length']))) * 1.4826
scale = mad if mad > 0 else 1.0
gaze['gaze_mad_z'] = np.abs(gaze['length'] - np.nanmedian(gaze['length'])) / scale
gaze_no_out = gaze[gaze['gaze_mad_z'] <= mad_z_thresh].copy()

def collect_avgfix_windows(entry_nr_target: int) -> pd.DataFrame:
    """
    For entry_nr_target (1 or 2), build 10s windows after the entry anchor and
    return per-object average fixation durations for each window:
      AvgFixationTime = sum(length)/count(length) for that object within the window.
    Entry selection:
      - First visits:  (entry_nr==1) & (isNewTurn==True)
      - Second visits: (entry_nr==2) only
    Output columns: SubjectID, street_id, Entry, AnchorTime, Collider, AvgFixationTime
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
                entry_label = 'First'
            else:
                anchors = df_ss[(df_ss['entry_nr'] == 2)]
                entry_label = 'Second'

            if anchors.empty:
                continue

            anchor_time = anchors[time_col].min()

            in_win = (gz[time_col] >= anchor_time) & (gz[time_col] <= anchor_time + window_seconds)
            win = gz.loc[in_win, [object_col, 'length']]
            if win.empty:
                continue

            per_obj = (win.groupby(object_col, as_index=False)
                         .agg(fixation_count=('length', 'count'),
                              total_dwell=('length', 'sum')))
            per_obj['AvgFixationTime'] = per_obj['total_dwell'] / per_obj['fixation_count']

            per_obj['SubjectID'] = subj
            per_obj['street_id'] = street_id
            per_obj['Entry'] = entry_label
            per_obj['AnchorTime'] = anchor_time
            per_obj.rename(columns={object_col: 'Collider'}, inplace=True)

            rows.append(per_obj[['SubjectID','street_id','Entry','AnchorTime','Collider','AvgFixationTime']])

    if not rows:
        return pd.DataFrame(columns=['SubjectID','street_id','Entry','AnchorTime','Collider','AvgFixationTime'])
    return pd.concat(rows, ignore_index=True)

# Collect per-window per-object average fixation durations
avgfix_first  = collect_avgfix_windows(entry_nr_target=1)
avgfix_second = collect_avgfix_windows(entry_nr_target=2)
avgfix_all = pd.concat([avgfix_first, avgfix_second], ignore_index=True)

print(f"Windows (first):  {avgfix_first[['SubjectID','street_id','AnchorTime']].drop_duplicates().shape[0]}")
print(f"Windows (second): {avgfix_second[['SubjectID','street_id','AnchorTime']].drop_duplicates().shape[0]}")
print(f"Rows (per-object avg fixation): first={len(avgfix_first)}, second={len(avgfix_second)}")

# Order colliders by overall median AvgFixationTime for nicer plotting
order = (avgfix_all.groupby('Collider')['AvgFixationTime']
                 .median()
                 .sort_values(ascending=False)
                 .index.tolist())

# Optionally limit to top_n colliders for readability
if top_n is not None and len(order) > top_n:
    keep = set(order[:top_n])
    plot_df = avgfix_all[avgfix_all['Collider'].isin(keep)].copy()
    order = order[:top_n]
else:
    plot_df = avgfix_all.copy()

# Plot: box-and-whisker (horizontal), hue = Entry (First vs Second)
sns.set(style="whitegrid")
g = sns.catplot(
    data=plot_df,
    y='Collider', x='AvgFixationTime',
    order=order,
    hue='Entry', kind='box',
    height=6, aspect=2, palette={'First': '#1f77b4', 'Second': '#ff7f0e'}
)
g.set(title="Average fixation duration per collider (10s after turn): First vs Second visits",
      xlabel="Average fixation duration (s)", ylabel="Collider")

# Overlay jittered points (optional, like your stripplot)
ax = sns.stripplot(
    data=plot_df, y='Collider', x='AvgFixationTime',
    order=order, hue='Entry', dodge=True, alpha=0.25,
    linewidth=0.5, edgecolor='gray', palette={'First': '#1f77b4', 'Second': '#ff7f0e'}
)
# Remove duplicate legend produced by stripplot (keep the one from catplot)
handles, labels = ax.get_legend_handles_labels()
if len(labels) > 0:
    ax.legend(handles[:2], labels[:2], title='Entry', loc='best')

plt.tight_layout()
plt.show()

# Simple summaries per entry (across per-object per-window values)
summary = (avgfix_all
           .groupby('Entry')['AvgFixationTime']
           .agg(['count','mean','median','std'])
           .round(3))
print("\nAverage fixation duration summary per entry:")
print(summary)