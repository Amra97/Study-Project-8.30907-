import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------- Config -----------------------
data_path = 'C:/Users/ameer/Downloads/combined_dataframe.csv'   # <-- replace with your file path
window_seconds = 10
top_n = 12                   # how many objects to show per plot (None = all)
mad_z_thresh = 3.5           # outlier threshold for fixation durations (MAD on fixation lengths)
bw_adjust = 1.0              # KDE smoothing; higher = smoother
x_max = 5.0                  # FORCE x-axis to 0..5 seconds
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

# Sort for time-consistent windowing
df = df.sort_values(['SubjectID', time_col]).reset_index(drop=True)

# Keep only fixation events (events==2) and remove outliers via MAD (on fixation durations)
gaze = df[df['events'] == 2].copy()
if len(gaze) == 0:
    raise ValueError("No fixation rows found (events == 2).")

mad = np.nanmedian(np.abs(gaze['length'] - np.nanmedian(gaze['length']))) * 1.4826
scale = mad if mad > 0 else 1.0
gaze['gaze_mad_z'] = np.abs(gaze['length'] - np.nanmedian(gaze['length'])) / scale
gaze_no_out = gaze[gaze['gaze_mad_z'] <= mad_z_thresh].copy()

def per_window_object_stats(entry_nr_target: int) -> pd.DataFrame:
    """
    Build 10s windows after each entry anchor and return per-window per-object dwell stats:
      - total_dwell (sum of fixation lengths in the window)
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
    Aggregates across all windows per object:
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
                appearances=('names', 'count'))
           .sort_values('median_dwell_per_window', ascending=False))
    return agg

obj_first = per_object_aggregates(win_first)
obj_second = per_object_aggregates(win_second)

print("\nPer-object (first entry) — top 10 by median dwell:")
print(obj_first.head(10))
print("\nPer-object (second entry) — top 10 by median dwell:")
print(obj_second.head(10))

# ----------------------- Density plots -----------------------
sns.set(style='whitegrid')

def plot_dwell_density(win_df: pd.DataFrame,
                       obj_agg: pd.DataFrame,
                       title: str,
                       top_n: int | None = 12,
                       top_by: str = 'median_dwell_per_window',
                       bw_adjust: float = 1.0,
                       xlim: tuple[float, float] = (0.0, 5.0)):
    """
    Plot KDE (density) of per-window dwell times (total_dwell) for top_n objects.
    xlim sets the exact x-axis and also clips the KDE to that range.
    """
    if win_df.empty:
        print("No data to plot.")
        return

    # Pick objects
    if obj_agg.empty:
        objects = win_df['names'].value_counts().index.tolist()
    else:
        objects = obj_agg.sort_values(top_by, ascending=False).index.tolist()
        if top_n is not None:
            objects = objects[:top_n]

    data = win_df[win_df['names'].isin(objects)].copy()
    data = data.dropna(subset=['total_dwell'])
    if data.empty:
        print("No dwell data for selected objects.")
        return

    palette = sns.color_palette('tab20', n_colors=len(objects))
    plt.figure(figsize=(10, 6))
    for color, name in zip(palette, objects):
        s = data.loc[data['names'] == name, 'total_dwell']
        if len(s) >= 2:
            sns.kdeplot(s, bw_adjust=bw_adjust, label=name, color=color, linewidth=2,
                        clip=xlim, cut=0)  # cut=0 stops tails beyond clip
        else:
            # Single observation fallback
            val = float(s.iloc[0])
            if xlim[0] <= val <= xlim[1]:
                plt.axvline(val, color=color, linestyle='--', alpha=0.6, label=f"{name} (single)")

    plt.xlim(xlim)
    plt.xlabel('Per-window dwell time (s)')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend(loc='upper right', ncol=1, fontsize=8, frameon=True)
    plt.tight_layout()
    plt.show()

# Force the same x-axis (0..5 s) for both plots
shared_xlim = (0.0, float(x_max))

# Density plot for FIRST visit
plot_dwell_density(
    win_first,
    obj_first,
    title='Density of Per-window Dwell Times by Object (First Entry, 10s after turn)',
    top_n=top_n,
    top_by='median_dwell_per_window',
    bw_adjust=bw_adjust,
    xlim=shared_xlim
)

# Density plot for SECOND visit
plot_dwell_density(
    win_second,
    obj_second,
    title='Density of Per-window Dwell Times by Object (Second Entry, 10s after turn)',
    top_n=top_n,
    top_by='median_dwell_per_window',
    bw_adjust=bw_adjust,
    xlim=shared_xlim
)