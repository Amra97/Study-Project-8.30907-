import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your data
data_path = 'C:/Users/ameer/Downloads/1031_1_wTurns.csv'
df = pd.read_csv(data_path)

# 2. If needed, calculate cumulative time
df['total_time'] = df['Time_Shift'].cumsum()

# 3. Filter for second entry turns (entry_nr == 2)
turn_rows = df[(df['entry_nr'] == 2)]

# 4. For each turn, sum fixation durations per object in the window, then average across windows
total_per_window = []

for t in turn_rows.index:
    turn_time = df.at[t, 'total_time']
    # Find all rows in 10s window after this turn
    mask = (df['total_time'] >= turn_time) & (df['total_time'] <= turn_time + 10)
    subset = df.loc[mask & df['isFix'].notnull() & (df['isFix'] > 0), ['Collider_Categorical', 'isFix']]
    # Sum durations per object in this window
    win_totals = subset.groupby('Collider_Categorical')['isFix'].sum().reset_index()
    win_totals['turn_index'] = t
    total_per_window.append(win_totals)

if total_per_window:
    window_df = pd.concat(total_per_window)
    # 5. Get the average (mean) total fixation time per object per window
    avg_per_window = window_df.groupby('Collider_Categorical')['isFix'].mean().sort_values(ascending=False)
    print("Average TOTAL fixation time per object PER 10s window (second entry only):")
    print(avg_per_window)

    # 6. Bar plot
    plt.figure(figsize=(12, 6))
    avg_per_window.plot(kind='bar', color='dodgerblue')
    plt.title('Average Total Fixation Time per Object\n(Per 10s Window After SECOND Entry)')
    plt.xlabel('Object (Collider_Categorical)')
    plt.ylabel('Mean Total Fixation Time per Window (s)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("No fixations found in the 10s windows after second entries!")