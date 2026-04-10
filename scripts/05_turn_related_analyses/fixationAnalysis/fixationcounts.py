import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

data_path = 'C:/Users/ameer/Downloads/combined_dataframe.csv'
df = pd.read_csv(data_path)
df['total_time'] = df['Time_Shift'].cumsum()

def analyze_entry(entry_number, plot_color):
    # Get all turns of this entry_nr
    turn_rows = df[(df['isNewTurn'] == True) & (df['entry_nr'] == entry_number)]
    total_per_window = []
    for t in turn_rows.index:
        turn_time = df.at[t, 'total_time']
        mask = (df['total_time'] >= turn_time) & (df['total_time'] <= turn_time + 10)
        subset = df.loc[mask & df['isFix'].notnull() & (df['isFix'] > 0), ['Collider_Categorical', 'isFix']]
        win_totals = subset.groupby('Collider_Categorical')['isFix'].sum().reset_index()
        win_totals['turn_index'] = t
        total_per_window.append(win_totals)
    if total_per_window:
        window_df = pd.concat(total_per_window)
        # ANOVA/Kruskal-Wallis: Does Collider_Categorical explain variance in isFix?
        groups = [g['isFix'].values for _, g in window_df.groupby('Collider_Categorical')]
        if len(groups) > 1:
            fstat, pval = stats.f_oneway(*groups)
            print(f"\nEntry {entry_number}: ANOVA F={fstat:.2f}, p={pval:.4f}")
            hstat, pkruskal = stats.kruskal(*groups)
            print(f"Entry {entry_number}: Kruskal-Wallis H={hstat:.2f}, p={pkruskal:.4f}")
        else:
            print("Not enough object categories for test.")

        # Show average per object as bar chart
        avg_per_object = window_df.groupby('Collider_Categorical')['isFix'].mean().sort_values(ascending=False)
        print(f"\nEntry {entry_number}: Avg fixation time per object:")
        print(avg_per_object)
        plt.figure(figsize=(10,5))
        avg_per_object.plot(kind='bar', color=plot_color)
        plt.tight_layout()
        plt.title(f'Avg Total Fixation per Object ({["1st","2nd"][entry_number-1]} entries)')
        plt.ylabel('Avg fixation time (s)')
        plt.xlabel('Object')
        plt.show()
    else:
        print(f"No fixations in 10s windows for entry {entry_number}.")

# Run for first entries
analyze_entry(1, plot_color='deepskyblue')
