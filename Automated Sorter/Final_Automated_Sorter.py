import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import numpy as np
import os
from markovchain import MarkovChain

# Get current script directory
def load_csv(script_dir):
# Load CSVs using full path
    fixations = pd.read_csv(os.path.join(script_dir, 'fixations.csv'))
    saccades = pd.read_csv(os.path.join(script_dir, 'saccades.csv'))
    blinks = pd.read_csv(os.path.join(script_dir, 'blinks.csv'))
    annotations = pd.read_csv(os.path.join(script_dir, 'annotations.csv'))
    return fixations, saccades, blinks, annotations

def rename(blinks, saccades, fixations, annotations):
# Rename columns for convenience
    blinks['start'] = blinks['start timestamp [ns]']
    blinks['end'] = blinks['end timestamp [ns]']
    blinks['duration'] = blinks['duration [ms]']

    saccades['start'] = saccades['start timestamp [ns]']
    saccades['end'] = saccades['end timestamp [ns]']
    saccades['duration'] = saccades['duration [ms]']

    fixations['start'] = fixations['start timestamp [ns]']
    fixations['end'] = fixations['end timestamp [ns]']
    fixations['duration'] = fixations['duration [ms]']

    # Attach annotation data and match AOI to fixation
    fixations2 = fixations.copy()
    fixations2['AOI'] = annotations['label']

    return fixations2, saccades, blinks

def create_glances(fixations2, saccades):
    glances = []
    i = 0

    while i < len(saccades) - 1:
        aoi = fixations2.loc[i + 1, 'AOI']
        start_time = int(saccades.loc[i, 'start'])
        end_time = fixations2.loc[i + 1, 'end']
        #total_duration = saccades.loc[i, 'duration'] + fixations2.loc[i + 1, 'duration']

        j = i + 1

        # merge consecutive glances with the same AOI
        while j < len(saccades) - 1 and fixations2.loc[j + 1, 'AOI'] == aoi:
            end_time = fixations2.loc[j + 1, 'end']
            #duration calculated from added durations from csv files
            #total_duration += saccades.loc[j, 'duration'] + fixations2.loc[j + 1, 'duration']

            j += 1

        glances.append({
            'AOI': aoi,
            'start': start_time,
            'end': end_time,
            #'duration': total_duration,
        })

        i = j

    glance_df = pd.DataFrame(glances)
    return glance_df

def calc_actual_duration(glance_df):
    #discrepency with duration found in csv file and calculated duration from start and end times
    #this function calculates the accurate duration from start and end times
    glance_df['actual duration'] = (glance_df['end'] - glance_df['start']) 
    return glance_df

def apply_blink_rules(glance_df, blinks):
    if len(glance_df) == 0 or len(blinks) == 0:
        return glance_df
    
    adjusted_glances = []

    for _, glance in glance_df.iterrows():
        g_start = glance['start']
        g_end = glance['end']
        g_aoi = glance['AOI']
        skip_remaining = False
        blink_count = 0

        relevant_blinks = blinks[(blinks['start'] < g_end) & (blinks['end'] > g_start)]
        if relevant_blinks.empty: 
            adjusted_glances.append({
                'AOI': g_aoi,
                'start': int(g_start), 
                'end': g_end,
                'duration': (g_end - g_start),
                'num_blinks': blink_count
            })
        else:
            for _,blink in relevant_blinks.iterrows():
                b_start, b_end, b_duration = blink['start'], blink['end'], blink['duration']                

                #Case 1
                if b_start > g_start and b_end < g_end and b_duration >= 500:
                    adjusted_glances.append({
                        'AOI': g_aoi,
                        'start': int(g_start), 
                        'end': b_start,
                        'duration': (b_start - g_start),
                        'num_blinks': blink_count
                    })
                    blink_count = 0
                    g_start = b_end
                    continue
                
                #Case 2
                elif b_start > g_start and b_end < g_end and b_duration < 500:
                    blink_count += 1
                    continue

                #Case 3
                elif b_start > g_start and b_end > g_end:
                    adjusted_glances.append({
                            'AOI': g_aoi,
                            'start': int(g_start), 
                            'end': b_start,
                            'duration': (b_start - g_start), 
                            'num_blinks': blink_count
                    })
                    skip_remaining = True
                    break 
                #case 4
                elif b_start < g_start and b_end < g_end:
                    g_start = b_end
                    blink_count = 0
                    continue

            if not skip_remaining and g_start < g_end:
                adjusted_glances.append({
                'AOI': g_aoi,
                'start': int(g_start),
                'end': g_end,
                'duration': (g_end - g_start), 
                'num_blinks': blink_count
            })
               
    return pd.DataFrame(adjusted_glances)

def count_saccades_per_glance(glance_df, saccades):
    glance_df = glance_df.copy()
    saccade_counts = []

    for _, glance in glance_df.iterrows():
        g_start = glance['start']
        g_end = glance['end']
        count = len(saccades[(saccades['start'] >= g_start) & (saccades['end'] <= g_end)])
        saccade_counts.append(count)

    glance_df['num_saccades'] = saccade_counts
    return glance_df

def count_glances_per_aoi(glance_df):
    return glance_df['AOI'].value_counts().reset_index().rename(columns={'index': 'AOI', 'AOI': 'num_glances'})

def average_duration_per_aoi(glance_df):
    return glance_df.groupby('AOI')['duration'].mean().reset_index().rename(columns={'duration': 'avg_duration'})

def transition_matrix(glance_df):
    # Get the sequence of AOIs
    aois = glance_df['AOI'].dropna().tolist()

    # Get unique AOIs in sorted order (for consistent matrix labels)
    unique_aois = sorted(set(aois))
    aoi_to_index = {aoi: i for i, aoi in enumerate(unique_aois)}

    # Initialize transition count matrix
    n = len(unique_aois)
    count_matrix = np.zeros((n, n))

    # Count transitions between AOIs
    for current, nxt in zip(aois[:-1], aois[1:]):
        i = aoi_to_index[current]
        j = aoi_to_index[nxt]
        count_matrix[i][j] += 1

    # Normalize to get probabilities (Markov transition matrix)
    row_sums = count_matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        prob_matrix = np.nan_to_num(count_matrix / row_sums)

    return prob_matrix, unique_aois
     
def plot_markov_chain(matrix_tuple):
    matrix, labels = matrix_tuple
    mc = MarkovChain(matrix, labels)
    mc.draw("/Users/affiefan/Desktop/Hfast/Neon CodeTester/2/img/markov-chain.png") 

def visualize_glances(glance_df):
    """Plot a timeline of glances by AOI"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 4))
    for _, row in glance_df.iterrows():
        plt.plot([row['start'], row['end']], [row['AOI'], row['AOI']], 
                'o-',  # Marker and line style
                linewidth=5,  # Thicker lines
                markersize=8,  # Larger dots
                label=row['AOI'])
    
    plt.xlabel('Time (ns)', fontsize=12)
    plt.ylabel('AOI', fontsize=12)
    plt.title('Glance Timeline', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)  # Add grid lines
    plt.tight_layout()  # Prevent label cutoff
    plt.show()
            
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fixations, saccades, blinks, annotations = load_csv(script_dir)
    fixations2, saccades, blinks = rename(blinks, saccades, fixations, annotations)

    glance_df = create_glances(fixations2, saccades)
    #print (glance_df)

    glance_df = calc_actual_duration(glance_df)
    #print (glance_df)
    #print (blinks)

    glance_df = apply_blink_rules(glance_df, blinks)
    #print(glance_df)

    glance_df = count_saccades_per_glance(glance_df, saccades)
    print(glance_df)

    # Summary stats
    print("\nNumber of glances per AOI:")
    print(count_glances_per_aoi(glance_df))

    print("\nAverage duration per AOI:")
    print(average_duration_per_aoi(glance_df))

    plot_markov_chain(transition_matrix(glance_df))

    visualize_glances(glance_df) 

if __name__ == '__main__':
    main()
