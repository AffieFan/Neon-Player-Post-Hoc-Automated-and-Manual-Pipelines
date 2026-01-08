import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import numpy as np
import os
from markovchain import MarkovChain

#Change these if annotation labels change
AOI_TYPES = {
    'road': {'start': 'road start', 'end': 'road end'},
    'dashboard': {'start': 'dashboard start', 'end':'dashboard end'},
    'mirror': {'start':'mirror start', 'end':'mirror end'},
    'touchscreen': {'start':'touchscreen start', 'end':'touchscreen end'},
    'physical controls': {'start':'physical control start', 'end':'physical control end'}
}

#Load annotation csv
def load_csv(script_dir):
    return pd.read_csv(os.path.join(script_dir, 'annotations.csv'))

#Create glances from annotated start/end markers
def create_glances(annotations):

    if len(annotations) < 2:
        return pd.DataFrame(columns = ['AOI', 'start', 'end', 'duration'])
    
    open_glances = {}
    glances = []

    for idx, row in annotations.iterrows():
        label = row['label'].strip().lower()

        #Find start marker
        for aoi, markers in AOI_TYPES.items():
            if label == markers['start']:
                if aoi in open_glances:
                    continue
                open_glances[aoi] = {
                    'start': row['timestamp [ns]'],
                    'start_index': idx
                }
                break
            elif label == markers['end']:
                if aoi not in open_glances:
                    continue

                duration = (row['timestamp [ns]'] - open_glances[aoi]['start']) / 1e6

                glances.append({
                    'AOI': aoi,
                    'start': open_glances[aoi]['start'],
                    'end': row['timestamp [ns]'],
                    'duration': duration, 
                    'start_index': open_glances[aoi]['start_index'],
                    'end_index':idx
                })
                
                del open_glances[aoi]
                break
    
    # Create DataFrame 
    glance_df = pd.DataFrame(glances)
    if not glance_df.empty:
        glance_df = glance_df.sort_values('start').reset_index(drop=True)
        glance_df = glance_df[['AOI', 'start', 'end', 'duration']]  # Clean columns
    return glance_df

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

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    annotations = load_csv(script_dir)
    glance_df = create_glances(annotations)
    
    # Print results
    print("Found {} complete glances:".format(len(glance_df)))
    print(glance_df)
    
    visualize_glances(glance_df) 

    # Summary stats
    print("\nNumber of glances per AOI:")
    print(count_glances_per_aoi(glance_df))

    print("\nAverage duration per AOI:")
    print(average_duration_per_aoi(glance_df))

    plot_markov_chain(transition_matrix(glance_df))


    #print(glance_df.groupby('AOI')['duration'].describe())

    # Optional: Save to CSV
    # glance_df.to_csv(os.path.join(script_dir, 'glances.csv'), index=False)

if __name__ == '__main__':
    main()