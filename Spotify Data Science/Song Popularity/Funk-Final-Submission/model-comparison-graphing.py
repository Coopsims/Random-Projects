from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


def main():
    # Define the file paths
    file_paths = [
        './Model-Results/resultsof10.csv',
        './Model-Results/resultsof20.csv',
        './Model-Results/resultsof30.csv',
        './Model-Results/resultsof40.csv',
        './Model-Results/resultsof50.csv',
        './Model-Results/resultsof60.csv',
        './Model-Results/resultsof70.csv',
        './Model-Results/resultsof80.csv'
    ]

    # Initialize an empty list to hold dataframes
    dfs = []

    palette = {
        'KNN-Regressor': 'skyblue',
        'KNN-Classifier': 'orange',
        'Neural': 'red',
        'Decision': 'green',
        'Random': 'purple',
        'SVM': 'brown'
    }
        # Loop through file paths, read each file, add an identifier, and append to the list
    for file_path in file_paths:
        # Extract the identifier (e.g., "10", "20") from the file path
        identifier = file_path.split('resultsof')[1].split('.')[0]

        # Read the file
        df = pd.read_csv(file_path)

        # Add an identifier column to distinguish between different datasets
        df['dataset_size'] = identifier

        # Append the dataframe to the list
        dfs.append(df)

    # Combine all dataframes into a single dataframe
    df_combined = pd.concat(dfs, ignore_index=True)
    # Filter the DataFrame to include only testing results
    df_testing = df_combined[df_combined['algorithm'].str.contains('Testing')]

    df_training = df_combined[df_combined['algorithm'].str.contains('Training')]

    df_testing['model_source'] = df_testing['algorithm'].apply(
        lambda x: 'dataset.csv' if 'dataset.csv' in x else 'tracks.csv')
    # Extract the base model name from the 'algorithm' column (ignoring "Training" or "Testing")
    df_testing['base_model'] = df_testing['algorithm'].apply(lambda x: x.split(' ')[1])

    # Calculate average F1 scores for each base model and model source combination
    average_f1_by_model_source = df_testing.groupby(['base_model', 'model_source'])['f1_score'].mean().unstack()

    # Plotting the double bar chart for each model
    average_f1_by_model_source.plot(kind='bar')
    plt.title('Average F1 Score by Model and Source',fontweight='bold')
    plt.xlabel('Model',fontweight='bold')
    plt.ylabel('Average F1 Score',fontweight='bold')
    plt.xticks(rotation=45)
    plt.legend(title='Model Source')
    plt.grid(axis='y')

    # Save plot
    plt.savefig('./generated-images/Average-f1-score.png', dpi=300, bbox_inches='tight')
    plt.clf()

# Ensure the 'model_name' column is added to the df_testing DataFrame correctly
    df_testing['model_name'] = df_testing['algorithm'].apply(lambda x: x.split(' ')[1])

    # Update the subsets for dataset.csv and tracks.csv with the new df_testing
    df_dataset_csv = df_testing[df_testing['model_source'] == 'dataset.csv']
    df_tracks_csv = df_testing[df_testing['model_source'] == 'tracks.csv']

    # Plot for models trained on dataset.csv
    sns.lineplot(data=df_dataset_csv[df_dataset_csv['model_name'] == 'KNN-Regressor'],
                 x='dataset_size', y='f1_score', hue='model_name',
                 marker='o', palette={'KNN-Regressor': 'lightblue'},
                 linewidth=4, markersize=10)

    sns.lineplot(data=df_dataset_csv[df_dataset_csv['model_name'] != 'KNN-Regressor'],
                 x='dataset_size', y='f1_score', hue='model_name',
                 marker='o', palette=palette, linewidth=1.5, markersize=6)
    plt.title('F1 Score vs. Popularity Cutoff (dataset.csv)',fontweight='bold')
    plt.xlabel('Popularity Cutoff',fontweight='bold')
    plt.ylabel('F1 Score',fontweight='bold')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.savefig('./generated-images/f1-score-versus-popularity-600kdataset.png', dpi=300, bbox_inches='tight')
    plt.clf()
    # Plot for models trained on tracks.csv
    sns.lineplot(data=df_tracks_csv, x='dataset_size', y='f1_score', hue='model_name', marker='o', palette='tab10')
    plt.title('F1 Score vs. Popularity Cutoff (tracks.csv)',fontweight='bold')
    plt.xlabel('Minimum Popularity Cutoff',fontweight='bold')
    plt.ylabel('F1 Score',fontweight='bold')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.savefig('./generated-images/f1-score-versus-popularity-100kdataset.png', dpi=300, bbox_inches='tight')
    plt.clf()


if __name__ == "__main__":
    main()