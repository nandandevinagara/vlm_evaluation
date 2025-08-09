import pandas as pd
import sys


# python accuracy_finder.py abc.csv

def calculate_accuracies(csv_file):
    # Read CSV file (adjust delimiter if needed)
    df = pd.read_csv(csv_file, delimiter=';')

    # Clean column names (strip spaces, remove BOM)
    df.columns = df.columns.str.strip()

    # Calculate accuracies
    df['top-1'] = df['top-1'].astype(str).str.strip().map({'True': True, 'False': False})
    df['top-3'] = df['top-3'].astype(str).str.strip().map({'True': True, 'False': False})

    top1_acc = (df['top-1'].sum() / len(df)) * 100
    top3_acc = (df['top-3'].sum() / len(df)) * 100

    # Print results
    print("Results:")
    print(csv_file)
    print(f"top1 accuracy - {top1_acc:.2f}%")
    print(f"top3 accuracy - {top3_acc:.2f}%")

# Example usage
calculate_accuracies(sys.argv[1])
