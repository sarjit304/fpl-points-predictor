import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Input dataset path
INPUT_PATH = "data/raw/combined_all_seasons.csv"

# Output paths
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

# Load dataset
df = pd.read_csv(INPUT_PATH)

# Split 70% train / 30% test
train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    shuffle=True
)

# Create output directory if needed
os.makedirs("data", exist_ok=True)

# Save files
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

print("Train/Test split completed:")
print(f"Train size: {len(train_df)}")
print(f"Test size: {len(test_df)}")
