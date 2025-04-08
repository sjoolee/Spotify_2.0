import pandas as pd

# Define genre categories
genre_categories = ["anime", "pop", "indie", "lo-fi", "rock", "musicals"]

def simplify_genre(genre_text):
    """Simplifies genre based on predefined categories:
       - If multiple genres are found & "anime" is included -> return "anime"
       - If multiple genres exist (without anime) -> return "pop"
       - If only one genre is found -> return it
       - If no match -> return "other"
    """
    if pd.isna(genre_text):
        return "other"
    
    genre_text = genre_text.lower()
    matched_genres = [genre for genre in genre_categories if genre in genre_text]

    if len(matched_genres) > 1:
        return "anime" if "anime" in matched_genres else "pop"
    elif len(matched_genres) == 1:
        return matched_genres[0]
    else:
        return "other"

# Load the original dataset
file = "cleaned2020-2022.csv"
df_original = pd.read_csv(file)  # Keep original dataset unchanged

# Create a copy to update
df_updated = df_original.copy()

# Apply the function to the 'Genres' column
df_updated["simplified_genre"] = df_updated["Genres"].apply(simplify_genre)

# Save the updated dataset as a separate file
df_updated.to_csv("updated_simplified_genres.csv", index=False)

# Display sample output
print(df_updated[["Genres", "simplified_genre"]].head())
