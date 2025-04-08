import pandas as pd
from datetime import datetime

'''
def remove_nan_or_empty_rows(df):
    df.replace("", float("nan"), inplace=True)
    
    df_cleaned = df.dropna()
    
    return df_cleaned

def preprocess_data(df):   
    # Convert 'Release Date' to age (years since release)
    def parse_release_date(date):
        try:
            if pd.isna(date):
                return None
            date_str = str(date)
            if len(date_str) == 4:  # If only the year is provided
                return datetime(int(date_str), 1, 1)
            return pd.to_datetime(date, errors='coerce')
        except:
            return None
    
    df['Release Date'] = df['Release Date'].apply(parse_release_date)
    df['Song Age'] = (datetime.now() - df['Release Date']).dt.days / 365.25
    df.drop(columns=['Release Date'], inplace=True)
    
    # Convert 'Key' and 'Time Signature' to categorical codes safely
    df['Key'] = pd.to_numeric(df['Key'], errors='coerce').fillna(0).astype(int)
    df['Time Signature'] = pd.to_numeric(df['Time Signature'], errors='coerce').fillna(0).astype(int)
    
    return df

def process_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    
    df = remove_nan_or_empty_rows(df)
    df = preprocess_data(df)
    

    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")


import pandas as pd
from datetime import datetime


def preprocess_data(df):
    
    # Convert 'Release Date' to age (years since release)
    def parse_release_date(date):
        try:
            if pd.isna(date):
                return None
            date_str = str(date)
            if len(date_str) == 4:  # If only the year is provided
                return datetime(int(date_str), 1, 1)
            return pd.to_datetime(date, errors='coerce')
        except:
            return None
    
    df['Release Date'] = df['Release Date'].apply(parse_release_date)
    df['Song Age'] = (datetime.now() - df['Release Date']).dt.days / 365.25
    df.drop(columns=['Release Date'], inplace=True)
    
    # Assign Genre based on Artist Name
    pop_artists = ['Taylor Swift', 'Bruno Mars', 'Miley Cyrus', 'Sia', 'SZA', 'Sam Smith','Kim Petras','Nova Twins','Alexa Cappelli', 'P!nk', 'MKTO','One Direction','Harry Styles','Neon Trees','FINNEAS']
    rock_artists = ['Artist1', 'Artist2']  # Add more rock artists
    
    df.loc[df['Artist Name(s)'].isin(pop_artists), 'Genres'] = 'pop'
    #df.loc[df['Artist Name'].isin(rock_artists), 'Genres'] = 'rock'
    
    # Convert 'Key' and 'Time Signature' to categorical codes safely
    df['Key'] = pd.to_numeric(df['Key'], errors='coerce').fillna(0).astype(int)
    df['Time Signature'] = pd.to_numeric(df['Time Signature'], errors='coerce').fillna(0).astype(int)
    
    return df

def process_csv(input_file, output_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file)
    
    df = preprocess_data(df)
    
    # Save the processed DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")


# MAIN
input_file = "Combined_Play_Count2023-2024.csv"  
output_file = "genreprocessed2023-2024.csv" 
process_csv(input_file, output_file)

'''


import pandas as pd
from datetime import datetime

def remove_nan_or_empty_rows(df):
    df.replace("", float("nan"), inplace=True)
    
    df_cleaned = df.dropna()
    
    return df_cleaned

def preprocess_data(df):   
    # Convert 'Release Date' to age (years since release)
    def parse_release_date(date):
        try:
            if pd.isna(date):
                return None
            date_str = str(date)
            if len(date_str) == 4:  # If only the year is provided
                return datetime(int(date_str), 1, 1)
            return pd.to_datetime(date, errors='coerce')
        except:
            return None
    
    df['Release Date'] = df['Release Date'].apply(parse_release_date)
    df['Song Age'] = (datetime.now() - df['Release Date']).dt.days / 365.25
    df.drop(columns=['Release Date'], inplace=True)
    df.drop(columns=['Added By'], inplace=True)
    df.drop(columns=['Track ID'], inplace=True)
    df.drop(columns=['Added At'], inplace=True)
    df.drop(columns=['Record Label'], inplace=True)
    
    return df

def process_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    
    df = remove_nan_or_empty_rows(df)
    df = preprocess_data(df)
    

    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")



# MAIN
input_file = "cols2024-2025_Exportify.csv"  
output_file = "cleaned2024-2025.csv" 
process_csv(input_file, output_file)

