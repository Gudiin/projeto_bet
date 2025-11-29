import pandas as pd

def calculate_rolling_stats(df, window=5):
    # Ensure data is sorted by date
    df = df.sort_values('start_timestamp')
    
    # Create separate DataFrames for Home and Away stats per team
    # We need to restructure the data so each row is a team-match
    
    matches_home = df[['match_id', 'start_timestamp', 'home_team_id', 'corners_home_ft', 'shots_ot_home_ft', 'home_score']].copy()
    matches_home.columns = ['match_id', 'timestamp', 'team_id', 'corners', 'shots', 'goals']
    matches_home['is_home'] = 1
    
    matches_away = df[['match_id', 'start_timestamp', 'away_team_id', 'corners_away_ft', 'shots_ot_away_ft', 'away_score']].copy()
    matches_away.columns = ['match_id', 'timestamp', 'team_id', 'corners', 'shots', 'goals']
    matches_away['is_home'] = 0
    
    team_stats = pd.concat([matches_home, matches_away]).sort_values(['team_id', 'timestamp'])
    
    # Calculate rolling averages
    team_stats['avg_corners_last_5'] = team_stats.groupby('team_id')['corners'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    team_stats['avg_shots_last_5'] = team_stats.groupby('team_id')['shots'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    team_stats['avg_goals_last_5'] = team_stats.groupby('team_id')['goals'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    
    # Merge back to original match rows
    # We need to join twice: once for home team stats, once for away team stats
    
    df_features = df.copy()
    
    # Join Home Stats
    home_stats = team_stats[team_stats['is_home'] == 1][['match_id', 'avg_corners_last_5', 'avg_shots_last_5', 'avg_goals_last_5']]
    home_stats.columns = ['match_id', 'home_avg_corners', 'home_avg_shots', 'home_avg_goals']
    df_features = df_features.merge(home_stats, on='match_id', how='left')
    
    # Join Away Stats
    away_stats = team_stats[team_stats['is_home'] == 0][['match_id', 'avg_corners_last_5', 'avg_shots_last_5', 'avg_goals_last_5']]
    away_stats.columns = ['match_id', 'away_avg_corners', 'away_avg_shots', 'away_avg_goals']
    df_features = df_features.merge(away_stats, on='match_id', how='left')
    
    # Drop rows with NaN (first few games of season)
    df_features = df_features.dropna()
    
    return df_features

def prepare_training_data(df):
    df_processed = calculate_rolling_stats(df)
    
    # Features (X)
    features = [
        'home_avg_corners', 'home_avg_shots', 'home_avg_goals',
        'away_avg_corners', 'away_avg_shots', 'away_avg_goals'
    ]
    
    X = df_processed[features]
    
    # Targets (y) - Example: Total Corners
    y_corners = df_processed['corners_home_ft'] + df_processed['corners_away_ft']
    
    return X, y_corners, df_processed
