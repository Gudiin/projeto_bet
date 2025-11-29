import sqlite3
import pandas as pd
from datetime import datetime

class DBManager:
    def __init__(self, db_path="data/football_data.db"):
        self.db_path = db_path
        self.conn = None
        self.create_tables()

    def connect(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        return self.conn

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def create_tables(self):
        conn = self.connect()
        cursor = conn.cursor()

        # Tabela de Jogos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                match_id INTEGER PRIMARY KEY,
                tournament_name TEXT,
                season_id INTEGER,
                round INTEGER,
                status TEXT,
                start_timestamp INTEGER,
                home_team_id INTEGER,
                home_team_name TEXT,
                away_team_id INTEGER,
                away_team_name TEXT,
                home_score INTEGER,
                away_score INTEGER
            )
        ''')

        # Tabela de Estatísticas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS match_stats (
                match_id INTEGER PRIMARY KEY,
                
                -- Escanteios
                corners_home_ft INTEGER,
                corners_away_ft INTEGER,
                corners_home_ht INTEGER,
                corners_away_ht INTEGER,
                
                -- Chutes
                shots_ot_home_ft INTEGER,
                shots_ot_away_ft INTEGER,
                shots_ot_home_ht INTEGER,
                shots_ot_away_ht INTEGER,
                
                FOREIGN KEY(match_id) REFERENCES matches(match_id)
            )
        ''')

        # Tabela de Previsões (Feedback Loop)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                prediction_type TEXT, -- 'ML', 'Statistical'
                predicted_value REAL, -- ex: 9.5 escanteios
                market TEXT, -- ex: 'Over 9.5'
                probability REAL,
                odds REAL, -- Odd Justa
                category TEXT, -- 'Top7', 'Easy', 'Medium', 'Hard'
                status TEXT DEFAULT 'PENDING', -- 'PENDING', 'GREEN', 'RED'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(match_id) REFERENCES matches(match_id)
            )
        ''')
        
        # Add columns if they don't exist (Migration for existing DB)
        try:
            cursor.execute("ALTER TABLE predictions ADD COLUMN odds REAL")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE predictions ADD COLUMN category TEXT")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE predictions ADD COLUMN market_group TEXT")
        except:
            pass

        conn.commit()

    def save_prediction(self, match_id, pred_type, value, market, prob, odds=0.0, category=None, market_group=None, verbose=False):
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO predictions (match_id, prediction_type, predicted_value, market, probability, odds, category, market_group)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (match_id, pred_type, value, market, prob, odds, category, market_group))
            conn.commit()
            if verbose:
                print(f"Previsão salva para o jogo {match_id}!")
        except Exception as e:
            print(f"Erro ao salvar previsão: {e}")

    def check_predictions(self):
        # Verifica previsões pendentes
        conn = self.connect()
        cursor = conn.cursor()
        
        # Busca previsões pendentes de jogos que já terminaram
        query = '''
            SELECT p.id, p.match_id, p.market, p.predicted_value, s.corners_home_ft, s.corners_away_ft
            FROM predictions p
            JOIN matches m ON p.match_id = m.match_id
            JOIN match_stats s ON m.match_id = s.match_id
            WHERE p.status = 'PENDING' AND m.status = 'finished'
        '''
        
        pending = pd.read_sql_query(query, conn)
        
        if pending.empty:
            return
            
        print(f"Verificando {len(pending)} previsões pendentes...")
        
        for _, row in pending.iterrows():
            total_corners = row['corners_home_ft'] + row['corners_away_ft']
            status = 'RED'
            
            # Lógica simples para Over/Under
            if 'Over' in row['market']:
                line = float(row['market'].split(' ')[1])
                if total_corners > line:
                    status = 'GREEN'
            elif 'Under' in row['market']:
                line = float(row['market'].split(' ')[1])
                if total_corners < line:
                    status = 'GREEN'
            
            # Atualiza status
            cursor.execute("UPDATE predictions SET status = ? WHERE id = ?", (status, row['id']))
            print(f"Previsão {row['id']} (Jogo {row['match_id']}): {row['market']} vs {total_corners} Cantos -> {status}")
            
            print(f"Previsão {row['id']} (Jogo {row['match_id']}): {row['market']} vs {total_corners} Cantos -> {status}")
            
        conn.commit()

    def delete_predictions(self, match_id):
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM predictions WHERE match_id = ?", (match_id,))
            conn.commit()
            print(f"Previsões antigas removidas para o jogo {match_id}.")
        except Exception as e:
            print(f"Erro ao remover previsões antigas: {e}")

    def save_match(self, match_data):
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO matches (
                    match_id, tournament_name, season_id, round, status, 
                    start_timestamp, home_team_id, home_team_name, 
                    away_team_id, away_team_name, home_score, away_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                match_data['id'], match_data['tournament'], match_data['season_id'],
                match_data.get('round'), match_data['status'], match_data['timestamp'],
                match_data['home_id'], match_data['home_name'],
                match_data['away_id'], match_data['away_name'],
                match_data['home_score'], match_data['away_score']
            ))
            conn.commit()
        except Exception as e:
            print(f"Erro ao salvar jogo {match_data['id']}: {e}")

    def save_stats(self, match_id, stats_data):
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO match_stats (
                    match_id, 
                    corners_home_ft, corners_away_ft, 
                    corners_home_ht, corners_away_ht,
                    shots_ot_home_ft, shots_ot_away_ft,
                    shots_ot_home_ht, shots_ot_away_ht
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                match_id,
                stats_data['corners_home_ft'], stats_data['corners_away_ft'],
                stats_data['corners_home_ht'], stats_data['corners_away_ht'],
                stats_data['shots_ot_home_ft'], stats_data['shots_ot_away_ft'],
                stats_data['shots_ot_home_ht'], stats_data['shots_ot_away_ht']
            ))
            conn.commit()
        except Exception as e:
            print(f"Erro ao salvar stats do jogo {match_id}: {e}")

    def get_historical_data(self):
        conn = self.connect()
        # Avoid selecting match_id twice by specifying columns or using a different join strategy
        # SQLite doesn't support 'SELECT * EXCEPT ...'
        # We will select m.* and specific stats columns
        query = '''
            SELECT 
                m.*, 
                s.corners_home_ft, s.corners_away_ft, 
                s.corners_home_ht, s.corners_away_ht,
                s.shots_ot_home_ft, s.shots_ot_away_ft,
                s.shots_ot_home_ht, s.shots_ot_away_ht
            FROM matches m
            JOIN match_stats s ON m.match_id = s.match_id
            WHERE m.status = 'finished'
            ORDER BY m.start_timestamp ASC
        '''
        return pd.read_sql_query(query, conn)
