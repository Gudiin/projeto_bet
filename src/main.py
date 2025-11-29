import sys
import os
import pandas as pd
import re

#
# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.database.db_manager import DBManager
from src.scrapers.sofascore import SofaScoreScraper
from src.ml.feature_engineering import prepare_training_data
from src.ml.model import CornerPredictor
from src.analysis.statistical import StatisticalAnalyzer, Colors

def update_database():
    db = DBManager()
    
    # Check for feedback loop updates first
    print("Verificando resultados de previsões anteriores...")
    db.check_predictions()
    
    scraper = SofaScoreScraper(headless=True) # Set headless=False to debug
    
    try:
        scraper.start()
        
        # 1. Get Tournament/Season IDs
        t_id = scraper.get_tournament_id("Brasileirão")
        if not t_id:
            print("Torneio não encontrado.")
            return
            
        s_id = scraper.get_season_id(t_id, "2025")
        if not s_id:
            print("Temporada não encontrada.")
            return
            
        print(f"ID Torneio: {t_id}, ID Temporada: {s_id}")
        
        # 2. Get Matches
        matches = scraper.get_matches(t_id, s_id)
        print(f"Encontrados {len(matches)} jogos.")
        
        # 3. Process Matches & Stats
        for i, m in enumerate(matches):
            if m['status']['type'] == 'finished':
                print(f"[{i+1}/{len(matches)}] Processando {m['homeTeam']['name']} vs {m['awayTeam']['name']}...")
                
                # Save Match Info
                match_data = {
                    'id': m['id'],
                    'tournament': m['tournament']['name'],
                    'season_id': s_id,
                    'round': m['roundInfo']['round'],
                    'status': 'finished',
                    'timestamp': m['startTimestamp'],
                    'home_id': m['homeTeam']['id'],
                    'home_name': m['homeTeam']['name'],
                    'away_id': m['awayTeam']['id'],
                    'away_name': m['awayTeam']['name'],
                    'home_score': m['homeScore']['display'],
                    'away_score': m['awayScore']['display']
                }
                db.save_match(match_data)
                
                # Get & Save Stats
                stats = scraper.get_match_stats(m['id'])
                db.save_stats(m['id'], stats)
                
    except Exception as e:
        print(f"Erro: {e}")
    finally:
        scraper.stop()
        db.close()

def train_model():
    db = DBManager()
    df = db.get_historical_data()
    db.close()
    
    if df.empty:
        print("Banco de dados vazio. Execute a atualização primeiro.")
        return
        
    print(f"Carregados {len(df)} registros para treino.")
    
    X, y, _ = prepare_training_data(df)
    
    predictor = CornerPredictor()
    predictor.train(X, y)

def analyze_match_url():
    url = input("Cole a URL do jogo do SofaScore: ")
    match_id_search = re.search(r'id:(\d+)', url)
    
    if not match_id_search:
        print("ID do jogo não encontrado na URL.")
        return

    match_id = match_id_search.group(1)
    print(f"Analisando jogo ID: {match_id}...")
    
    scraper = SofaScoreScraper(headless=True)
    try:
        scraper.start()
        
        # Get Match Details
        api_url = f"https://www.sofascore.com/api/v1/event/{match_id}"
        ev_data = scraper._fetch_api(api_url)
        
        if not ev_data or 'event' not in ev_data:
            print("Erro ao buscar dados do jogo.")
            return
            
        ev = ev_data['event']
        home_id = ev['homeTeam']['id']
        away_id = ev['awayTeam']['id']
        match_name = f"{ev['homeTeam']['name']} vs {ev['awayTeam']['name']}"
        print(f"Jogo: {match_name}")
        
        # Save Match Info to DB (for retrieval in Option 4)
        db = DBManager()
        match_data = {
            'id': match_id,
            'tournament': ev.get('tournament', {}).get('name', 'Unknown'),
            'season_id': ev.get('season', {}).get('id', 0),
            'round': ev.get('roundInfo', {}).get('round', 0),
            'status': 'finished', # Assuming finished for analysis context or update later
            'timestamp': ev.get('startTimestamp', 0),
            'home_id': home_id,
            'home_name': ev['homeTeam']['name'],
            'away_id': away_id,
            'away_name': ev['awayTeam']['name'],
            'home_score': ev.get('homeScore', {}).get('display', 0),
            'away_score': ev.get('awayScore', {}).get('display', 0)
        }
        db.save_match(match_data)
        db.close()
        
        # Get Last Games for Home and Away
        print("Coletando histórico recente...")
        
        db = DBManager()
        df = db.get_historical_data()
        db.close()
        
        if df.empty:
            print("Banco de dados vazio. Treine o modelo primeiro para melhores resultados.")
            return

        # Filter for Home Team
        home_games = df[(df['home_team_id'] == home_id) | (df['away_team_id'] == home_id)].tail(5)
        away_games = df[(df['home_team_id'] == away_id) | (df['away_team_id'] == away_id)].tail(5)
        
        if len(home_games) < 3 or len(away_games) < 3:
            print("Dados insuficientes no histórico para análise precisa.")
        
        # Calculate averages for ML
        def get_avg_corners(games, team_id):
            corners = []
            for _, row in games.iterrows():
                if row['home_team_id'] == team_id:
                    corners.append(row['corners_home_ft'])
                else:
                    corners.append(row['corners_away_ft'])
            return sum(corners) / len(corners) if corners else 0

        h_avg_corners = get_avg_corners(home_games, home_id)
        a_avg_corners = get_avg_corners(away_games, away_id)
        
        print(f"Média Escanteios (Últimos 5): Casa {h_avg_corners:.1f} | Fora {a_avg_corners:.1f}")
        
        # Clear old predictions for this match to avoid duplicates
        db = DBManager()
        db.delete_predictions(match_id)
        db.close()
        
        # ML Prediction
        predictor = CornerPredictor()
        ml_prediction = 0
        if predictor.load_model():
            X_new = [[h_avg_corners, 0, 0, a_avg_corners, 0, 0]] 
            pred = predictor.predict(X_new)
            ml_prediction = pred[0]
            print(f"\n🤖 Previsão da IA (Random Forest): {ml_prediction:.2f} Escanteios")
            
            # Save ML Prediction
            db = DBManager()
            db.save_prediction(match_id, 'ML', ml_prediction, f"Over {int(ml_prediction)}", 0.0, verbose=True)
            db.close()
            
        # Statistical Analysis
        analyzer = StatisticalAnalyzer()
        
        def prepare_team_df(games, team_id):
            data = []
            for _, row in games.iterrows():
                is_home = row['home_team_id'] == team_id
                data.append({
                    'corners_ft': row['corners_home_ft'] if is_home else row['corners_away_ft'],
                    'corners_ht': row['corners_home_ht'] if is_home else row['corners_away_ht'],
                    'corners_2t': (row['corners_home_ft'] - row['corners_home_ht']) if is_home else (row['corners_away_ft'] - row['corners_away_ht']),
                    'shots_ht': row['shots_ot_home_ht'] if is_home else row['shots_ot_away_ht']
                })
            return pd.DataFrame(data)

        df_h_stats = prepare_team_df(home_games, home_id)
        df_a_stats = prepare_team_df(away_games, away_id)

        # Run Analysis (Pass ML Prediction for alignment)
        top_picks = analyzer.analyze_match(df_h_stats, df_a_stats, ml_prediction=ml_prediction, match_name=match_name)
        
        # Save Predictions (Feedback Loop)
        db = DBManager()
        
        # 1. Save Top 7 Opportunities
        for pick in top_picks:
            db.save_prediction(
                match_id, 
                'Statistical', 
                0, 
                pick['Seleção'], 
                pick['Prob'],
                odds=pick['Odd'],
                category='Top7',
                market_group=pick['Mercado']
            )
            
        # 2. Save AI Suggestions
        suggestions = analyzer.generate_suggestions(top_picks, ml_prediction=ml_prediction)
        for level, pick in suggestions.items():
            if pick:
                db.save_prediction(
                    match_id,
                    'Statistical',
                    0,
                    pick['Seleção'],
                    pick['Prob'],
                    odds=pick['Odd'],
                    category=f"Suggestion_{level}",
                    market_group=pick['Mercado']
                )
        
        print("✅ Previsões salvas no banco de dados.")
        db.close()

    except Exception as e:
        print(f"Erro na análise: {e}")
    finally:
        scraper.stop()

def retrieve_analysis():
    match_id = input("Digite o ID do jogo: ")
    db = DBManager()
    conn = db.connect()
    
    # Get Match Details
    match_query = "SELECT home_team_name, away_team_name FROM matches WHERE match_id = ?"
    match_info = pd.read_sql_query(match_query, conn, params=(match_id,))
    
    match_name = None
    if not match_info.empty:
        match_name = f"{match_info.iloc[0]['home_team_name']} vs {match_info.iloc[0]['away_team_name']}"
    
    # Fetch ML Prediction
    query_ml = "SELECT predicted_value FROM predictions WHERE match_id = ? AND prediction_type = 'ML'"
    ml_pred = pd.read_sql_query(query_ml, conn, params=(match_id,))
    
    if not ml_pred.empty:
        print(f"\n🤖 Previsão da IA (Random Forest): {ml_pred.iloc[0]['predicted_value']:.2f} Escanteios")

    # Fetch Top 7
    query_top7 = "SELECT market_group, market, probability, odds FROM predictions WHERE match_id = ? AND category = 'Top7' ORDER BY probability DESC"
    top7 = pd.read_sql_query(query_top7, conn, params=(match_id,))
    
    if not top7.empty:
        if match_name:
             print(f"\n⚽ {Colors.BOLD}{match_name}{Colors.RESET}")
        print(f"🏆 {Colors.BOLD}TOP 7 OPORTUNIDADES (RECUPERADO){Colors.RESET}")
        tabela_display = []
        for _, row in top7.iterrows():
            prob = row['probability']
            # Reconstruct Type based on market string (simple heuristic)
            tipo = "OVER" if "Over" in row['market'] else "UNDER"
            cor = Colors.GREEN if tipo == "OVER" else Colors.CYAN
            seta = "▲" if tipo == "OVER" else "▼"
            
            # Use market_group if available, else 'RECUPERADO'
            m_group = row['market_group'] if row['market_group'] else "RECUPERADO"
            
            linha_fmt = f"{cor}{row['market']}{Colors.RESET}"
            prob_fmt = f"{prob * 100:.1f}%"
            odd_fmt = f"{Colors.BOLD}@{row['odds']:.2f}{Colors.RESET}"
            direcao_fmt = f"{cor}{seta} {tipo}{Colors.RESET}"
            
            tabela_display.append([m_group, linha_fmt, prob_fmt, odd_fmt, direcao_fmt])
            
        headers = ["MERCADO", "LINHA", "PROB.", "ODD JUSTA", "TIPO"]
        from tabulate import tabulate
        print(tabulate(tabela_display, headers=headers, tablefmt="fancy_grid", stralign="center"))
    else:
        print("Nenhuma análise Top 7 encontrada para este ID.")

    # Fetch Suggestions
    query_sugg = "SELECT category, market_group, market, probability, odds FROM predictions WHERE match_id = ? AND category LIKE 'Suggestion_%'"
    suggs = pd.read_sql_query(query_sugg, conn, params=(match_id,))
    
    if not suggs.empty:
        print(f"\n🎯 {Colors.BOLD}SUGESTÕES DA IA (RECUPERADO):{Colors.RESET}")
        for _, row in suggs.iterrows():
            level = row['category'].split('_')[1]
            cor_nivel = Colors.GREEN if level == "Easy" else (Colors.YELLOW if level == "Medium" else Colors.RED)
            m_group = row['market_group'] if row['market_group'] else ""
            print(f"{cor_nivel}[{level.upper()}]{Colors.RESET} {m_group} - {row['market']} (@{row['odds']:.2f}) | Prob: {row['probability']*100:.1f}%")
    else:
        print("Nenhuma sugestão da IA encontrada para este ID.")
        
    db.close()

def main():
    while True:
        print("\n--- SISTEMA DE PREVISÃO DE ESCANTEIOS (ML) ---")
        print("1. Atualizar Banco de Dados (Scraping Completo)")
        print("2. Treinar Modelo de IA")
        print("3. Analisar Jogo (URL)")
        print("4. Consultar Análise (ID)")
        print("5. Sair")
        
        choice = input("Escolha uma opção: ")
        
        if choice == '1':
            update_database()
        elif choice == '2':
            train_model()
        elif choice == '3':
            analyze_match_url()
        elif choice == '4':
            retrieve_analysis()
        elif choice == '5':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
