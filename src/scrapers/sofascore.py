import time
import random
from playwright.sync_api import sync_playwright

class SofaScoreScraper:
    def __init__(self, headless=True):
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.page = None

    def start(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.page = self.browser.new_page()
        self.page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        })
        # Go to a neutral page to initialize
        self.page.goto("https://www.sofascore.com")

    def stop(self):
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def _fetch_api(self, url):
        time.sleep(random.uniform(0.5, 1.5)) # Rate limiting
        script = f"""
            async () => {{
                try {{
                    const r = await fetch('{url}');
                    if (r.status !== 200) return null;
                    return await r.json();
                }} catch {{ return null; }}
            }}
        """
        return self.page.evaluate(script)

    def get_tournament_id(self, query="Brasileirão"):
        # Search for the tournament to get ID and Season ID
        url = f"https://www.sofascore.com/api/v1/search/{query}"
        print(f"Buscando torneio: {query}...")
        data = self._fetch_api(url)
        
        if data and 'results' in data:
            for item in data['results']:
                if item['type'] == 'uniqueTournament':
                    entity = item['entity']
                    print(f"Encontrado: {entity['name']} (ID: {entity['id']})")
                    if query.lower() in entity['name'].lower() or entity['name'].lower() in query.lower():
                        return entity['id']
        return None

    def get_season_id(self, tournament_id, year="2024"):
        url = f"https://www.sofascore.com/api/v1/unique-tournament/{tournament_id}/seasons"
        data = self._fetch_api(url)
        if data and 'seasons' in data:
            for s in data['seasons']:
                if s['year'] == year:
                    return s['id']
        return None

    def get_matches(self, tournament_id, season_id):
        matches = []
        # Rounds usually go from 1 to 38
        for round_num in range(1, 39):
            print(f"Coletando rodada {round_num}...")
            url = f"https://www.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{season_id}/events/round/{round_num}"
            data = self._fetch_api(url)
            if data and 'events' in data:
                for event in data['events']:
                    matches.append(event)
        return matches

    def get_match_stats(self, match_id):
        url = f"https://www.sofascore.com/api/v1/event/{match_id}/statistics"
        data = self._fetch_api(url)
        
        stats = {
            'corners_home_ft': 0, 'corners_away_ft': 0,
            'corners_home_ht': 0, 'corners_away_ht': 0,
            'shots_ot_home_ft': 0, 'shots_ot_away_ft': 0,
            'shots_ot_home_ht': 0, 'shots_ot_away_ht': 0
        }

        if not data or 'statistics' not in data:
            return stats

        def extract_val(groups, keywords, is_home):
            if not groups: return 0
            for g in groups:
                for item in g['statisticsItems']:
                    if any(k in item['name'].lower() for k in keywords):
                        try:
                            return int(item['home' if is_home else 'away'])
                        except:
                            return 0
            return 0

        # Periods
        stats_all = next((p['groups'] for p in data['statistics'] if p['period'] == 'ALL'), [])
        stats_1st = next((p['groups'] for p in data['statistics'] if p['period'] == '1ST'), [])
        
        # Corners
        stats['corners_home_ft'] = extract_val(stats_all, ['corner', 'escanteio'], True)
        stats['corners_away_ft'] = extract_val(stats_all, ['corner', 'escanteio'], False)
        stats['corners_home_ht'] = extract_val(stats_1st, ['corner', 'escanteio'], True)
        stats['corners_away_ht'] = extract_val(stats_1st, ['corner', 'escanteio'], False)
        
        # Shots on Target
        stats['shots_ot_home_ft'] = extract_val(stats_all, ['shots on target', 'chutes no gol'], True)
        stats['shots_ot_away_ft'] = extract_val(stats_all, ['shots on target', 'chutes no gol'], False)
        stats['shots_ot_home_ht'] = extract_val(stats_1st, ['shots on target', 'chutes no gol'], True)
        stats['shots_ot_away_ht'] = extract_val(stats_1st, ['shots on target', 'chutes no gol'], False)

        return stats
