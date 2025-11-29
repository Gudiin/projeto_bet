import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom
from tabulate import tabulate

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"

class StatisticalAnalyzer:
    def __init__(self):
        pass

    def monte_carlo_simulation(self, lambda_val, var_val, n_sims=10000):
        if var_val > lambda_val:
            # Negative Binomial
            p = lambda_val / var_val
            n = (lambda_val ** 2) / (var_val - lambda_val)
            sims = nbinom.rvs(n, p, size=n_sims)
        else:
            # Poisson
            sims = poisson.rvs(lambda_val, size=n_sims)
        return sims

    def generate_suggestions(self, opportunities, ml_prediction=None):
        # Filter opportunities to find Easy, Medium, Hard
        suggestions = {
            "Easy": None,
            "Medium": None,
            "Hard": None
        }
        
        # Sort by probability (descending)
        sorted_ops = sorted(opportunities, key=lambda x: x['Prob'], reverse=True)
        
        # Helper to check alignment with ML
        def aligns_with_ml(op):
            if ml_prediction is None: return True
            # If ML predicts high corners (e.g. 11.7), favor Overs
            # If ML predicts low corners (e.g. 8.0), favor Unders
            # This is a simple heuristic
            if "Over" in op['Seleção'] and ml_prediction > 10.5: return True
            if "Under" in op['Seleção'] and ml_prediction < 9.5: return True
            # If ML is neutral (9.5-10.5), accept both
            if 9.5 <= ml_prediction <= 10.5: return True
            return False

        # Easy: High probability (> 70%), Low Odd (~1.30 - 1.50)
        for op in sorted_ops:
            if op['Prob'] >= 0.70 and 1.25 <= op['Odd'] <= 1.60:
                if aligns_with_ml(op):
                    suggestions["Easy"] = op
                    break
        
        # Medium: Medium probability (50% - 70%), Medium Odd (~1.60 - 2.00)
        for op in sorted_ops:
            if 0.50 <= op['Prob'] < 0.75 and 1.60 <= op['Odd'] <= 2.20:
                if aligns_with_ml(op):
                    suggestions["Medium"] = op
                    break
                
        # Hard: Lower probability (< 50%), High Odd (> 2.20) - Value Bet
        for op in sorted_ops:
            if 0.30 <= op['Prob'] < 0.55 and op['Odd'] > 2.20:
                if aligns_with_ml(op):
                    suggestions["Hard"] = op
                    break
                
        return suggestions

    def analyze_match(self, df_home, df_away, ml_prediction=None, match_name=None):
        # df_home/df_away should contain columns: 
        # 'corners_ft', 'corners_ht', 'corners_2t', 'shots_ht'
        
        if df_home.empty or df_away.empty:
            print(f"{Colors.RED}Dados insuficientes para análise estatística.{Colors.RESET}")
            return {}

        if match_name:
            print(f"\n⚽ {Colors.BOLD}{match_name}{Colors.RESET}")

        mercados = [
            {"nome": "JOGO COMPLETO", "df_h": df_home['corners_ft'], "df_a": df_away['corners_ft'],
             "linhas": [8.5, 9.5, 10.5, 11.5, 12.5]},
            {"nome": "TOTAL MANDANTE", "df_h": df_home['corners_ft'], "df_a": None, "linhas": [4.5, 5.5, 6.5]},
            {"nome": "TOTAL VISITANTE", "df_h": None, "df_a": df_away['corners_ft'], "linhas": [3.5, 4.5, 5.5]},
            
            {"nome": "1º TEMPO (HT)", "df_h": df_home['corners_ht'], "df_a": df_away['corners_ht'],
             "linhas": [3.5, 4.5, 5.5]},
            {"nome": "2º TEMPO (FT)", "df_h": df_home['corners_2t'], "df_a": df_away['corners_2t'], 
             "linhas": [3.5, 4.5, 5.5]},
             
            {"nome": "MANDANTE 1º TEMPO", "df_h": df_home['corners_ht'], "df_a": None, "linhas": [1.5, 2.5, 3.5]},
            {"nome": "VISITANTE 1º TEMPO", "df_h": None, "df_a": df_away['corners_ht'], "linhas": [1.5, 2.5, 3.5]},
            
            {"nome": "MANDANTE 2º TEMPO", "df_h": df_home['corners_2t'], "df_a": None, "linhas": [1.5, 2.5, 3.5]},
            {"nome": "VISITANTE 2º TEMPO", "df_h": None, "df_a": df_away['corners_2t'], "linhas": [1.5, 2.5, 3.5]}
        ]

        oportunidades = []

        print("\n" + "▓" * 80)
        print(f" 🧠 CÉREBRO ESTATÍSTICO (Monte Carlo)")
        print("▓" * 80)

        for m in mercados:
            lambdas = []
            vars_val = []

            for df in [m['df_h'], m['df_a']]:
                if df is not None:
                    mean_10 = df.mean()
                    mean_5 = df.head(5).mean()
                    l_ajustado = (mean_10 * 0.6) + (mean_5 * 0.4)
                    lambdas.append(l_ajustado)
                    vars_val.append(df.var())

            lambda_final = sum(lambdas)
            var_final = sum(vars_val)
            if len(lambdas) == 1: var_final = vars_val[0]
            
            # Handle NaN variance (single game history)
            if pd.isna(var_final): var_final = lambda_final

            # Bonus Pressão (Simplificado, pois não temos chutes no DB ainda corretamente mapeados as vezes)
            # if "1º TEMPO" in m['nome']: ...

            simulacoes = self.monte_carlo_simulation(lambda_final, var_final)

            for linha in m['linhas']:
                # OVER
                prob_over = (simulacoes > linha).mean()
                odd_justa_over = 1 / prob_over if prob_over > 0 else 99
                
                if 1.20 <= odd_justa_over <= 3.00: # Range mais amplo para capturar sugestões
                    cv = (var_final ** 0.5) / lambda_final if lambda_final > 0 else 1
                    score = prob_over * (1 - (cv * 0.3))
                    oportunidades.append({
                        "Mercado": m['nome'], "Seleção": f"Over {linha}", 
                        "Prob": prob_over, "Odd": odd_justa_over,
                        "Score": score, "Tipo": "OVER"
                    })

                # UNDER
                prob_under = (simulacoes <= linha).mean()
                odd_justa_under = 1 / prob_under if prob_under > 0 else 99
                
                if 1.20 <= odd_justa_under <= 2.50:
                    cv = (var_final ** 0.5) / lambda_final if lambda_final > 0 else 1
                    score = prob_under * (1 - (cv * 0.5))
                    oportunidades.append({
                        "Mercado": m['nome'], "Seleção": f"Under {linha}", 
                        "Prob": prob_under, "Odd": odd_justa_under,
                        "Score": score, "Tipo": "UNDER"
                    })

        oportunidades.sort(key=lambda x: x['Score'], reverse=True)
        top_picks = oportunidades[:7]

        # --- TABELA FORMATADA ---
        print(f"\n🏆 {Colors.BOLD}TOP 7 OPORTUNIDADES (DATA DRIVEN){Colors.RESET}")
        tabela_display = []
        for pick in top_picks:
            if pick['Tipo'] == "OVER":
                cor = Colors.GREEN
                seta = "▲"
            else:
                cor = Colors.CYAN
                seta = "▼"

            linha_fmt = f"{cor}{pick['Seleção']}{Colors.RESET}"
            prob_fmt = f"{pick['Prob'] * 100:.1f}%"
            odd_fmt = f"{Colors.BOLD}@{pick['Odd']:.2f}{Colors.RESET}"
            direcao_fmt = f"{cor}{seta} {pick['Tipo']}{Colors.RESET}"

            tabela_display.append([pick['Mercado'], linha_fmt, prob_fmt, odd_fmt, direcao_fmt])

        headers = ["MERCADO", "LINHA", "PROB.", "ODD JUSTA", "TIPO"]
        print(tabulate(tabela_display, headers=headers, tablefmt="fancy_grid", stralign="center"))
        
        # --- SUGESTÕES ---
        suggestions = self.generate_suggestions(oportunidades)
        print(f"\n🎯 {Colors.BOLD}SUGESTÕES DA IA:{Colors.RESET}")
        
        for level, pick in suggestions.items():
            if pick:
                cor_nivel = Colors.GREEN if level == "Easy" else (Colors.YELLOW if level == "Medium" else Colors.RED)
                print(f"{cor_nivel}[{level.upper()}]{Colors.RESET} {pick['Mercado']} - {pick['Seleção']} (@{pick['Odd']:.2f}) | Prob: {pick['Prob']*100:.1f}%")
            else:
                print(f"[{level.upper()}] Nenhuma oportunidade encontrada.")

        return top_picks
