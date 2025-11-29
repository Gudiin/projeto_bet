# Sistema de Previsão de Escanteios com ML

Este projeto foi refatorado para utilizar Machine Learning na previsão de escanteios no Brasileirão Série A.

## Estrutura

- `src/scrapers`: Coleta de dados do SofaScore.
- `src/database`: Armazenamento em SQLite.
- `src/ml`: Engenharia de features e modelos de IA.
- `data/`: Onde o banco de dados e modelos salvos ficam.

## Como Usar

1.  **Instalar Dependências**:

    ```bash
    pip install -r requirements.txt
    playwright install chromium
    ```

2.  **Executar o Sistema**:

    ```bash
    python src/main.py
    ```

3.  **Fluxo Recomendado**:
    - Selecione a opção **1** para baixar o histórico de jogos da temporada. (Isso pode demorar um pouco).
    - Selecione a opção **2** para treinar a Inteligência Artificial com os dados baixados.
