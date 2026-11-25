# Análise Empírica do Problema do Caixeiro Viajante
---
## 📋 Descrição do Problema
O objetivo deste projeto é executar três algoritmos diferentes para o TSP, e coletar dados das execuções para fazer uma análise empírica de complexidade. Para mais detalhes, leia o relatório em ```docs/relatorio.pdf```.

---

## 📂 Estrutura do projeto
```sh
├─ data/            # Instâncias do TSP em formato .tsp
│ └─ ...
├─ docs/            # Tudo relacionado a documentação do trabalho
│ ├─ prompt_history/    # Histórico de prompts utilizados nas LLMs
│ │  └─ ...
│ ├─ relatorio_latex/   # Relatório em LaTeX
│ │  └─ ...
│ └─ relatorio.pdf      # Relatório em pdf
├─ raw_ai_scripts/  # Scripts sem modificação (não utilizados)
│ └─ ...
└─ scripts/         # Scripts refatoratos (efetivos)
  └─ ...
```

---

## ▶️ Instruções de Execução

**⚠️ ATENÇÃO:** O programa foi testado para Python 3.13.15. Não podemos garantir que a execução será exata em outras versões.
Para configurar os parâmetros da execução, vá em ```scripts/main.py``` e configure as variáveis globais no cabeçalho do script como desejar.

1. Abra o terminal na pasta do projeto;

2. Rode o comando: ```./run.sh```