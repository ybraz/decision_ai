# Decision AI Talent Match

Bem-vindo(a)! Meu nome é Yuri e este repositório é o meu trabalho final da pós-tech em machine learning. O objetivo aqui é construir um sistema de recomendação de candidatos para vagas de emprego usando boas práticas de MLOps e segurança.

Abaixo explico de forma bem direta o que fiz, por que fiz e como você pode reproduzir os resultados.

---

## Visão Geral

O projeto recebe três arquivos JSON (`applicants.json`, `vagas.json` e `prospects.json`) com dados de candidatos, vagas e resultados de processos seletivos. Depois da ingestão, geramos features de texto com SBERT e TF‑IDF e, por fim, treinamos um classificador LightGBM para ranquear os candidatos conforme a probabilidade de contratação.

Como entregável principal desenvolvi uma API REST em Python (FastAPI) que expõe um endpoint `/predict`. Assim é possível consultar a probabilidade de um candidato se adequar à vaga em tempo real. Toda a infraestrutura foi pensada para rodar em containers e seguir o fluxo de CI/CD no GitHub Actions.

---

## Como Executar

1. **Pré-requisitos**
   - Python 3.11 (uso pyenv, mas qualquer instalação compatível serve)
   - `virtualenv` ou similar
   - Docker, caso queira reproduzir a imagem final

2. **Instalação**
```bash
# clone o repositório
$ git clone git@github.com:suporte-ml/decision-ai.git && cd decision-ai

# ambiente virtual
$ python -m venv .venv && source .venv/bin/activate

# dependências de runtime e dev
$ pip install --upgrade pip
$ pip install -r requirements.txt
```
3. **Ingestão e features**
```bash
$ python -m decision_ai.data.ingest
$ python -m decision_ai.features.engineer
```
4. **Treinamento**
```bash
$ python -m decision_ai.models.train \
       --model lgbm \
       --trials 80 \
       --timeout 10800 \
       --calibrate sigmoid \
       --n_jobs 4
```
5. **Servir localmente**
```bash
$ uvicorn decision_ai.api.main:app --reload
```

Com esses passos a API ficará acessível em `http://localhost:8000/predict`.

6. **Testar a API**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: supersecret" \
  -d @payload.json
```

---

## Decisões de Projeto
Abaixo resumo as principais escolhas de arquitetura e por que cada uma delas foi adotada.
- **Features de texto**: optei por usar SBERT (multilingue) e TF-IDF para representar currículos e vagas.

- **Modelo**: LightGBM com otimização via Optuna. A matriz de features fica bastante esparsa e o LightGBM lida bem com esse cenário.
- **Pipeline**: implementei uma versão com Prefect para execução orquestrada e outra sequencial para rodar fácil em qualquer ambiente.
- **Segurança**: hashes de dados sensíveis antes de salvar, variáveis secretas fora do código e container sem usuário root.
- **Parâmetros de treino**: `--trials` define quantas buscas o Optuna executa, `--timeout` limita o tempo total, `--calibrate` ajusta as probabilidades e `--n_jobs` usa mais CPUs.

---

## Lições Aprendidas

- **Validação de dados** é essencial: erros nos JSONs atrapalhavam todo o fluxo, então investi em esquemas Pydantic bem completos.
- **Sparsidade** dos dados textuais é um problema real. Reduzir a dimensionalidade com SVD ajudou, mas ainda vejo margem para melhorar a representação.
- **Automatização** com CI/CD simplificou os testes e a geração de imagem Docker. Aprendi a lidar com falhas de dependência e a manter o build reproduzível.
- **Métricas**: acompanhar ROC-AUC e PR-AUC em conjunto forneceu uma visão mais realista do comportamento do modelo, já que o conjunto é bem desbalanceado.

---

## Estrutura do Repositório
```
decision_ai/
├── data/           # dados brutos e processados
├── src/decision_ai # código fonte principal
├── tests/          # unitários e integração
└── .github/        # fluxos de CI/CD
```

---

Espero que este README facilite a avaliação do projeto. Qualquer dúvida ou sugestão estou à disposição!

