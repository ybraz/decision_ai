# Decision AI Talent Match

**Abstract**  
Este documento apresenta o projeto Decision AI Talent Match, desenvolvido como trabalho final do curso de Machine Learning Engineering. A solução integra pipelines de ingestão, engenharia de features, treinamento, avaliação e API REST para predição de candidatos, adotando boas práticas de MLOps, segurança de dados e metodologias acadêmicas de avaliação de desempenho.

## 1. Introdução
O recrutamento de talentos é um processo crítico e complexo, frequentemente limitado por vieses humanos e pela dificuldade de analisar grandes volumes de currículos. Este projeto propõe uma abordagem híbrida de Processamento de Linguagem Natural (PLN) e Aprendizado de Máquina para ranquear candidatos segundo sua probabilidade de contratação, fornecendo métricas quantificáveis e auditáveis. A arquitetura é projetada para ser reprodutível, escalável e segura.

## 2. Arquitetura do Sistema
A solução é composta por cinco módulos principais:
1. **Ingestão de Dados**: leitura e validação de arquivos JSON, normalização de schema e hashing de dados sensíveis.  
2. **Engenharia de Features**: extração de representações textuais (SBERT, TF‑IDF+SVD), codificação de variáveis estruturadas e cálculo de similaridade semântica.  
3. **Treinamento de Modelo**: otimização de hiperparâmetros via Optuna, balanceamento de classes e calibração de probabilidades usando ROC-AUC.  
4. **Avaliação**: geração de métricas (AUC, precisão, recall, F1) e visualizações (curvas ROC e PR).  
5. **API REST**: serviço web com FastAPI, protegido por API Key, para predição em tempo real.

## 3. Pipeline de Dados
### 3.1 Ingestão de Dados
- Utiliza Pydantic para validação de esquema.  
- Converte JSON de candidatos (`applicants.json`), vagas (`vagas.json`) e interações (`prospects.json`) em tabelas no formato *star schema* (`dim_applicant`, `dim_job`, `fact_prospect`).  
- Aplica hashing SHA‑256 em colunas sensíveis (telefone, e-mail).  
- Salva artefatos em Parquet e versiona com DVC para rastreabilidade.

### 3.2 Engenharia de Features
- **SBERT Embeddings** nos campos `cv_text` e `job_text`, capturando semântica textual.  
- **TF‑IDF + TruncatedSVD** para descrição de vagas, reduzindo dimensionalidade e extraindo componentes latentes.  
- **One-Hot Encoding** para variáveis categóricas (tipo de contratação, senioridade, etc.).  
- **Feature de Similaridade de Cosseno** entre embeddings de currículo e vaga.  
- **Passthrough** de atributos numéricos e flags de skills.

### 3.3 Treinamento do Modelo
- Carrega matriz de features `X` e rótulos `y`.  
- Executa busca de hiperparâmetros com Optuna, otimizando LightGBM e ajustando `scale_pos_weight`.  
- Realiza calibração de probabilidades usando `CalibratedClassifierCV(method='sigmoid')`.  
- Persiste modelo final via joblib.

### 3.4 Avaliação do Modelo
- Avalia métricas de classificação: AUC, precisão, recall e F1-score.  
- Gera curvas ROC e Precision-Recall com matplotlib.  
- Exporta relatórios e gráficos para diretório selecionado.

### 3.5 Serviço de Predição (API REST)
- Endpoints principais:  
  - `POST /predict`: recebe JSON com `cv_text`, `job_text` e demais features, retorna `proba` e `pred`.  
  - `GET /healthz`: verifica estado de saúde do serviço.  
- Autenticação via header `X-API-Key`.  
- Containerizado com Docker e configurado para CI/CD em GitHub Actions.

## 4. Decisões de Projeto
- **Representação de Texto**: combinação de embeddings densos (SBERT) e vetores esparsos (TF‑IDF+SVD) para robustez semântica e desempenho.  
- **Modelagem**: LightGBM por sua eficiência e desempenho em dados tabulares, com balanceamento de classes e calibração de probabilidades.  
- **Otimização de Hiperparâmetros**: Optuna para exploração automática de espaços de busca e maximização de ROC-AUC.  
- **Rastreabilidade de Artefatos**: uso de DVC para versionamento de datasets e pipelines.  
- **Segurança de Dados**: hashing de campos sensíveis e validação de esquema para conformidade e privacidade.

## 5. Lições Aprendidas
1. Validação de dados é crucial para evitar falhas silenciosas em pipelines de produção.  
2. Hashing de dados sensíveis equilibra privacidade com necessidade de rastreabilidade.  
3. Pipeline modular facilita manutenção, testes e reuso de componentes.  
4. Calibração de probabilidades melhora interpretabilidade e confiabilidade do modelo.  
5. Automação CI/CD reduz erros manuais e acelera entregas.

## 6. Uso – Passo a Passo
1. **Clonar repositório e ativar ambiente**  
   ```bash
   git clone git@github.com:ybraz/decision_ai.git
   cd decision_ai
   python -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```  
2. **Ingestão de Dados**  
   ```bash
   python -m decision_ai.data.ingest \
     --raw-dir src/data/raw \
     --out-dir data/processed
   ```  
3. **Engenharia de Features**  
   ```bash
   python -m decision_ai.features.engineer \
     --tfidf-dim 30000 \
     --svd-dim 512
   ```  
4. **Treinamento do Modelo**  
   ```bash
   python -m decision_ai.models.train --trials 80
   ```  
5. **Avaliação do Modelo**  
   ```bash
   python -m decision_ai.models.evaluate --threshold 0.25 --export reports/
   ```  
6. **Executar API REST**  
   ```bash
   uvicorn decision_ai.api.main:app --reload
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -H "X-API-Key: supersecret" \
     -d @payload.json
   ```

## 7. Estrutura do Repositório
```text
decision_ai/
├── data/                   # Dados brutos e processados (DVC)
├── src/decision_ai/        # Código-fonte
│   ├── api/                # FastAPI endpoints
│   ├── data/               # Scripts de ingestão
│   ├── features/           # Engenharia de features
│   └── models/             # Treinamento, avaliação e predição
├── tests/                  # Testes unitários e de integração
└── .github/                # Configuração de CI/CD
```

## 8. Referências
- Wu, Y. et al. (2019). **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**.  
- Ke, G. et al. (2017). **LightGBM: A Highly Efficient Gradient Boosting Decision Tree**.  
- Akiba, T. et al. (2019). **Optuna: A Next-generation Hyperparameter Optimization Framework**.  
- Niculescu-Mizil, A. & Caruana, R. (2005). **Predicting Good Probabilities with Supervised Learning**.  
- OWASP Foundation. **OWASP Secure Software Development Framework (SSDF)**.  
- DVC. **Data Version Control** Documentation.  
