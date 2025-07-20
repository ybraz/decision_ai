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

Apresento a seguir as decisões arquiteturais e as respectivas justificativas técnicas para o desenvolvimento do sistema.

* **Representação de Texto:** Para a vetorização de currículos e descrições de vagas, foram empregadas duas abordagens complementares:
    * **SBERT (Sentence-BERT) Multilíngue:** Selecionado por sua capacidade de gerar embeddings densos e semanticamente ricos, essenciais para capturar nuances contextuais em múltiplos idiomas, o que otimiza a correspondência entre documentos textuais.
    * **TF-IDF (Term Frequency-Inverse Document Frequency):** Utilizado para capturar a importância de termos específicos dentro dos documentos. Essa representação esparsa é particularmente eficaz para identificar palavras-chave distintivas, complementando a representação densa do SBERT.

* **Modelo Preditivo:**
    * **LightGBM (Light Gradient Boosting Machine):** Escolhido como o algoritmo principal devido à sua comprovada eficiência e desempenho em conjuntos de dados com alta dimensionalidade e esparsidade. Sua arquitetura otimizada permite um treinamento rápido e escalável, tornando-o ideal para lidar com a matriz de características gerada pelas representações textuais.
    * **Otimização de Hiperparâmetros via Optuna:** A ferramenta Optuna foi integrada para a otimização automática dos hiperparâmetros do LightGBM. Essa abordagem bayesiana explora o espaço de busca de forma eficiente, resultando em modelos com performance preditiva superior e reduzindo a necessidade de ajustes manuais.

* **Orquestração e Execução do Pipeline:**
    * **Pipeline Orquestrado com Prefect:** Uma versão do pipeline foi desenvolvida utilizando **Prefect** para orquestração de fluxos de trabalho. Esta escolha garante robustez, monitoramento em tempo real, tratamento de falhas e reexecução de tarefas, otimizando a governança e a confiabilidade das operações do sistema.
    * **Pipeline Sequencial:** Adicionalmente, uma implementação sequencial foi providenciada para ambientes com menor complexidade de infraestrutura, permitindo a execução facilitada em diversos cenários sem a necessidade de uma orquestração dedicada.

* **Segurança da Informação:** As seguintes medidas foram implementadas para garantir a integridade e confidencialidade dos dados:
    * **Hashing de Dados Sensíveis:** Antes do armazenamento, os dados sensíveis são submetidos a algoritmos de hashing, protegendo as informações contra acessos não autorizadas e vazamentos.
    * **Gestão de Variáveis Secretas:** Variáveis e credenciais secretas são armazenadas externamente ao código-fonte, utilizando práticas recomendadas de segurança para evitar exposição e facilitar a gestão.
    * **Execução em Contêiner sem Usuário Root:** Os contêineres de execução são configurados para operar com privilégios mínimos (sem usuário root), reduzindo a superfície de ataque e mitigando riscos de segurança.

* **Parâmetros de Treinamento e Otimização:** Os parâmetros cruciais para o processo de treinamento e otimização do modelo são:
    * `--trials`: Define o número de iterações de busca que o Optuna executará para encontrar a melhor combinação de hiperparâmetros.
    * `--timeout`: Estabelece um limite de tempo máximo para a execução total do processo de otimização, garantindo a conclusão dentro de um prazo predefinido.
    * `--calibrate`: Habilita a calibração de probabilidades do modelo, ajustando as saídas para que representem probabilidades mais precisas e confiáveis.
    * `--n_jobs`: Especifica o número de CPUs a serem utilizadas em paralelo durante o processo de otimização, acelerando significativamente o tempo de execução.

---

## Lições Aprendidas e Perspectivas Futuras

Durante o desenvolvimento e a implementação do sistema, diversas lições cruciais foram extraídas, as quais impactaram diretamente as escolhas arquitetônicas e metodológicas.

* **Validação de Dados Robustas:** A importância da **validação de dados** tornou-se evidente ao longo do projeto. Erros sutis ou inconsistências nos arquivos JSON de entrada frequentemente causavam interrupções inesperadas em todo o fluxo de processamento. Para mitigar esse problema, foi implementada uma validação rigorosa utilizando **esquemas Pydantic** abrangentes. Essa abordagem permitiu não apenas a detecção precoce de irregularidades, mas também garantiu a integridade e a conformidade dos dados, otimizando a confiabilidade do pipeline.

* **Desafios da Esparsidade em Dados Textuais:** A **esparsidade intrínseca dos dados textuais** representou um desafio significativo na fase de representação e modelagem. Embora a aplicação da **Decomposição por Valores Singulares (SVD)** tenha sido eficaz na redução da dimensionalidade e na captura de padrões latentes, ainda se observa uma margem considerável para aprimoramento. Futuras investigações podem explorar técnicas mais avançadas de embedding e representação de texto para otimizar a densidade e a expressividade das características, potencializando o desempenho do modelo em cenários com alta esparsidade.

* **Benefícios da Automatização via CI/CD:** A adoção de um pipeline de **Integração Contínua e Entrega Contínua (CI/CD)** foi fundamental para a eficiência do processo de desenvolvimento. Essa automatização simplificou drasticamente a execução de testes, a verificação de código e a geração de imagens Docker reprodutíveis. Além disso, a experiência adquirida na gestão de **falhas de dependência** e na garantia da **reprodutibilidade dos builds** solidificou as práticas de engenharia de software, contribuindo para um ciclo de desenvolvimento mais ágil e confiável.

* **Avaliação Abrangente do Modelo com Métricas Adequadas:** No contexto de um conjunto de dados **desbalanceado**, o acompanhamento conjunto das métricas **ROC-AUC (Receiver Operating Characteristic - Area Under the Curve)** e **PR-AUC (Precision-Recall - Area Under the Curve)** revelou-se indispensável. Enquanto a ROC-AUC forneceu uma visão geral do poder discriminatório do modelo, a PR-AUC ofereceu uma perspectiva mais realista do seu comportamento em classes minoritárias, sendo crucial para entender o desempenho em cenários onde a precisão e o *recall* da classe positiva são críticos. Essa abordagem multifacetada garantiu uma avaliação mais precisa e um entendimento aprofundado do desempenho do modelo.

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

