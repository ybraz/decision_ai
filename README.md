# Decision AI Talent Match

Bem-vindo(a)! Este repositório é sobre o trabalho final da pós-tech em machine learning engineering. O objetivo aqui é construir um sistema de recomendação de candidatos para vagas de emprego usando boas práticas de MLOps e segurança.

---

## Visão Geral

O projeto ingere dados de currículos e descrições de vagas para, em seguida, aplicar um **Cross-Encoder baseado em Transformer** que avalia diretamente o **grau de similaridade semântica** entre pares (CV, vaga). Utiliza-se o modelo `cross-encoder/ms-marco-MiniLM-L-6-v2` (Reimers & Gurevych, 2020), pré-treinado em tarefas de recuperação de informação sobre o corpus MS MARCO (Nguyen et al., 2016). Essa abordagem unificada dispensa a extração explícita de features estruturais e aproveita o poder contextual do Transformer para gerar scores contínuos em [0,1].

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
3. **Ingestão de dados e Engenharia de Features**
   
   Os scripts estão localizados em `src/decision_ai/data/ingest.py` e `src/decision_ai/features/engineer.py`. Para executar, use:

   ```bash
   # Executa a etapa de ingestão (geralmente lê raw → processado)
   python -m decision_ai.data.ingest \
       --raw-dir src/data/raw \
       --out-dir src/data/processed

   # Executa a engenharia de features (TF-IDF + SBERT + pipeline enxuto)
   python -m decision_ai.features.engineer \
       --tfidf-dim 30000 \
       --svd-dim 512
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

5. **Avaliação do Modelo**
```bash
# avaliação rápida com threshold padrão (0.50)
$ python -m decision_ai.models.evaluate

# avaliando com threshold customizado (ex.: 0.25)
$ python -m decision_ai.models.evaluate --threshold 0.25

# salvando relatórios e gráficos em diretório específico
$ python -m decision_ai.models.evaluate --threshold 0.25 --export reports/
```


> **Interpretação do *threshold***  
> A escolha do ponto de corte é agora uma **decisão de negócio**.  
> * **Mais *recall*** (não perder talentos): use `--threshold 0.25` – o modelo captura ~80 % dos candidatos contratáveis, mas com ~15 % de falsos‑positivos.  
> * **Menos ruído** para o time de seleção: use `--threshold 0.50` (ou até `0.60`) – quase nenhum falso‑positivo, porém ~30 % dos bons perfis ficam de fora.  
> Ajuste conforme o trade‑off entre volume de triagem manual e risco de perder bons perfis.


6. **Servir localmente**
```bash
$ uvicorn decision_ai.api.main:app --reload
```

Com esses passos a API ficará acessível em `http://localhost:8000/predict`.

7. **Testar a API**
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
    * **Cross-Encoder Transformer:** Adotado por sua capacidade de capturar **interações bidirecionais** entre tokens de texto de CV e de vaga, superando abordagens dual-encoder quando a granularidade semântica é crítica (Reimers & Gurevych, 2020). O modelo `ms-marco-MiniLM-L-6-v2` foi escolhido por seu **trade‑off** entre precisão e eficiência computacional, apresentando alto desempenho em tarefas de similaridade/recuperação.<br/>
    * **Inferência por escore contínuo:** O Cross-Encoder retorna logits convertidos em probabilidades via **sigmoid**, permitindo ajustar o ponto de corte (`threshold`) conforme critérios de negócio de recall vs. precisão.

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
