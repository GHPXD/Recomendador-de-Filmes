# 🎬 Recomendador de Filmes Interativo

Este é um sistema de recomendação de filmes em linha de comando, construído em Python. O projeto vai além de uma simples recomendação, criando uma experiência interativa para o usuário, combinando múltiplas técnicas de Machine Learning para fornecer sugestões personalizadas e precisas.

---

## ✨ Features Principais

- **Sistema Híbrido:** Combina Filtragem por Conteúdo (analisando gêneros e tags) e Filtragem Colaborativa (aprendendo com o comportamento de outros usuários) para recomendações mais assertivas.
- **Busca Inteligente:** Utiliza correspondência de texto aproximada (fuzzy matching) para encontrar filmes mesmo que o usuário cometa erros de digitação.
- **Entrevista com o Usuário:** Para novos usuários ou quando se quer descobrir algo novo, o sistema pode fazer uma "entrevista" sobre gêneros preferidos para gerar uma lista inicial.
- **Feedback Interativo:** Permite que o usuário selecione uma das recomendações para "aprofundar" a busca e receber novas sugestões similares à sua escolha.
- **Conteúdo Enriquecido:** As recomendações de conteúdo não se baseiam apenas em gêneros, mas também em tags criadas pela comunidade, tornando a análise mais rica.

---

## 🎥 Demonstração de Uso

```plaintext
--- 🎬 Sistema de Recomendação de Filmes Híbrido 🎬 ---
1. Carregando datasets...
✅ Datasets carregados com sucesso!
2. Enriquecendo dados de conteúdo (Gêneros + Tags)...
✅ Conteúdo enriquecido!
3. Preparando o modelo de recomendação por conteúdo...
✅ Modelo de conteúdo pronto!
4. Treinando o modelo de filtragem colaborativa (SVD)...
✅ Modelo SVD treinado!
5. Preparando o sistema híbrido...

==================================================
✅ Sistema de Recomendação Pronto!
==================================================

Por favor, digite seu ID de usuário (ex: 1 a 610): 42

O que você gostaria de fazer?
[1] Receber recomendações baseadas em um filme que você gostou
[2] Descobrir filmes baseados nos seus gêneros preferidos
[3] Sair
Escolha uma opção: 1

Digite o nome de um filme: Batamn the dark knight
Você quis dizer 'Batman: The Dark Knight (2008)'? (sim/nao) sim

Calculando recomendações híbridas baseadas em 'Batman: The Dark Knight (2008)' para o usuário 42...

--- Aqui estão suas recomendações ---
[1] Dark Knight Rises, The (2012)
[2] Batman Begins (2005)
[3] Inception (2010)
[4] Matrix, The (1999)
[5] Fight Club (1999)
[6] Forrest Gump (1994)
[7] Pulp Fiction (1994)
[8] Star Wars: Episode V - The Empire Strikes Back (1980)
[9] Prestige, The (2006)
[10] Shawshank Redemption, The (1994)

Algum desses te interessou? Digite o número para ver filmes similares, ou 'voltar': 3

Ótima escolha! Procurando filmes parecidos com 'Inception (2010)'...

--- Aqui estão suas recomendações ---
[1] Shutter Island (2010)
[2] Prestige, The (2006)
[3] Memento (2000)
...
🛠️ Tecnologias Utilizadas
Python 3

Pandas – Manipulação e análise de dados.

Scikit-learn – Vetorização de texto (TfidfVectorizer) e cálculo de similaridade (cosine_similarity).

Scikit-surprise – Implementação de algoritmos de Filtragem Colaborativa (SVD).

Fuzzywuzzy – Correspondência aproximada de títulos de filmes.

NumPy – Base para computação científica em Python.

⚙️ Configuração do Ambiente e Instalação
1. Pré-requisitos
Python 3.8 ou superior.

2. Clone o Repositório
bash
Copiar
Editar
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
3. Crie um Ambiente Virtual (Recomendado)
Windows:

bash
Copiar
Editar
python -m venv venv
.\venv\Scripts\activate
macOS/Linux:

bash
Copiar
Editar
python3 -m venv venv
source venv/bin/activate
4. Instale as Dependências
Crie um arquivo requirements.txt com o seguinte conteúdo:

plaintext
Copiar
Editar
pandas
scikit-learn
scikit-surprise
numpy<2.0
fuzzywuzzy
python-Levenshtein
Instale as dependências com:

bash
Copiar
Editar
pip install -r requirements.txt
💡 Nota: A versão do numpy está fixada em <2.0 para garantir compatibilidade com o scikit-surprise.

5. Baixe os Dados do MovieLens
Acesse: GroupLens Datasets

Baixe a versão ml-latest-small.zip

Extraia o arquivo e copie os seguintes arquivos para a raiz do projeto:

Copiar
Editar
movies.csv
ratings.csv
tags.csv
Estrutura final esperada:

css
Copiar
Editar
seu-repositorio/
├── .venv/
├── main.py
├── movies.csv
├── ratings.csv
├── tags.csv
├── requirements.txt
└── README.md
🚀 Como Executar o Projeto
Com o ambiente ativado e os dados no lugar, execute:

bash
Copiar
Editar
python main.py
📖 Como Funciona
O sistema utiliza uma abordagem híbrida para gerar recomendações:

Filtragem por Conteúdo: Analisa características dos filmes (gêneros e tags) para encontrar similares.

Filtragem Colaborativa: Baseia-se no histórico de avaliações para identificar padrões entre usuários.

Sistema Híbrido: Combina os dois métodos, ponderando seus resultados para gerar uma lista final mais precisa e diversificada.

📄 Licença
Este projeto está sob a licença MIT. Consulte o arquivo LICENSE para mais informações.
