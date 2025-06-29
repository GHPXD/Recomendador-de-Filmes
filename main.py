# -*- coding: utf-8 -*-
"""
🎬 Recomendador de Filmes Interativo - Versão Final 🎬

Descrição: Uma aplicação de console que combina:
1.  Recomendação Híbrida (Conteúdo + Colaborativa).
2.  Correção de digitação de títulos de filmes (Fuzzy Matching).
3.  Entrevista com o usuário para entender seus gostos por gênero.
4.  Feedback interativo para refinar as buscas.
"""

# --- Etapa 1: Importar Bibliotecas ---
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from fuzzywuzzy import process
import numpy as np

# --- Etapa 2: Funções de Carregamento e Preparação ---
def load_and_prepare_data():
    """Carrega e prepara todos os datasets necessários."""
    print("1. Carregando datasets (movies, ratings, tags)...")
    try:
        movies_df = pd.read_csv('movies.csv')
        ratings_df = pd.read_csv('ratings.csv')
        tags_df = pd.read_csv('tags.csv')
    except FileNotFoundError:
        print("Erro: Arquivos .csv não encontrados.")
        return None, None, None

    print("2. Enriquecendo dados de conteúdo (Gêneros + Tags)...")
    movies_df['genres'] = movies_df['genres'].str.replace('|', ' ', regex=False).fillna('')
    tags_df['tag'] = tags_df['tag'].fillna('')
    movie_tags = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    movies_df = pd.merge(movies_df, movie_tags, on='movieId', how='left')
    movies_df['tag'] = movies_df['tag'].fillna('')
    movies_df['content'] = movies_df['genres'] + ' ' + movies_df['tag']
    
    return movies_df, ratings_df, tags_df

# --- Etapa 3: Funções dos Modelos de Recomendação ---

def prepare_content_model(movies_df):
    """Prepara o modelo TF-IDF para recomendação de conteúdo."""
    print("3. Preparando o modelo de recomendação por conteúdo...")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['content'])
    indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()
    return tfidf_vectorizer, tfidf_matrix, indices

def train_svd_model(ratings_df):
    """Treina o modelo SVD para filtragem colaborativa."""
    print("4. Treinando o modelo de filtragem colaborativa (SVD)...")
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    svd_model = SVD(n_factors=150, n_epochs=30, lr_all=0.005, reg_all=0.04, random_state=42)
    svd_model.fit(trainset)
    return svd_model

# Funções de geração de recomendação (Conteúdo, Colaborativa, Híbrida)
def get_content_based_recommendations(title, indices, movies_df, tfidf_matrix, count=10):
    if title not in indices: return pd.DataFrame(columns=['title', 'score'])
    idx = indices[title]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)
    sim_scores = list(enumerate(sim_scores[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:count+1]
    movie_indices = [i[0] for i in sim_scores]
    recommendations = movies_df.iloc[movie_indices][['title']].copy()
    recommendations['score'] = [score for _, score in sim_scores]
    return recommendations

def get_collaborative_recommendations(user_id, svd_model, movies_df, ratings_df, count=10):
    all_movie_ids = movies_df['movieId'].unique()
    movies_rated = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()
    predictions = [ (mid, svd_model.predict(user_id, mid).est) for mid in all_movie_ids if mid not in movies_rated ]
    predictions.sort(key=lambda x: x[1], reverse=True)
    rec_ids = [mid for mid, _ in predictions[:count]]
    recommendations = movies_df[movies_df['movieId'].isin(rec_ids)].copy()
    pred_map = dict(predictions[:count])
    recommendations['score'] = recommendations['movieId'].map(pred_map)
    return recommendations.sort_values('score', ascending=False)[['title', 'score']]

def get_hybrid_recommendations(user_id, title, svd_model, indices, movies_df, ratings_df, tfidf_matrix, count=10):
    content_recs = get_content_based_recommendations(title, indices, movies_df, tfidf_matrix, 20)
    content_recs['score'] *= 0.4
    collab_recs = get_collaborative_recommendations(user_id, svd_model, movies_df, ratings_df, 20)
    collab_recs['score'] = (collab_recs['score'] / 5.0) * 0.6
    hybrid_recs = pd.concat([content_recs, collab_recs]).groupby('title')['score'].sum().reset_index()
    hybrid_recs = hybrid_recs[hybrid_recs['title'] != title]
    return hybrid_recs.sort_values('score', ascending=False).head(count)

# --- Etapa 4: Funções Interativas ---

def find_closest_title(input_title, all_titles):
    """Encontra o título de filme mais próximo de uma entrada do usuário."""
    closest_match = process.extractOne(input_title, all_titles)
    return closest_match[0] if closest_match and closest_match[1] > 80 else None

def interview_user(tfidf_vectorizer, tfidf_matrix, movies_df, indices):
    """Faz perguntas ao usuário para criar um perfil de gosto e gerar recomendações."""
    print("\n--- Entrevista de Gosto ---")
    print("Para te dar boas recomendações, responda 'sim' ou 'nao'.\n")
    genres = ['Action', 'Adventure', 'Sci-Fi', 'Fantasy', 'Comedy', 'Drama', 'Thriller', 'Romance', 'Animation', 'Crime']
    prefs = [ g for g in genres if input(f"Você gosta de filmes de {g}? ").lower() == 'sim' ]
    if not prefs:
        print("Ok, sem preferências. Não posso recomendar assim.")
        return
    print("\nÓtimo! Procurando filmes baseados no seu gosto...")
    ideal_vec = tfidf_vectorizer.transform([' '.join(prefs)])
    sim_scores = list(enumerate(cosine_similarity(ideal_vec, tfidf_matrix)[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[0:10]
    movie_indices = [i[0] for i in sim_scores]
    recommendations = movies_df.iloc[movie_indices][['title']].copy()
    recommendations['score'] = [s[1] for s in sim_scores]
    display_and_get_feedback(recommendations, indices, movies_df, tfidf_matrix)

def display_and_get_feedback(recommendations, indices, movies_df, tfidf_matrix):
    """Apresenta as recomendações e pede feedback para refinar a busca."""
    if recommendations.empty:
        print("Não foi possível gerar recomendações.")
        return
        
    print("\n--- Aqui estão suas recomendações ---")
    rec_list = recommendations['title'].tolist()
    for i, title in enumerate(rec_list):
        print(f"[{i+1}] {title}")
    
    while True:
        feedback = input("\nAlgum desses te interessou? Digite o número para ver filmes similares, ou 'voltar': ").lower()
        if feedback == 'voltar': break
        try:
            choice = int(feedback) - 1
            if 0 <= choice < len(rec_list):
                chosen_movie = rec_list[choice]
                print(f"\nÓtima escolha! Procurando filmes parecidos com '{chosen_movie}'...")
                refined_recs = get_content_based_recommendations(chosen_movie, indices, movies_df, tfidf_matrix)
                display_and_get_feedback(refined_recs, indices, movies_df, tfidf_matrix)
                break
            else:
                print("Número inválido.")
        except ValueError:
            print("Entrada inválida. Por favor, digite um número ou 'voltar'.")

# --- Etapa 5: Fluxo Principal da Aplicação ---
def main():
    """Função principal que executa o loop interativo da aplicação."""
    # Preparação única
    movies_df, ratings_df, tags_df = load_and_prepare_data()
    if movies_df is None: return
    
    tfidf_v, tfidf_m, indices = prepare_content_model(movies_df)
    svd_model = train_svd_model(ratings_df)
    all_titles = movies_df['title'].tolist()
    valid_user_ids = ratings_df['userId'].unique()

    print("\n" + "="*50)
    print("✅ Sistema de Recomendação Pronto!")
    print("="*50)

    # Loop principal
    while True:
        try:
            user_id = int(input("\nPor favor, digite seu ID de usuário (ex: 1 a 610): "))
            if user_id not in valid_user_ids:
                print("ID de usuário inválido. Tente novamente.")
                continue
        except ValueError:
            print("Por favor, digite um número válido.")
            continue

        print("\nO que você gostaria de fazer?")
        print("[1] Receber recomendações baseadas em um filme que você gostou")
        print("[2] Descobrir filmes baseados nos seus gêneros preferidos")
        print("[3] Sair")
        choice = input("Escolha uma opção: ")

        if choice == '1':
            input_title = input("\nDigite o nome de um filme: ")
            matched_title = find_closest_title(input_title, all_titles)
            if not matched_title:
                print(f"Desculpe, não encontrei nada parecido com '{input_title}'.")
                continue
            
            if matched_title.lower() != input_title.lower():
                if input(f"Você quis dizer '{matched_title}'? (sim/nao) ").lower() != 'sim':
                    continue
            
            print(f"\nCalculando recomendações híbridas baseadas em '{matched_title}' para o usuário {user_id}...")
            recs = get_hybrid_recommendations(user_id, matched_title, svd_model, indices, movies_df, ratings_df, tfidf_m)
            display_and_get_feedback(recs, indices, movies_df, tfidf_m)

        elif choice == '2':
            interview_user(tfidf_v, tfidf_m, movies_df, indices)

        elif choice == '3':
            print("Até a próxima!")
            break
        else:
            print("Opção inválida. Tente novamente.")


if __name__ == "__main__":
    main()