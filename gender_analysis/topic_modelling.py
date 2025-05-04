def topic_modeling_by_gender(df, num_topics=5):
    """Perform topic modeling to identify gender-based topic differences."""
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Process text by gender
    male_texts = df[df['gender'] == 'Male']['full_text'].fillna('').tolist()
    female_texts = df[df['gender'] == 'Female']['full_text'].fillna('').tolist()
    
    # Create vectorizer
    vectorizer = CountVectorizer(max_features=1000, stop_words='english', min_df=2)
    
    # Process male documents
    male_vectors = vectorizer.fit_transform(male_texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Topic modeling for male documents
    male_lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    male_lda.fit(male_vectors)
    
    # Get top words for each topic in male documents
    male_topics = []
    for topic_idx, topic in enumerate(male_lda.components_):
        top_words_idx = topic.argsort()[:-10 - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        male_topics.append({
            'topic_id': topic_idx,
            'top_words': top_words
        })
    
    # Refit vectorizer for female documents
    vectorizer = CountVectorizer(max_features=1000, stop_words='english', min_df=2)
    female_vectors = vectorizer.fit_transform(female_texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Topic modeling for female documents
    female_lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    female_lda.fit(female_vectors)
    
    # Get top words for each topic in female documents
    female_topics = []
    for topic_idx, topic in enumerate(female_lda.components_):
        top_words_idx = topic.argsort()[:-10 - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        female_topics.append({
            'topic_id': topic_idx,
            'top_words': top_words
        })
    
    return male_topics, female_topics