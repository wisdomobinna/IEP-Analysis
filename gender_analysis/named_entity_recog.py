def named_entity_analysis(df):
    """Analyze named entities in IEP text by gender."""
    import spacy
    
    # Load SpaCy model
    nlp = spacy.load('en_core_web_lg')
    
    # Define entity categories of interest
    categories = {
        'SKILL': ['independence', 'reading', 'writing', 'communication', 'social'],
        'SUPPORT': ['accommodation', 'modification', 'assistance', 'aid', 'help'],
        'EMOTIONAL': ['confidence', 'anxiety', 'stress', 'calm', 'frustration']
    }
    
    # Process documents by gender
    results = {'Male': {}, 'Female': {}}
    
    for gender in ['Male', 'Female']:
        gender_docs = df[df['gender'] == gender]['full_text'].fillna('').tolist()
        
        # Initialize category counts
        for category in categories:
            results[gender][category] = 0
        
        # Process each document
        for doc_text in gender_docs:
            # Process with spaCy
            doc = nlp(doc_text)
            
            # Count category mentions
            for category, terms in categories.items():
                for term in terms:
                    results[gender][category] += doc_text.lower().count(term)
        
        # Normalize by number of documents
        for category in categories:
            results[gender][category] /= len(gender_docs) if gender_docs else 1
    
    return results