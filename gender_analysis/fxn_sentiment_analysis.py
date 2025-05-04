def sentiment_analysis_by_gender(df):
    """Analyze sentiment differences between male and female IEPs."""
    from nltk.sentiment import SentimentIntensityAnalyzer
    
    # Initialize VADER sentiment analyzer
    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()
    
    # Add sentiment scores to dataframe
    df['compound_sentiment'] = df['full_text'].apply(
        lambda x: sid.polarity_scores(x)['compound'] if isinstance(x, str) else 0
    )
    df['positive_sentiment'] = df['full_text'].apply(
        lambda x: sid.polarity_scores(x)['pos'] if isinstance(x, str) else 0
    )
    df['negative_sentiment'] = df['full_text'].apply(
        lambda x: sid.polarity_scores(x)['neg'] if isinstance(x, str) else 0
    )
    
    # Compare sentiment by gender
    sentiment_by_gender = df.groupby('gender')[
        ['compound_sentiment', 'positive_sentiment', 'negative_sentiment']
    ].mean().reset_index()
    
    # Compare by disability and gender
    sentiment_by_disability_gender = df.groupby(['disability', 'gender'])[
        ['compound_sentiment', 'positive_sentiment', 'negative_sentiment']
    ].mean().reset_index()
    
    return sentiment_by_gender, sentiment_by_disability_gender