import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anthropic

# Initialize Claude client
client = anthropic.Anthropic(api_key="your_api_key")

# Set up paths
DATA_DIR = "iep_documents/"
OUTPUT_DIR = "analysis_results/"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Function to load IEP documents
def load_iep_documents(data_dir):
    """Load all IEP documents from the data directory."""
    documents = []
    
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(data_dir, file)
            
            # Extract metadata from filename
            # Assuming filename format: disability_grade_gender_race_religion.txt
            parts = file.replace(".txt", "").split("_")
            if len(parts) >= 3:
                disability = parts[0]
                grade = parts[1]
                gender = parts[2]
                race = parts[3] if len(parts) > 3 else "Unspecified"
                religion = parts[4] if len(parts) > 4 else "Unspecified"
            else:
                continue  # Skip if filename format doesn't match
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                documents.append({
                    'filename': file,
                    'disability': disability,
                    'grade': grade,
                    'gender': gender,
                    'race': race,
                    'religion': religion,
                    'content': content
                })
    
    return documents

# Function to analyze a document using Claude API
def analyze_document_with_claude(document):
    """Use Claude API to analyze gender-based language patterns in a document."""
    
    prompt = f"""
    Analyze the following IEP (Individualized Education Program) document for gender-based language patterns. 
    Focus on identifying language related to:
    1. Independence/strength themes (independence, strength, performance, task completion, problem-solving)
    2. Communication/emotion themes (communication, emotional expression, social interaction, support)
    
    Document metadata:
    Disability: {document['disability']}
    Grade: {document['grade']}
    Gender: {document['gender']}
    Race: {document['race']}
    Religion: {document['religion']}
    
    Document content:
    {document['content']}
    
    Please provide a detailed analysis with specific examples and word counts for each theme.
    Format your response as JSON with the following structure:
    {{
        "independence_strength": {{
            "word_count": <number>,
            "examples": [<list of examples>],
            "intensity": <score from 1-10>
        }},
        "communication_emotion": {{
            "word_count": <number>,
            "examples": [<list of examples>],
            "intensity": <score from 1-10>
        }}
    }}
    """
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        system="You are an expert in analyzing educational documents for language patterns. You provide detailed, objective analysis in the format requested.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extract JSON from response
    response_text = message.content[0].text
    try:
        # Find JSON in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            analysis = json.loads(json_str)
            return analysis
        else:
            print(f"No valid JSON found in response for {document['filename']}")
            return None
    except json.JSONDecodeError:
        print(f"Error parsing JSON for {document['filename']}")
        return None

# Function to analyze all documents and aggregate results
def analyze_gender_patterns_with_claude():
    """Use Claude API to analyze gender-based language patterns across all documents."""
    documents = load_iep_documents(DATA_DIR)
    print(f"Loaded {len(documents)} documents.")
    
    results = []
    
    for i, document in enumerate(documents):
        print(f"Analyzing document {i+1}/{len(documents)}: {document['filename']}")
        analysis = analyze_document_with_claude(document)
        
        if analysis:
            results.append({
                'filename': document['filename'],
                'disability': document['disability'],
                'grade': document['grade'],
                'gender': document['gender'],
                'race': document['race'],
                'religion': document['religion'],
                'independence_strength_count': analysis['independence_strength']['word_count'],
                'independence_strength_intensity': analysis['independence_strength']['intensity'],
                'communication_emotion_count': analysis['communication_emotion']['word_count'],
                'communication_emotion_intensity': analysis['communication_emotion']['intensity'],
                'examples_independence': analysis['independence_strength']['examples'],
                'examples_communication': analysis['communication_emotion']['examples']
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save raw results
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'claude_gender_analysis.csv'), index=False)
    
    # Create aggregated analysis by gender
    gender_analysis = results_df.groupby('gender').agg({
        'independence_strength_count': 'mean',
        'independence_strength_intensity': 'mean',
        'communication_emotion_count': 'mean',
        'communication_emotion_intensity': 'mean'
    }).reset_index()
    
    # Save gender aggregation
    gender_analysis.to_csv(os.path.join(OUTPUT_DIR, 'claude_gender_aggregation.csv'), index=False)
    
    # Create aggregation by disability and gender
    disability_gender_analysis = results_df.groupby(['disability', 'gender']).agg({
        'independence_strength_count': 'mean',
        'independence_strength_intensity': 'mean',
        'communication_emotion_count': 'mean',
        'communication_emotion_intensity': 'mean'
    }).reset_index()
    
    # Save disability-gender aggregation
    disability_gender_analysis.to_csv(os.path.join(OUTPUT_DIR, 'claude_disability_gender_analysis.csv'), index=False)
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    # Visualize results
    plt.figure(figsize=(12, 8))
    
    # Create bar chart for gender comparison
    gender_means = gender_analysis[gender_analysis['gender'].isin(['Male', 'Female'])]
    
    # Set up plot
    x = np.arange(2)  # Two gender categories
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    independence_bars = ax.bar(x - width/2, gender_means['independence_strength_intensity'], 
                              width, label='Independence/Strength')
    communication_bars = ax.bar(x + width/2, gender_means['communication_emotion_intensity'], 
                               width, label='Communication/Emotion')
    
    # Add labels and titles
    ax.set_xlabel('Gender')
    ax.set_ylabel('Average Intensity Score (1-10)')
    ax.set_title('Gender-Based Language Patterns in IEP Documents')
    ax.set_xticks(x)
    ax.set_xticklabels(gender_means['gender'])
    ax.legend()
    
    # Add value labels on the bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    add_labels(independence_bars)
    add_labels(communication_bars)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'claude_gender_comparison.png'))
    
    # Create heatmap for disability + gender analysis
    plt.figure(figsize=(14, 10))
    
    # Create pivot table for independence intensity
    independence_pivot = disability_gender_analysis.pivot(
        index='disability', 
        columns='gender', 
        values='independence_strength_intensity'
    )
    
    # Generate heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(independence_pivot, annot=True, cmap='Blues', fmt='.1f')
    plt.title('Independence/Strength Language Intensity by Disability and Gender')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'claude_independence_by_disability_gender.png'))
    
    # Create heatmap for communication intensity
    plt.figure(figsize=(14, 10))
    
    # Create pivot table for communication intensity
    communication_pivot = disability_gender_analysis.pivot(
        index='disability', 
        columns='gender', 
        values='communication_emotion_intensity'
    )
    
    # Generate heatmap
    sns.heatmap(communication_pivot, annot=True, cmap='Reds', fmt='.1f')
    plt.title('Communication/Emotion Language Intensity by Disability and Gender')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'claude_communication_by_disability_gender.png'))
    
    # Create ratio heatmap (independence to communication)
    plt.figure(figsize=(14, 10))
    
    # Calculate ratio
    disability_gender_analysis['ratio'] = (
        disability_gender_analysis['independence_strength_intensity'] / 
        disability_gender_analysis['communication_emotion_intensity']
    )
    
    # Create pivot table for ratio
    ratio_pivot = disability_gender_analysis.pivot(
        index='disability',
        columns='gender',
        values='ratio'
    )
    
    # Generate heatmap
    sns.heatmap(ratio_pivot, annot=True, cmap='RdBu_r', center=1, fmt='.2f')
    plt.title('Ratio of Independence to Communication Language by Disability and Gender')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'claude_language_ratio_by_disability_gender.png'))
    
    return results_df, gender_analysis, disability_gender_analysis

# Function to perform deeper qualitative analysis with Claude
def qualitative_analysis_with_claude(sample_documents):
    """Use Claude API to perform deeper qualitative analysis on a sample of documents."""
    
    # Select a diverse sample of documents if not provided
    if not sample_documents:
        documents = load_iep_documents(DATA_DIR)
        
        # Create stratified sample by disability and gender
        sampled_docs = []
        for disability in set(doc['disability'] for doc in documents):
            for gender in ['Male', 'Female']:
                matching_docs = [doc for doc in documents 
                                if doc['disability'] == disability and doc['gender'] == gender]
                if matching_docs:
                    sampled_docs.append(matching_docs[0])  # Take first matching document
        
        sample_documents = sampled_docs[:10]  # Limit to 10 documents for analysis
    
    # Prompt for qualitative analysis
    qualitative_prompt = """
    Perform a detailed qualitative analysis of the following IEP documents, focusing on gender-based language patterns.
    
    For each document, identify:
    1. Specific phrases that reflect gender stereotypes or biases
    2. Differences in how goals, needs, and accommodations are framed
    3. Any patterns in language tone, emphasis, or expectations
    
    After analyzing each document, provide an overall assessment of patterns across documents.
    
    Documents:
    {documents}
    
    Format your response as a detailed qualitative analysis with specific examples and observations.
    """
    
    # Prepare document text for prompt
    doc_texts = []
    for i, doc in enumerate(sample_documents):
        doc_text = f"Document {i+1}:\nDisability: {doc['disability']}\nGender: {doc['gender']}\n\nContent:\n{doc['content'][:2000]}...\n\n"
        doc_texts.append(doc_text)
    
    # Split analysis if needed due to context limits
    results = []
    batch_size = 2  # Analyze 2 documents at a time
    
    for i in range(0, len(sample_documents), batch_size):
        batch_docs = doc_texts[i:i+batch_size]
        batch_prompt = qualitative_prompt.format(documents="\n".join(batch_docs))
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4000,
            temperature=0,
            system="You are an expert in educational psychology and linguistic analysis. You provide detailed, nuanced analysis of language patterns in educational documents.",
            messages=[
                {"role": "user", "content": batch_prompt}
            ]
        )
        
        results.append(message.content[0].text)
    
    # Save qualitative analysis
    with open(os.path.join(OUTPUT_DIR, 'claude_qualitative_analysis.txt'), 'w') as f:
        for i, result in enumerate(results):
            f.write(f"=== BATCH {i+1} ANALYSIS ===\n\n")
            f.write(result)
            f.write("\n\n")
    
    # Ask Claude to synthesize findings
    synthesis_prompt = """
    Based on the individual document analyses below, synthesize the key findings about gender-based language patterns in IEP documents.
    
    Focus on:
    1. Consistent patterns across different disability categories
    2. Specific examples of gendered language
    3. Implications for educational equity
    4. Recommendations for improving the IEP process
    
    Previous analyses:
    {analyses}
    
    Format your response as a comprehensive research report with sections for each focus area.
    """
    
    synthesis_message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4000,
        temperature=0,
        system="You are an expert educational researcher specializing in bias and equity in special education. Provide an evidence-based synthesis of findings.",
        messages=[
            {"role": "user", "content": synthesis_prompt.format(analyses="\n".join(results))}
        ]
    )
    
    # Save synthesis
    with open(os.path.join(OUTPUT_DIR, 'claude_gender_patterns_synthesis.txt'), 'w') as f:
        f.write(synthesis_message.content[0].text)
    
    return synthesis_message.content[0].text

# Main function
def main():
    """Run the complete analysis pipeline using Claude API."""
    print("Starting analysis...")
    
    # Quantitative analysis
    print("Performing quantitative analysis with Claude...")
    results_df, gender_analysis, disability_gender_analysis = analyze_gender_patterns_with_claude()
    print("Completed quantitative analysis.")
    
    # Qualitative analysis
    print("Performing qualitative analysis with Claude...")
    synthesis = qualitative_analysis_with_claude(None)  # Auto-select sample
    print("Completed qualitative analysis.")
    
    print("Analysis complete! Results saved to:", OUTPUT_DIR)
    
    # Print key findings
    print("\nKEY FINDINGS:")
    print("-" * 50)
    
    # Gender comparison
    male_independence = gender_analysis[gender_analysis['gender'] == 'Male']['independence_strength_intensity'].values[0]
    female_independence = gender_analysis[gender_analysis['gender'] == 'Female']['independence_strength_intensity'].values[0]
    
    male_communication = gender_analysis[gender_analysis['gender'] == 'Male']['communication_emotion_intensity'].values[0]
    female_communication = gender_analysis[gender_analysis['gender'] == 'Female']['communication_emotion_intensity'].values[0]
    
    print(f"Male independence/strength score: {male_independence:.2f}")
    print(f"Female independence/strength score: {female_independence:.2f}")
    print(f"Difference: {male_independence - female_independence:.2f}")
    print()
    print(f"Male communication/emotion score: {male_communication:.2f}")
    print(f"Female communication/emotion score: {female_communication:.2f}")
    print(f"Difference: {female_communication - male_communication:.2f}")
    
    # Print consistency across disabilities
    print("\nConsistency of gender patterns across disabilities:")
    for disability in disability_gender_analysis['disability'].unique():
        disability_data = disability_gender_analysis[disability_gender_analysis['disability'] == disability]
        
        if len(disability_data) >= 2:  # Both male and female data available
            male_data = disability_data[disability_data['gender'] == 'Male']
            female_data = disability_data[disability_data['gender'] == 'Female']
            
            if not male_data.empty and not female_data.empty:
                male_ratio = male_data['independence_strength_intensity'].values[0] / male_data['communication_emotion_intensity'].values[0]
                female_ratio = female_data['independence_strength_intensity'].values[0] / female_data['communication_emotion_intensity'].values[0]
                
                print(f"{disability}: Male I/C ratio: {male_ratio:.2f}, Female I/C ratio: {female_ratio:.2f}")

if __name__ == "__main__":
    main()