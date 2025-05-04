# IEP Success Rate Analysis
# This script analyzes IEP documents to determine success rates across disability categories

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import nltk
from nltk.tokenize import sent_tokenize
import glob
from tqdm import tqdm

# Ensure necessary NLTK data is downloaded
nltk.download('punkt', quiet=True)

# Define paths
DATA_FOLDER = "iep_documents/"  # Folder containing all IEP documents
OUTPUT_FOLDER = "results/"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Define disability categories and their keywords for classification
DISABILITY_CATEGORIES = {
    "Physical Disabilities": [
        "orthopedic impairment", "cerebral palsy", "spina bifida", 
        "muscular dystrophy", "quadriplegia", "paraplegia", 
        "limb deficiency", "physical disability"
    ],
    "Cognitive/Developmental": [
        "autism spectrum", "sensory processing", "high-functioning autism", 
        "nonverbal communication", "developmental disability", 
        "intellectual disability", "cognitive impairment"
    ],
    "Learning Disabilities": [
        "dyslexia", "learning disability", "specific learning disorder", 
        "reading disability", "writing disability", "dyscalculia"
    ],
    "Neurological/Processing": [
        "neurological processing", "adhd", "attention deficit", 
        "executive function", "tourette", "complex neurological", 
        "processing disorder"
    ],
    "Mental Health": [
        "anxiety disorder", "emotional disturbance", "depression", 
        "mental health", "emotional disability", "psychological"
    ]
}

# Define demographic identifiers
GENDER_IDENTIFIERS = {
    "Male": ["male", "boy", "he", "his", "him"],
    "Female": ["female", "girl", "she", "her", "hers"]
}

RACE_IDENTIFIERS = {
    "White": ["white"],
    "Black": ["black", "african american"],
    "Asian": ["asian"]
}

RELIGION_IDENTIFIERS = {
    "Christian": ["christian"],
    "Muslim": ["muslim"],
    "Jewish": ["jewish"],
    "Atheist": ["atheist"]
}

def extract_files_from_folder(folder_path):
    """Extract all text files from a folder"""
    files = glob.glob(os.path.join(folder_path, "*.txt"))
    return files

def extract_demographic_info(text):
    """Extract demographic information from text"""
    demographics = {
        "gender": "Not Specified",
        "race": "Not Specified",
        "religion": "Not Specified"
    }
    
    text_lower = text.lower()
    
    # Extract gender
    for gender, keywords in GENDER_IDENTIFIERS.items():
        if any(keyword in text_lower for keyword in keywords):
            demographics["gender"] = gender
            break
    
    # Extract race
    for race, keywords in RACE_IDENTIFIERS.items():
        if any(keyword in text_lower for keyword in keywords):
            demographics["race"] = race
            break
    
    # Extract religion
    for religion, keywords in RELIGION_IDENTIFIERS.items():
        if any(keyword in text_lower for keyword in keywords):
            demographics["religion"] = religion
            break
    
    return demographics

def extract_disability_type(text):
    """Extract the primary disability type from text"""
    text_lower = text.lower()
    
    for category, keywords in DISABILITY_CATEGORIES.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    
    return "Other"

def extract_grade_level(text):
    """Extract grade level from text"""
    grade_pattern = r'(\d+)(?:st|nd|rd|th)? grade'
    age_pattern = r'(\d+)[-\s]year[-\s]old'
    
    grade_match = re.search(grade_pattern, text.lower())
    if grade_match:
        return int(grade_match.group(1))
    
    age_match = re.search(age_pattern, text.lower())
    if age_match:
        age = int(age_match.group(1))
        # Approximate grade level based on age
        grade = age - 5
        return max(1, min(12, grade))  # Ensure grade is between 1-12
    
    return None

def extract_sections(text):
    """Extract the main sections from IEP text"""
    sections = {}
    
    # Define section patterns
    section_patterns = {
        "present_levels": r'Present Levels of Performance(.*?)(?:Student Needs|Goals and Objectives|Accommodations|$)',
        "student_needs": r'Student Needs and Impact of Disability(.*?)(?:Goals and Objectives|Accommodations|$)',
        "goals": r'Goals and Objectives(.*?)(?:Accommodations|$)',
        "accommodations": r'Accommodations and Modifications(.*?)(?:$)'
    }
    
    # Extract each section using regex
    for section_name, pattern in section_patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[section_name] = match.group(1).strip()
        else:
            sections[section_name] = ""
    
    return sections

def extract_goals_and_objectives(goals_section):
    """Extract individual goals and objectives from the goals section"""
    goals = []
    
    # Split by "Measurable Goal:" or similar patterns
    goal_pattern = r'(?:Measurable Goal:|Goal:|Objective:)(.*?)(?:Measurable Goal:|Goal:|Objective:|$)'
    goal_matches = re.finditer(goal_pattern, goals_section, re.DOTALL | re.IGNORECASE)
    
    for match in goal_matches:
        goal_text = match.group(1).strip()
        if goal_text:
            goals.append(goal_text)
    
    return goals

def analyze_goal_success_metrics(goals):
    """Analyze goals for success metrics and criteria"""
    success_metrics = []
    
    for goal in goals:
        # Look for common success metric patterns
        percentage_match = re.search(r'(\d{1,3})%', goal)
        ratio_match = re.search(r'(\d+)\s*(?:out of|\/)\s*(\d+)', goal)
        accuracy_match = re.search(r'(\d{1,3})%\s*accuracy', goal)
        trials_match = re.search(r'(\d+)\s*(?:consecutive)?\s*trials', goal)
        
        if percentage_match:
            success_metrics.append(int(percentage_match.group(1)))
        elif ratio_match:
            numerator = int(ratio_match.group(1))
            denominator = int(ratio_match.group(2))
            success_metrics.append(round(numerator / denominator * 100))
        elif accuracy_match:
            success_metrics.append(int(accuracy_match.group(1)))
        elif trials_match:
            # Rough approximation if only trials are mentioned
            success_metrics.append(80)  # Default to 80% as this is the common pattern we observed
    
    return success_metrics

def classify_success_rate(metrics):
    """Classify success rate into categories"""
    if not metrics:
        return None, None
    
    avg_metric = sum(metrics) / len(metrics)
    
    # Classify into categories
    if 75 <= avg_metric <= 85:
        rate_category = "Most Common Success Rate (80%)"
        rate_value = 80
    elif avg_metric < 75:
        rate_category = "Below Standard (<80%)"
        rate_value = avg_metric
    else:
        rate_category = "Above Standard (>80%)"
        rate_value = avg_metric
    
    return rate_category, rate_value

def analyze_iep_document(file_path):
    """Analyze a single IEP document"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract basic information
        demographics = extract_demographic_info(content)
        disability_type = extract_disability_type(content)
        grade_level = extract_grade_level(content)
        
        # Extract sections
        sections = extract_sections(content)
        
        # Extract goals and objectives
        goals = extract_goals_and_objectives(sections.get("goals", ""))
        
        # Analyze success metrics
        success_metrics = analyze_goal_success_metrics(goals)
        success_category, success_value = classify_success_rate(success_metrics)
        
        return {
            "file_path": file_path,
            "disability_type": disability_type,
            "grade_level": grade_level,
            "gender": demographics["gender"],
            "race": demographics["race"],
            "religion": demographics["religion"],
            "success_metrics": success_metrics,
            "success_category": success_category,
            "success_value": success_value,
            "goals_count": len(goals)
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def analyze_all_documents(folder_path):
    """Analyze all IEP documents in a folder"""
    files = extract_files_from_folder(folder_path)
    results = []
    
    print(f"Analyzing {len(files)} IEP documents...")
    
    for file_path in tqdm(files):
        result = analyze_iep_document(file_path)
        if result:
            results.append(result)
    
    return pd.DataFrame(results)

def calculate_success_rate_distribution(df):
    """Calculate success rate distribution across disability categories"""
    # Ensure we only include documents with valid success categories
    valid_df = df[df['success_category'].notna()]
    
    # Group by disability type and success category
    success_dist = valid_df.groupby(['disability_type', 'success_category']).size().unstack(fill_value=0)
    
    # Calculate percentages
    success_dist_percent = success_dist.div(success_dist.sum(axis=1), axis=0) * 100
    
    # Ensure all categories are present
    for category in ["Most Common Success Rate (80%)", "Below Standard (<80%)", "Above Standard (>80%)"]:
        if category not in success_dist_percent.columns:
            success_dist_percent[category] = 0
    
    success_dist_percent = success_dist_percent.round(0).astype(int)
    
    return success_dist_percent

def plot_success_rate_table(success_dist_percent):
    """Create a table visualization of success rate distribution"""
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    # Rename columns for clarity
    success_dist_percent = success_dist_percent.rename(columns={
        "Most Common Success Rate (80%)": "Most Common\nSuccess\nRate (80%)",
        "Below Standard (<80%)": "Below\nStandard\n(<80%)",
        "Above Standard (>80%)": "Above\nStandard\n(>80%)"
    })
    
    # Add "Other Success Rates" column
    if "Other Success Rates" not in success_dist_percent.columns:
        other_rates = 100 - success_dist_percent.sum(axis=1)
        success_dist_percent["Other\nSuccess\nRates"] = other_rates
    
    # Define the column order
    column_order = ["Most Common\nSuccess\nRate (80%)", "Other\nSuccess\nRates", 
                    "Below\nStandard\n(<80%)", "Above\nStandard\n(>80%)"]
    
    # Reorder columns
    success_dist_percent = success_dist_percent[column_order]
    
    # Add percentage sign to values
    success_dist_percent_formatted = success_dist_percent.applymap(lambda x: f"{x}%")
    
    # Create color mapping for values
    cell_colors = np.zeros_like(success_dist_percent.values, dtype=object)
    
    # Set background colors based on values
    for i in range(success_dist_percent.shape[0]):
        for j in range(success_dist_percent.shape[1]):
            val = success_dist_percent.iloc[i, j]
            if j == 0:  # Most Common Success Rate
                cell_colors[i, j] = "#B3C6E7"  # Light blue
            elif j == 1:  # Other Success Rates
                cell_colors[i, j] = "#D9D9D9"  # Light gray
            elif j == 2:  # Below Standard
                cell_colors[i, j] = "#E78587" if val > 5 else "#F2D7D8"  # Red or light red
            elif j == 3:  # Above Standard
                cell_colors[i, j] = "#7DC77D" if val > 8 else "#D5EBD5"  # Green or light green
    
    # Create the table
    table = plt.table(
        cellText=success_dist_percent_formatted.values,
        rowLabels=success_dist_percent.index,
        colLabels=success_dist_percent.columns,
        cellColours=cell_colors,
        cellLoc='center',
        loc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Add title
    plt.suptitle("IEP Success Rates by Disability Category", fontsize=16, y=0.95)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc="#B3C6E7", label="Most Common Success Rate (80%)"),
        plt.Rectangle((0, 0), 1, 1, fc="#E78587", label="Below Standard (<80%)"),
        plt.Rectangle((0, 0), 1, 1, fc="#7DC77D", label="Above Standard (>80%)")
    ]
    
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02),
              ncol=3, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "success_rate_table.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_success_rate_bars(success_dist_percent):
    """Create a bar chart visualization of success rates"""
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    categories = success_dist_percent.index
    x = np.arange(len(categories))
    width = 0.2
    
    # Define data for each success rate category
    common_rate = success_dist_percent["Most Common Success Rate (80%)"].values
    below_rate = success_dist_percent["Below Standard (<80%)"].values if "Below Standard (<80%)" in success_dist_percent.columns else np.zeros(len(categories))
    above_rate = success_dist_percent["Above Standard (>80%)"].values if "Above Standard (>80%)" in success_dist_percent.columns else np.zeros(len(categories))
    
    # Plot bars
    plt.bar(x - width, common_rate, width, label='80% Success Rate', color='navy')
    plt.bar(x, below_rate, width, label='75% Success Rate', color='forestgreen')
    plt.bar(x + width, above_rate, width, label='90% Success Rate', color='darkred')
    
    # Add labels and legend
    plt.xlabel('Disability Categories')
    plt.ylabel('Percentage of IEPs')
    plt.title('Success Rate Distribution by Disability Category')
    plt.xticks(x, categories)
    plt.ylim(0, 100)
    plt.legend()
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "success_rate_bars.png"), dpi=300)
    plt.close()

def analyze_success_rate_by_demographic(df):
    """Analyze success rates across demographic variables"""
    demographic_vars = ['gender', 'race', 'religion']
    results = {}
    
    for var in demographic_vars:
        # Skip if all values are "Not Specified"
        if all(df[var] == "Not Specified"):
            continue
        
        # Calculate success rate distribution by demographic
        success_by_demo = df.groupby([var, 'success_category']).size().unstack(fill_value=0)
        success_by_demo_percent = success_by_demo.div(success_by_demo.sum(axis=1), axis=0) * 100
        success_by_demo_percent = success_by_demo_percent.round(0).astype(int)
        
        results[var] = success_by_demo_percent
    
    return results

def calculate_consistency_scores(df):
    """Calculate how consistent the 80% success rate is across categories"""
    # Group by disability type
    disability_groups = df.groupby('disability_type')
    
    consistency_scores = {}
    
    for disability, group in disability_groups:
        # Calculate percentage of IEPs in this disability with 80% success rate
        valid_group = group[group['success_category'].notna()]
        if len(valid_group) == 0:
            continue
            
        count_80_percent = len(valid_group[valid_group['success_category'] == "Most Common Success Rate (80%)"])
        percentage_80 = (count_80_percent / len(valid_group)) * 100
        
        consistency_scores[disability] = percentage_80
    
    # Calculate overall consistency score
    valid_df = df[df['success_category'].notna()]
    count_80_percent_overall = len(valid_df[valid_df['success_category'] == "Most Common Success Rate (80%)"])
    overall_consistency = (count_80_percent_overall / len(valid_df)) * 100 if len(valid_df) > 0 else 0
    
    consistency_scores['Overall'] = overall_consistency
    
    return consistency_scores

def analyze_success_metrics_patterns(df):
    """Analyze patterns in the specific success metrics used"""
    # Extract individual metrics from the lists in success_metrics column
    all_metrics = []
    for metrics in df['success_metrics'].dropna():
        all_metrics.extend(metrics)
    
    # Calculate frequency of each metric value
    metric_counts = pd.Series(all_metrics).value_counts().sort_index()
    
    # Plot histogram of metric values
    plt.figure(figsize=(10, 6))
    sns.histplot(all_metrics, bins=20, kde=True)
    plt.xlabel('Success Metric Value (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of IEP Success Metrics')
    plt.grid(True, alpha=0.3)
    
    # Add vertical line at 80%
    plt.axvline(x=80, color='r', linestyle='--', label='80% Benchmark')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "success_metrics_distribution.png"), dpi=300)
    plt.close()
    
    return metric_counts

def main():
    # Analyze all IEP documents
    df = analyze_all_documents(DATA_FOLDER)
    
    # Save raw data
    df.to_csv(os.path.join(OUTPUT_FOLDER, "iep_analysis_results.csv"), index=False)
    
    # Calculate success rate distribution
    success_dist_percent = calculate_success_rate_distribution(df)
    
    # Create visualizations
    plot_success_rate_table(success_dist_percent)
    plot_success_rate_bars(success_dist_percent)
    
    # Analyze success rates by demographic
    demographic_results = analyze_success_rate_by_demographic(df)
    
    # Calculate consistency scores
    consistency_scores = calculate_consistency_scores(df)
    
    # Analyze success metrics patterns
    metric_patterns = analyze_success_metrics_patterns(df)
    
    # Print key findings
    print("\n===== KEY FINDINGS =====")
    print(f"Overall consistency of 80% success rate: {consistency_scores['Overall']:.1f}%")
    print("\nSuccess rate distribution by disability category:")
    print(success_dist_percent)
    
    print("\nConsistency scores by disability category:")
    for disability, score in consistency_scores.items():
        if disability != 'Overall':
            print(f"{disability}: {score:.1f}%")
    
    print("\nMost common success metrics:")
    print(metric_patterns.head(10))
    
    print("\nAnalysis complete. Results saved to the 'results' folder.")

if __name__ == "__main__":
    main()