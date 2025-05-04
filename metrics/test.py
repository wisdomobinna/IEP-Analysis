#!/usr/bin/env python3
# example_usage.py
# Example script demonstrating the IEP Success Metrics Analyzer

from iep_success_metrics_analyzer import IEPSuccessMetricsAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Set up configuration
CONFIG = {
    "documents_folder": "sample_ieps",
    "output_folder": "analysis_results",
    "api_key": "YOUR_ANTHROPIC_API_KEY"  # Replace with your actual API key
}

# Initialize the analyzer
analyzer = IEPSuccessMetricsAnalyzer(
    documents_folder=CONFIG["documents_folder"],
    output_folder=CONFIG["output_folder"],
    api_key=CONFIG["api_key"]
)

# Run the full analysis
print("Running full analysis...")
results = analyzer.run_analysis()

# Access specific results
success_dist = results["success_distribution"]
detailed_metrics = results["detailed_metrics"]

# Create additional custom visualizations
print("Creating custom visualizations...")

# 1. Compare 80% pattern usage across disability categories (bar chart)
plt.figure(figsize=(12, 6))
ax = detailed_metrics["percentage_with_80_pattern"].sort_values(ascending=False).plot(
    kind='bar', 
    color='skyblue',
    edgecolor='black'
)
plt.axhline(y=detailed_metrics["percentage_with_80_pattern"].mean(), 
           color='red', 
           linestyle='--',
           label=f'Average: {detailed_metrics["percentage_with_80_pattern"].mean():.1f}%')
plt.title('Percentage of IEPs Using 80% Success Metric by Disability Category', fontsize=14)
plt.xlabel('Disability Category')
plt.ylabel('Percentage of IEPs')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["output_folder"], "80_percent_usage_by_category.png"), dpi=300)
plt.close()

# 2. Create a heatmap of success pattern categories by disability type
contingency_df = detailed_metrics["contingency_table"]
plt.figure(figsize=(12, 8))
sns.heatmap(contingency_df, annot=True, cmap="YlGnBu", fmt="d", cbar_kws={'label': 'Count'})
plt.title('Success Pattern Categories by Disability Type', fontsize=14)
plt.ylabel('Disability Type')
plt.xlabel('Success Pattern Category')
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["output_folder"], "success_pattern_heatmap.png"), dpi=300)
plt.close()

# 3. Analyze average success metrics by grade level
grade_metrics = results["data"].groupby('grade_level')['most_common_metric'].mean().dropna()
plt.figure(figsize=(10, 6))
grade_metrics.plot(kind='line', marker='o', color='green', linewidth=2)
plt.title('Average Success Metric by Grade Level', fontsize=14)
plt.xlabel('Grade Level')
plt.ylabel('Average Success Metric (%)')
plt.grid(True, alpha=0.3)
plt.xticks(grade_metrics.index)
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["output_folder"], "success_metrics_by_grade.png"), dpi=300)
plt.close()

# 4. Create a comparison chart for success metrics by disability and demographic
if 'gender' in results["demographic_results"]:
    # Prepare data for grouped bar chart
    gender_data = results["demographic_results"]["gender"]
    disability_data = success_dist
    
    # Focus on the "Most Common Success Rate (80%)" column
    gender_80pct = gender_data["Most Common Success Rate (80%)"].to_dict()
    disability_80pct = disability_data["Most Common Success Rate (80%)"].to_dict()
    
    # Combine into one dataframe
    comparison_data = pd.DataFrame({
        'Category': list(gender_80pct.keys()) + list(disability_80pct.keys()),
        'Value': list(gender_80pct.values()) + list(disability_80pct.values()),
        'Type': ['Gender'] * len(gender_80pct) + ['Disability'] * len(disability_80pct)
    })
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Category', y='Value', hue='Type', data=comparison_data)
    plt.title('80% Success Rate Usage by Gender and Disability', fontsize=14)
    plt.xlabel('Category')
    plt.ylabel('Percentage Using 80% Success Rate')
    plt.legend(title='Category Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_folder"], "80pct_gender_disability_comparison.png"), dpi=300)
    plt.close()

print(f"Analysis complete. Results saved to: {CONFIG['output_folder']}")
print(f"See the full report at: {os.path.join(CONFIG['output_folder'], 'success_metrics_report.md')}")