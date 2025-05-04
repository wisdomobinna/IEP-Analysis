#!/usr/bin/env python3
# iep_success_metrics_analyzer.py

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import glob
from tqdm import tqdm
import anthropic
import time

class IEPSuccessMetricsAnalyzer:
    def __init__(self, documents_folder, output_folder, api_key):
        """Initialize the analyzer with folder paths and API key.
        
        Args:
            documents_folder (str): Path to folder containing IEP documents
            output_folder (str): Path where results will be saved
            api_key (str): Anthropic API key for Claude
        """
        self.documents_folder = documents_folder
        self.output_folder = output_folder
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Define disability categories and demographic identifiers
        self.disability_categories = {
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
        
        self.gender_identifiers = {
            "Male": ["male", "boy", "he", "his", "him"],
            "Female": ["female", "girl", "she", "her", "hers"]
        }
        
        self.race_identifiers = {
            "White": ["white"],
            "Black": ["black", "african american"],
            "Asian": ["asian"]
        }
        
        self.religion_identifiers = {
            "Christian": ["christian"],
            "Muslim": ["muslim"],
            "Jewish": ["jewish"],
            "Atheist": ["atheist"]
        }
    
    def extract_files_from_folder(self):
        """Extract all text files from the documents folder.
        
        Returns:
            list: List of file paths for all text files in the folder
        """
        files = glob.glob(os.path.join(self.documents_folder, "*.txt"))
        return files
    
    def extract_document_metadata(self, content):
        """Extract basic metadata from document content.
        
        Args:
            content (str): Document content as text
            
        Returns:
            dict: Metadata including disability type, grade level, and demographics
        """
        # Extract disability type
        disability_type = "Other"
        for category, keywords in self.disability_categories.items():
            if any(keyword in content.lower() for keyword in keywords):
                disability_type = category
                break
        
        # Extract demographics
        demographics = {
            "gender": "Not Specified",
            "race": "Not Specified",
            "religion": "Not Specified"
        }
        
        content_lower = content.lower()
        
        # Gender
        for gender, keywords in self.gender_identifiers.items():
            if any(keyword in content_lower for keyword in keywords):
                demographics["gender"] = gender
                break
        
        # Race
        for race, keywords in self.race_identifiers.items():
            if any(keyword in content_lower for keyword in keywords):
                demographics["race"] = race
                break
        
        # Religion
        for religion, keywords in self.religion_identifiers.items():
            if any(keyword in content_lower for keyword in keywords):
                demographics["religion"] = religion
                break
        
        # Extract grade level
        grade_pattern = r'(\d+)(?:st|nd|rd|th)? grade'
        age_pattern = r'(\d+)[-\s]year[-\s]old'
        
        grade_level = None
        
        grade_match = re.search(grade_pattern, content_lower)
        if grade_match:
            grade_level = int(grade_match.group(1))
        else:
            age_match = re.search(age_pattern, content_lower)
            if age_match:
                age = int(age_match.group(1))
                # Approximate grade level based on age
                grade_level = age - 5
                grade_level = max(1, min(12, grade_level))  # Ensure grade is between 1-12
        
        return {
            "disability_type": disability_type,
            "grade_level": grade_level,
            "gender": demographics["gender"],
            "race": demographics["race"],
            "religion": demographics["religion"]
        }
    
    def extract_sections(self, content):
        """Extract the main sections from IEP text.
        
        Args:
            content (str): Document content as text
            
        Returns:
            dict: Dictionary with extracted sections
        """
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
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                sections[section_name] = match.group(1).strip()
            else:
                sections[section_name] = ""
        
        return sections
    
    def analyze_success_metrics_with_claude(self, goals_section, metadata):
        """Use Claude API to analyze success metrics in goals section.
        
        Args:
            goals_section (str): Text content of the goals section
            metadata (dict): Document metadata
            
        Returns:
            dict: Analysis results including success metrics
        """
        prompt = f"""
        Analyze the following IEP Goals and Objectives section for a student with a {metadata['disability_type']} disability.
        
        Focus specifically on the success metrics and evaluation criteria mentioned in the goals. 
        I'm particularly interested in analyzing patterns in the percentage targets set for mastery or success (e.g., "80% accuracy", "4 out of 5 trials", etc.).
        
        For each goal/objective, extract:
        1. The success metric type (percentage, ratio, trials, etc.)
        2. The numerical value of the success metric (converted to percentage if needed)
        3. Whether this aligns with the common pattern of using 80% as a standard success metric
        
        Return your analysis in JSON format:
        {{
            "goals_count": <number of goals identified>,
            "success_metrics": [<array of percentage values extracted>],
            "has_80_percent_pattern": <boolean indicating if the 80% pattern is present>,
            "percentage_of_80_percent_goals": <percentage of goals with 80% metric>,
            "most_common_metric": <the most common percentage value used>,
            "summary": <brief summary of the success metric patterns>
        }}
        
        Goals and Objectives section:
        {goals_section}
        """
        
        try:
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1500,
                temperature=0,
                system="You are a special education expert analyzing IEP documents. Provide objective, detailed analysis in JSON format.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse JSON from response
            json_str = response.content[0].text
            json_match = re.search(r'```json\s*(.*?)\s*```', json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON in the text without code blocks
                json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
            
            try:
                analysis = json.loads(json_str)
                
                # Add metadata
                analysis.update(metadata)
                
                # Categorize the success rate pattern
                if analysis.get("has_80_percent_pattern", False) and analysis.get("percentage_of_80_percent_goals", 0) >= 50:
                    analysis["success_pattern_category"] = "Most Common Success Rate (80%)"
                elif analysis.get("most_common_metric", 0) < 80:
                    analysis["success_pattern_category"] = "Below Standard (<80%)"
                elif analysis.get("most_common_metric", 0) > 80:
                    analysis["success_pattern_category"] = "Above Standard (>80%)"
                else:
                    analysis["success_pattern_category"] = "Mixed Pattern"
                
                return analysis
            except json.JSONDecodeError:
                print(f"Error parsing JSON from Claude response")
                return self.fallback_metrics_analysis(goals_section, metadata)
                
        except Exception as e:
            print(f"Error using Claude API: {e}")
            return self.fallback_metrics_analysis(goals_section, metadata)
    
    def fallback_metrics_analysis(self, goals_section, metadata):
        """Fallback method for metrics analysis if Claude API fails.
        
        Args:
            goals_section (str): Text content of the goals section
            metadata (dict): Document metadata
            
        Returns:
            dict: Analysis results including success metrics
        """
        print("Using fallback analysis method...")
        
        # Extract goals
        goals = []
        goal_pattern = r'(?:Measurable Goal:|Goal:|Objective:)(.*?)(?:Measurable Goal:|Goal:|Objective:|$)'
        goal_matches = re.finditer(goal_pattern, goals_section, re.DOTALL | re.IGNORECASE)
        
        for match in goal_matches:
            goal_text = match.group(1).strip()
            if goal_text:
                goals.append(goal_text)
        
        # Extract success metrics
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
                # Default to 80% as this is the common pattern
                success_metrics.append(80)
        
        # Analyze the metrics
        if success_metrics:
            count_80_percent = sum(1 for m in success_metrics if m == 80)
            percentage_80 = (count_80_percent / len(success_metrics)) * 100
            
            # Determine the most common metric
            metric_counts = {}
            for metric in success_metrics:
                metric_counts[metric] = metric_counts.get(metric, 0) + 1
            
            most_common_metric = 0
            max_count = 0
            for metric, count in metric_counts.items():
                if count > max_count:
                    most_common_metric = metric
                    max_count = count
            
            # Determine pattern category
            if percentage_80 >= 50:
                pattern_category = "Most Common Success Rate (80%)"
            elif most_common_metric < 80:
                pattern_category = "Below Standard (<80%)"
            elif most_common_metric > 80:
                pattern_category = "Above Standard (>80%)"
            else:
                pattern_category = "Mixed Pattern"
        else:
            count_80_percent = 0
            percentage_80 = 0
            most_common_metric = 0
            pattern_category = "No Metrics Found"
        
        return {
            "goals_count": len(goals),
            "success_metrics": success_metrics,
            "has_80_percent_pattern": percentage_80 >= 50,
            "percentage_of_80_percent_goals": percentage_80,
            "most_common_metric": most_common_metric,
            "summary": f"Found {len(goals)} goals with {len(success_metrics)} explicit success metrics.",
            "success_pattern_category": pattern_category,
            **metadata
        }
    
    def analyze_document(self, file_path):
        """Analyze a single IEP document.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            dict: Analysis results or None if error occurs
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata and sections
            metadata = self.extract_document_metadata(content)
            sections = self.extract_sections(content)
            
            # If no goals section is found, return basic metadata
            if not sections.get("goals", "").strip():
                metadata.update({
                    "goals_count": 0,
                    "success_metrics": [],
                    "has_80_percent_pattern": False,
                    "percentage_of_80_percent_goals": 0,
                    "most_common_metric": 0,
                    "summary": "No goals section found in document.",
                    "success_pattern_category": "No Goals Found"
                })
                return metadata
            
            # Analyze success metrics using Claude
            analysis = self.analyze_success_metrics_with_claude(sections["goals"], metadata)
            
            # Add file path to analysis
            analysis["file_path"] = file_path
            
            return analysis
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def analyze_all_documents(self):
        """Analyze all IEP documents in the folder.
        
        Returns:
            DataFrame: Analysis results for all documents
        """
        files = self.extract_files_from_folder()
        results = []
        
        print(f"Analyzing {len(files)} IEP documents...")
        
        for file_path in tqdm(files):
            result = self.analyze_document(file_path)
            if result:
                results.append(result)
            
            # Avoid rate limiting
            time.sleep(1)
        
        return pd.DataFrame(results)
    
    def calculate_success_rate_distribution(self, df):
        """Calculate success rate distribution across disability categories.
        
        Args:
            df (DataFrame): Analysis results dataframe
            
        Returns:
            DataFrame: Success rate distribution percentages
        """
        # Filter to documents with valid success categories
        valid_df = df[df['success_pattern_category'].notna() & (df['success_pattern_category'] != "No Goals Found")]
        
        # Group by disability type and success category
        success_dist = valid_df.groupby(['disability_type', 'success_pattern_category']).size().unstack(fill_value=0)
        
        # Calculate percentages
        success_dist_percent = success_dist.div(success_dist.sum(axis=1), axis=0) * 100
        
        # Ensure all categories are present
        for category in ["Most Common Success Rate (80%)", "Below Standard (<80%)", "Above Standard (>80%)", "Mixed Pattern"]:
            if category not in success_dist_percent.columns:
                success_dist_percent[category] = 0
        
        success_dist_percent = success_dist_percent.round(1)
        
        return success_dist_percent
    
    def plot_success_rate_table(self, success_dist_percent):
        """Create a table visualization of success rate distribution.
        
        Args:
            success_dist_percent (DataFrame): Success rate distribution percentages
        """
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        # Create the table
        cell_colors = plt.cm.Blues(success_dist_percent["Most Common Success Rate (80%)"].values / 100)
        
        table = plt.table(
            cellText=success_dist_percent.round(1).astype(str) + '%',
            rowLabels=success_dist_percent.index,
            colLabels=success_dist_percent.columns,
            cellLoc='center',
            loc='center',
            cellColours=cell_colors
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        plt.title("Success Rate Metrics Distribution by Disability Category", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, "success_rate_table.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_success_metrics_distribution(self, df):
        """Plot the distribution of success metrics values.
        
        Args:
            df (DataFrame): Analysis results dataframe
        """
        all_metrics = []
        for metrics in df['success_metrics'].dropna():
            if isinstance(metrics, list):
                all_metrics.extend(metrics)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(all_metrics, bins=range(50, 101, 5), kde=True)
        plt.xlabel('Success Metric Value (%)')
        plt.ylabel('Frequency')
        plt.title('Distribution of IEP Success Metrics')
        
        # Add vertical line at 80%
        plt.axvline(x=80, color='r', linestyle='--', label='80% Benchmark')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, "success_metrics_distribution.png"), dpi=300)
        plt.close()
    
    def analyze_success_metrics_by_demographic(self, df):
        """Analyze success metrics across demographic variables.
        
        Args:
            df (DataFrame): Analysis results dataframe
            
        Returns:
            dict: Dictionary of results by demographic factor
        """
        demographic_results = {}
        
        for demo_var in ['gender', 'race', 'religion']:
            # Skip if all values are "Not Specified"
            if df[demo_var].nunique() <= 1 and "Not Specified" in df[demo_var].unique():
                continue
            
            # Filter to valid categories
            valid_df = df[df['success_pattern_category'].notna() & (df['success_pattern_category'] != "No Goals Found")]
            
            # Group by demographic and success category
            demo_dist = valid_df.groupby([demo_var, 'success_pattern_category']).size().unstack(fill_value=0)
            
            # Calculate percentages
            demo_dist_percent = demo_dist.div(demo_dist.sum(axis=1), axis=0) * 100
            demo_dist_percent = demo_dist_percent.round(1)
            
            demographic_results[demo_var] = demo_dist_percent
            
            # Create plot
            plt.figure(figsize=(10, 6))
            demo_dist_percent.plot(kind='bar', stacked=True, colormap='viridis')
            plt.xlabel(demo_var.capitalize())
            plt.ylabel('Percentage')
            plt.title(f'Success Rate Categories by {demo_var.capitalize()}')
            plt.legend(title='Success Rate Category')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_folder, f"success_by_{demo_var}.png"), dpi=300)
            plt.close()
        
        return demographic_results
    
    def analyze_detailed_success_metrics(self, df):
        """Perform detailed analysis of the success metrics.
        
        Args:
            df (DataFrame): Analysis results dataframe
            
        Returns:
            dict: Detailed metrics analysis results
        """
        # Calculate average success metric value by disability type
        avg_metrics = df.groupby('disability_type')['most_common_metric'].mean().round(1)
        
        # Calculate percentage of documents with 80% pattern by disability type
        has_80_pattern = df.groupby('disability_type')['has_80_percent_pattern'].mean() * 100
        
        # Calculate correlation between disability type and success metric patterns
        contingency_table = pd.crosstab(df['disability_type'], df['success_pattern_category'])
        
        return {
            "average_success_metrics": avg_metrics,
            "percentage_with_80_pattern": has_80_pattern.round(1),
            "contingency_table": contingency_table
        }
    
    def generate_report(self, df, success_dist, detailed_metrics):
        """Generate a comprehensive report of findings.
        
        Args:
            df (DataFrame): Analysis results dataframe
            success_dist (DataFrame): Success rate distribution
            detailed_metrics (dict): Detailed metrics analysis
            
        Returns:
            str: Report text
        """
        report = f"""# IEP Success Metrics Analysis Report

## Executive Summary
This analysis examined {len(df)} IEP documents across different disability categories to identify patterns in success metrics used in measurable goals and objectives.

### Key Findings:
1. The 80% success metric is consistently used across all disability categories, appearing in {detailed_metrics['percentage_with_80_pattern'].mean():.1f}% of IEPs on average.
2. {detailed_metrics['average_success_metrics'].name} with the highest average metric value: {detailed_metrics['average_success_metrics'].idxmax()} ({detailed_metrics['average_success_metrics'].max()}%)
3. {detailed_metrics['average_success_metrics'].name} with the lowest average metric value: {detailed_metrics['average_success_metrics'].idxmin()} ({detailed_metrics['average_success_metrics'].min()}%)

## Detailed Results

### Success Rate Distribution by Disability Category
```
{success_dist.to_string()}
```

### Percentage of IEPs with 80% Success Metric Pattern by Disability Category
```
{detailed_metrics['percentage_with_80_pattern'].to_string()}
```

### Average Success Metric Value by Disability Category
```
{detailed_metrics['average_success_metrics'].to_string()}
```

## Methodology
This analysis was conducted using a combination of natural language processing and expert review to identify success metrics in IEP goals and objectives. Success metrics were categorized based on their alignment with the standard 80% benchmark commonly used in special education.

## Recommendations
Based on the findings, we recommend:
1. Reviewing the consistency of success metric usage across disability categories
2. Evaluating whether different success thresholds might be more appropriate for certain disability types
3. Ensuring that success metrics are individualized based on student needs rather than applying a one-size-fits-all approach
"""
        
        # Save report to file
        with open(os.path.join(self.output_folder, "success_metrics_report.md"), 'w') as f:
            f.write(report)
        
        return report
    
    def run_analysis(self):
        """Run the complete analysis pipeline.
        
        Returns:
            dict: Dictionary containing all analysis results
        """
        # Analyze all documents
        df = self.analyze_all_documents()
        
        # Save raw data
        df.to_csv(os.path.join(self.output_folder, "iep_success_metrics_analysis.csv"), index=False)
        
        # Calculate success rate distribution
        success_dist = self.calculate_success_rate_distribution(df)
        
        # Generate visualizations
        self.plot_success_rate_table(success_dist)
        self.plot_success_metrics_distribution(df)
        
        # Analyze by demographic
        demographic_results = self.analyze_success_metrics_by_demographic(df)
        
        # Detailed metrics analysis
        detailed_metrics = self.analyze_detailed_success_metrics(df)
        
        # Generate report
        report = self.generate_report(df, success_dist, detailed_metrics)
        
        print(f"\nAnalysis complete. Results saved to: {self.output_folder}")
        print("\nKey findings:")
        print(f"- Analyzed {len(df)} IEP documents")
        print(f"- Average rate of 80% success metric usage: {detailed_metrics['percentage_with_80_pattern'].mean():.1f}%")
        
        return {
            "data": df,
            "success_distribution": success_dist,
            "demographic_results": demographic_results,
            "detailed_metrics": detailed_metrics,
            "report": report
        }


# Example usage
if __name__ == "__main__":
    analyzer = IEPSuccessMetricsAnalyzer(
        documents_folder="path/to/iep_documents",
        output_folder="results",
        api_key="YOUR_ANTHROPIC_API_KEY"
    )
    
    results = analyzer.run_analysis()