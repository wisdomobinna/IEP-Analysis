import os
import re
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import anthropic
import time

class IEPSemanticAnalyzer:
    def __init__(self, documents_folder, api_key):
        """Initialize with documents folder and Anthropic API key."""
        self.documents_folder = documents_folder
        self.client = anthropic.Anthropic(api_key=api_key)
        self.documents = {}
        
        self.disability_types = {
            "Physical": ["Orthopedic Impairment", "Cerebral Palsy", "Spina Bifida", 
                         "Muscular Dystrophy", "Quadriplegia", "Limb Deficiency", "Paraplegia"],
            "Cognitive/Developmental": ["Sensory Processing", "Autism", "Nonverbal Communication"],
            "Learning": ["Dyslexia"],
            "Neurological": ["Complex Neurological", "ADHD", "Tourette"],
            "Mental Health": ["Anxiety"],
            "Health Conditions": ["Chronic Health", "Genetic Condition"]
        }
        
        # Demographic identifiers
        self.gender_identifiers = ["Male", "Female"]
        self.race_identifiers = ["White", "Black", "Asian"]
        self.religion_identifiers = ["Christian", "Muslim", "Jewish", "Atheist"]
        
        # Load documents
        self.load_documents()
    
    def load_documents(self):
        """Load all IEP documents from the documents folder."""
        for filename in os.listdir(self.documents_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.documents_folder, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                    # Extract metadata
                    disability = self.extract_disability(filename, content)
                    demographics = self.extract_demographics(filename, content)
                    
                    self.documents[filename] = {
                        "content": content,
                        "disability": disability,
                        "demographics": demographics,
                        "sections": self.extract_sections(content)
                    }
        
        print(f"Loaded {len(self.documents)} documents")
    
    def extract_disability(self, filename, content):
        """Extract disability type from filename or content."""
        for disability_category, keywords in self.disability_types.items():
            for keyword in keywords:
                if keyword.lower() in filename.lower() or keyword.lower() in content.lower():
                    return {"category": disability_category, "specific": keyword}
        
        return {"category": "Unknown", "specific": "Unknown"}
    
    def extract_demographics(self, filename, content):
        """Extract demographic identifiers from filename or content."""
        demographics = {"gender": None, "race": None, "religion": None}
        
        # Check gender
        for gender in self.gender_identifiers:
            if gender.lower() in filename.lower() or gender.lower() in content.lower():
                demographics["gender"] = gender
                break
        
        # Check race
        for race in self.race_identifiers:
            if race.lower() in filename.lower() or race.lower() in content.lower():
                demographics["race"] = race
                break
        
        # Check religion
        for religion in self.religion_identifiers:
            if religion.lower() in filename.lower() or religion.lower() in content.lower():
                demographics["religion"] = religion
                break
        
        return demographics
    
    def extract_sections(self, content):
        """Extract standard IEP sections from document content."""
        sections = {}
        
        # Common IEP section titles
        section_patterns = {
            "present_levels": r"(?:Present Levels of Performance|Present Levels|Current Performance)(.*?)(?:Student Needs|Goals and Objectives|Accommodations|$)",
            "student_needs": r"(?:Student Needs|Impact of Disability)(.*?)(?:Goals and Objectives|Accommodations|$)",
            "goals": r"(?:Goals and Objectives|Goals|Objectives)(.*?)(?:Accommodations|$)",
            "accommodations": r"(?:Accommodations and Modifications|Accommodations|Modifications)(.*?)(?:$)"
        }
        
        # Extract each section using regex
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                sections[section_name] = match.group(1).strip()
            else:
                sections[section_name] = ""
        
        return sections
    
    def analyze_document_framing(self, doc_content, disability, demographics):
        """Use Claude API to analyze the framing of disabilities in a document."""
        prompt = f"""
        Please analyze the following IEP document for a student with {disability['specific']} 
        and identify how the disability is framed. Focus specifically on:

        1. Strengths-based language: Identify language that focuses on the student's strengths, abilities, 
           and potential.
        2. Deficit-based language: Identify language that focuses on challenges, disabilities, 
           limitations, and problems.
        3. Key themes: Identify the main themes in how this disability is described.
        4. Accommodations focus: Are accommodations framed as ways to access strengths or as ways to 
           mitigate deficits?

        Return your analysis in JSON format with the following structure:
        {{
            "strengths_language": {{
                "count": <number of instances>,
                "examples": [<list of 5 examples>]
            }},
            "deficit_language": {{
                "count": <number of instances>,
                "examples": [<list of 5 examples>]
            }},
            "key_themes": [<list of 5 main themes>],
            "accommodations_framing": "<strengths-focused or deficit-focused or balanced>",
            "overall_framing": "<a 1-2 sentence summary of how the disability is framed>"
        }}

        Document content:
        {doc_content}
        """
        
        try:
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                temperature=0,
                system="You are a research assistant analyzing disability framing in educational documents. Provide objective, detailed analysis in JSON format.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse JSON from response
            json_str = response.content[0].text
            # Find JSON content (may be wrapped in ```json blocks)
            json_match = re.search(r'```json\s*(.*?)\s*```', json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            
            analysis = json.loads(json_str)
            
            # Add metadata
            analysis["disability"] = disability
            analysis["demographics"] = demographics
            
            return analysis
        
        except Exception as e:
            print(f"Error analyzing document: {e}")
            return None
    
    def compare_disability_framing(self):
        """Compare disability framing across different disabilities and demographics."""
        analyses = []
        
        # Analyze a subset of documents for each disability type to manage API costs
        for disability_category in self.disability_types:
            # Get documents for this disability category
            category_docs = {k: v for k, v in self.documents.items() 
                             if v["disability"]["category"] == disability_category}
            
            if not category_docs:
                continue
            
            # Take up to 5 representative documents for this category
            sample_docs = list(category_docs.items())[:5]
            
            for doc_name, doc_data in sample_docs:
                print(f"Analyzing document: {doc_name}")
                
                # Combine all sections for full document analysis
                full_content = "\n\n".join(doc_data["sections"].values())
                
                # Call Claude API for analysis
                analysis = self.analyze_document_framing(
                    full_content, 
                    doc_data["disability"],
                    doc_data["demographics"]
                )
                
                if analysis:
                    analysis["document"] = doc_name
                    analyses.append(analysis)
                
                # Avoid rate limiting
                time.sleep(1)
        
        return analyses
    
    def analyze_framing_by_demographic(self, analyses):
        """Analyze disability framing patterns across demographics."""
        # Organize analyses by disability category
        by_disability = defaultdict(list)
        for analysis in analyses:
            disability_category = analysis["disability"]["category"]
            by_disability[disability_category].append(analysis)
        
        # Calculate strengths vs. deficits ratio by disability category
        framing_by_disability = {}
        for category, category_analyses in by_disability.items():
            total_strengths = sum(a["strengths_language"]["count"] for a in category_analyses)
            total_deficits = sum(a["deficit_language"]["count"] for a in category_analyses)
            avg_strengths = total_strengths / len(category_analyses)
            avg_deficits = total_deficits / len(category_analyses)
            
            framing_by_disability[category] = {
                "avg_strengths": avg_strengths,
                "avg_deficits": avg_deficits,
                "ratio": avg_strengths / avg_deficits if avg_deficits > 0 else float('inf'),
                "common_themes": self.find_common_themes(category_analyses)
            }
        
        # Organize analyses by gender
        by_gender = defaultdict(list)
        for analysis in analyses:
            gender = analysis["demographics"]["gender"]
            if gender:  # Only include analyses with gender data
                by_gender[gender].append(analysis)
        
        # Calculate strengths vs. deficits ratio by gender
        framing_by_gender = {}
        for gender, gender_analyses in by_gender.items():
            total_strengths = sum(a["strengths_language"]["count"] for a in gender_analyses)
            total_deficits = sum(a["deficit_language"]["count"] for a in gender_analyses)
            avg_strengths = total_strengths / len(gender_analyses)
            avg_deficits = total_deficits / len(gender_analyses)
            
            framing_by_gender[gender] = {
                "avg_strengths": avg_strengths,
                "avg_deficits": avg_deficits,
                "ratio": avg_strengths / avg_deficits if avg_deficits > 0 else float('inf'),
                "common_themes": self.find_common_themes(gender_analyses)
            }
        
        # Similarly analyze by race
        by_race = defaultdict(list)
        framing_by_race = {}
        
        for analysis in analyses:
            race = analysis["demographics"]["race"]
            if race:  # Only include analyses with race data
                by_race[race].append(analysis)
        
        for race, race_analyses in by_race.items():
            total_strengths = sum(a["strengths_language"]["count"] for a in race_analyses)
            total_deficits = sum(a["deficit_language"]["count"] for a in race_analyses)
            avg_strengths = total_strengths / len(race_analyses)
            avg_deficits = total_deficits / len(race_analyses)
            
            framing_by_race[race] = {
                "avg_strengths": avg_strengths,
                "avg_deficits": avg_deficits,
                "ratio": avg_strengths / avg_deficits if avg_deficits > 0 else float('inf'),
                "common_themes": self.find_common_themes(race_analyses)
            }
        
        return {
            "by_disability": framing_by_disability,
            "by_gender": framing_by_gender,
            "by_race": framing_by_race
        }
    
    def find_common_themes(self, analyses):
        """Find common themes across a set of analyses."""
        all_themes = []
        for analysis in analyses:
            all_themes.extend(analysis["key_themes"])
        
        # Count theme occurrences
        theme_counts = {}
        for theme in all_themes:
            theme_lower = theme.lower()
            theme_counts[theme_lower] = theme_counts.get(theme_lower, 0) + 1
        
        # Sort by frequency
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 5 themes
        return [theme for theme, count in sorted_themes[:5]]
    
    def plot_strengths_vs_deficits(self, framing_data):
        """Create visualizations of strengths vs. deficits ratios."""
        # Plot by disability
        disability_df = pd.DataFrame({
            'Category': list(framing_data["by_disability"].keys()),
            'Strengths': [data["avg_strengths"] for data in framing_data["by_disability"].values()],
            'Deficits': [data["avg_deficits"] for data in framing_data["by_disability"].values()]
        })
        
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        
        disability_melted = pd.melt(
            disability_df, 
            id_vars=['Category'], 
            value_vars=['Strengths', 'Deficits'],
            var_name='Language Type', 
            value_name='Count'
        )
        
        ax = sns.barplot(x='Category', y='Count', hue='Language Type', data=disability_melted)
        plt.title('Strengths vs. Deficits Language by Disability Category', fontsize=16)
        plt.xlabel('Disability Category', fontsize=12)
        plt.ylabel('Average Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('strengths_deficits_by_disability.png')
        plt.close()
        
        # Similar plots for gender and race
        gender_df = pd.DataFrame({
            'Gender': list(framing_data["by_gender"].keys()),
            'Strengths': [data["avg_strengths"] for data in framing_data["by_gender"].values()],
            'Deficits': [data["avg_deficits"] for data in framing_data["by_gender"].values()]
        })
        
        plt.figure(figsize=(10, 6))
        gender_melted = pd.melt(
            gender_df, 
            id_vars=['Gender'], 
            value_vars=['Strengths', 'Deficits'],
            var_name='Language Type', 
            value_name='Count'
        )
        
        ax = sns.barplot(x='Gender', y='Count', hue='Language Type', data=gender_melted)
        plt.title('Strengths vs. Deficits Language by Gender', fontsize=16)
        plt.xlabel('Gender', fontsize=12)
        plt.ylabel('Average Count', fontsize=12)
        plt.tight_layout()
        plt.savefig('strengths_deficits_by_gender.png')
        plt.close()
        
        # Race plot
        race_df = pd.DataFrame({
            'Race': list(framing_data["by_race"].keys()),
            'Strengths': [data["avg_strengths"] for data in framing_data["by_race"].values()],
            'Deficits': [data["avg_deficits"] for data in framing_data["by_race"].values()]
        })
        
        plt.figure(figsize=(10, 6))
        race_melted = pd.melt(
            race_df, 
            id_vars=['Race'], 
            value_vars=['Strengths', 'Deficits'],
            var_name='Language Type', 
            value_name='Count'
        )
        
        ax = sns.barplot(x='Race', y='Count', hue='Language Type', data=race_melted)
        plt.title('Strengths vs. Deficits Language by Race', fontsize=16)
        plt.xlabel('Race', fontsize=12)
        plt.ylabel('Average Count', fontsize=12)
        plt.tight_layout()
        plt.savefig('strengths_deficits_by_race.png')
        plt.close()
    
    def analyze_accommodations_framing(self, analyses):
        """Analyze how accommodations are framed across different documents."""
        # Count accommodation framing types
        framing_counts = {"strengths-focused": 0, "deficit-focused": 0, "balanced": 0}
        for analysis in analyses:
            framing = analysis["accommodations_framing"].lower()
            if "strength" in framing:
                framing_counts["strengths-focused"] += 1
            elif "deficit" in framing:
                framing_counts["deficit-focused"] += 1
            else:
                framing_counts["balanced"] += 1
        
        # Group by disability category
        by_disability = defaultdict(lambda: {"strengths-focused": 0, "deficit-focused": 0, "balanced": 0})
        for analysis in analyses:
            category = analysis["disability"]["category"]
            framing = analysis["accommodations_framing"].lower()
            
            if "strength" in framing:
                by_disability[category]["strengths-focused"] += 1
            elif "deficit" in framing:
                by_disability[category]["deficit-focused"] += 1
            else:
                by_disability[category]["balanced"] += 1
        
        return {
            "overall": framing_counts,
            "by_disability": dict(by_disability)
        }
    
    def deep_semantic_analysis(self, sample_count=3):
        """Perform a deep semantic analysis on sample documents for each disability category."""
        category_samples = defaultdict(list)
        
        # Collect samples by disability category
        for doc_name, doc_data in self.documents.items():
            category = doc_data["disability"]["category"]
            if len(category_samples[category]) < sample_count:
                category_samples[category].append((doc_name, doc_data))
        
        results = []
        
        for category, samples in category_samples.items():
            for doc_name, doc_data in samples:
                print(f"Performing deep semantic analysis on {doc_name}...")
                
                # Combine all sections
                full_content = "\n\n".join(doc_data["sections"].values())
                
                # Prompt for deep semantic analysis
                prompt = f"""
                Please perform a deep semantic analysis of how disability is framed in this IEP document. 
                Beyond just counting strengths vs. deficits language, analyze:

                1. The underlying conceptual model of disability (medical model, social model, etc.)
                2. How agency is attributed (to the student, to supports, to the disability itself)
                3. The implicit assumptions about potential and limitations
                4. The nature of goals (remediation vs. compensation vs. development)
                5. The balance between individual adaptations vs. environmental modifications

                Return your analysis in JSON format:
                {{
                    "disability_model": "<medical/social/biopsychosocial/other>",
                    "agency_attribution": {{
                        "student_agency": <0-10 score>,
                        "support_agency": <0-10 score>,
                        "disability_as_agent": <0-10 score>,
                        "examples": [<key examples>]
                    }},
                    "potential_framing": "<limiting/open-ended/conditional/other>",
                    "goal_orientation": {{
                        "remediation_focus": <0-10 score>,
                        "compensation_focus": <0-10 score>,
                        "development_focus": <0-10 score>
                    }},
                    "adaptation_focus": {{
                        "individual_adaptation": <0-10 score>,
                        "environmental_modification": <0-10 score>
                    }},
                    "key_insights": [<list of 3-5 key insights>]
                }}

                Document:
                {full_content}
                """
                
                try:
                    response = self.client.messages.create(
                        model="claude-3-opus-20240229",
                        max_tokens=1500,
                        temperature=0,
                        system="You are a disability studies expert analyzing educational documents. Provide nuanced, detailed analysis in JSON format.",
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    
                    # Parse JSON from response
                    json_str = response.content[0].text
                    json_match = re.search(r'```json\s*(.*?)\s*```', json_str, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    
                    analysis = json.loads(json_str)
                    
                    # Add metadata
                    analysis["document"] = doc_name
                    analysis["disability"] = doc_data["disability"]
                    analysis["demographics"] = doc_data["demographics"]
                    
                    results.append(analysis)
                    
                    # Rate limiting
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"Error in deep analysis: {e}")
                    continue
        
        return results
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline and return results."""
        print("Starting document framing analysis...")
        framing_analyses = self.compare_disability_framing()
        
        print("Analyzing framing by demographic factors...")
        demographic_framing = self.analyze_framing_by_demographic(framing_analyses)
        
        print("Analyzing accommodations framing...")
        accommodations_analysis = self.analyze_accommodations_framing(framing_analyses)
        
        print("Creating visualizations...")
        self.plot_strengths_vs_deficits(demographic_framing)
        
        print("Performing deep semantic analysis...")
        semantic_analysis = self.deep_semantic_analysis(sample_count=2)  # Limit samples to manage API costs
        
        print("Analysis complete!")
        
        # Combine all results
        return {
            "framing_analyses": framing_analyses,
            "demographic_framing": demographic_framing,
            "accommodations_analysis": accommodations_analysis,
            "semantic_analysis": semantic_analysis
        }
    
    def generate_comprehensive_report(self, results):
        """Generate a comprehensive report based on all analyses."""
        prompt = f"""
        You are an expert in disability studies and educational equity. Please review the following 
        analysis results from a study of IEP documents and create a comprehensive report. Focus on:

        1. Patterns in how different disabilities are framed across documents
        2. Differences in language used based on disability type
        3. The balance between strengths-based and deficit-based approaches
        4. How accommodations are conceptualized
        5. Underlying models of disability and their implications
        6. Recommendations for more equitable and affirming disability language

        Analysis Results:
        {json.dumps(results, indent=2)}
        
        Your report should include:
        1. Executive Summary
        2. Key Findings
        3. Detailed Analysis
        4. Implications for Educational Equity
        5. Recommendations
        """
        
        try:
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4000,
                temperature=0,
                system="You are a disability studies and educational equity expert creating a comprehensive research report.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            report = response.content[0].text
            
            # Save report to file
            with open('iep_disability_framing_report.md', 'w', encoding='utf-8') as f:
                f.write(report)
            
            return report
            
        except Exception as e:
            print(f"Error generating report: {e}")
            return "Error generating report. Please check logs."


# Example usage
if __name__ == "__main__":
    analyzer = IEPSemanticAnalyzer(
        "path/to/iep_documents", 
        api_key="YOUR_ANTHROPIC_API_KEY"
    )
    
    results = analyzer.run_full_analysis()
    report = analyzer.generate_comprehensive_report(results)
    
    print("\nAnalysis complete! Report generated as 'iep_disability_framing_report.md'")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nStrengths vs. Deficits by Disability Category:")
    for category, data in results["demographic_framing"]["by_disability"].items():
        ratio = data["ratio"] if data["ratio"] != float('inf') else "∞"
        print(f"  {category}: {data['avg_strengths']:.1f} strengths / {data['avg_deficits']:.1f} deficits (ratio: {ratio})")
        print(f"    Common themes: {', '.join(data['common_themes'])}")