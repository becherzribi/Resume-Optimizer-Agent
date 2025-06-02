from langchain_community.llms import Ollama
import torch
from typing import Optional, Dict, Any
import time

class EnhancedLLMEngine:
    """Enhanced LLM engine with model management and optimization."""
    
    def __init__(self):
        self.available_models = self._check_available_models()
        self.model_cache = {}
        
    def _check_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Check which models are available and their configurations."""
        models = {
            "mistral:latest": {
                "description": "Mistral 7B - Fast and efficient for general tasks",
                "size": "4.1GB",
                "strengths": ["Speed", "General reasoning", "Multilingual"],
                "best_for": "Quick analysis and general feedback"
            },
            "llama3.1:latest": {
                "description": "Llama 3.1 - Strong reasoning and detailed analysis",
                "size": "4.9GB", 
                "strengths": ["Detailed analysis", "Code understanding", "Logic"],
                "best_for": "Comprehensive resume analysis"
            },
            "deepseek-r1:1.5b": {
                "description": "DeepSeek R1 - Lightweight and efficient",
                "size": "1.1GB",
                "strengths": ["Speed", "Low resource usage", "Structured output"],
                "best_for": "Quick feedback and structured analysis"
            },
            "gemma:7b": {
                "description": "Google Gemma 7B - Balanced performance",
                "size": "~5GB",
                "strengths": ["Balanced performance", "Safety", "Instruction following"],
                "best_for": "Professional resume writing"
            },
            "codellama:7b": {
                "description": "Code Llama 7B - Specialized for technical content",
                "size": "~4GB",
                "strengths": ["Technical skills", "Code analysis", "IT terminology"],
                "best_for": "Technical resume optimization"
            }
        }
        return models
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        return self.available_models.get(model_name, {})
    
    def get_cached_model(self, model_name: str) -> Ollama:
        """Get or create a cached model instance."""
        if model_name not in self.model_cache:
            self.model_cache[model_name] = Ollama(
                model=model_name,
                temperature=0.1,  # Low temperature for consistent results
                top_p=0.9,
                top_k=40
            )
        return self.model_cache[model_name]

# Global engine instance
llm_engine = EnhancedLLMEngine()

def analyze_resume(
    resume_text: str,
    job_text: str,
    prompt_template: str,
    model_name: str = "mistral:latest",
    target_role: str = "",
    language: str = "English"
) -> str:
    """
    Enhanced resume analysis with model optimization and error handling.
    
    Args:
        resume_text: The resume content
        job_text: The job description content
        prompt_template: The prompt template to use
        model_name: The LLM model to use
        target_role: Optional target role for focused analysis
        language: Language of the resume
    
    Returns:
        Analysis feedback from the LLM
    """
    try:
        # Get model instance
        llm = llm_engine.get_cached_model(model_name)
        
        # Enhanced prompt with additional context
        enhanced_prompt = f"""
Language: {language}
Target Role: {target_role if target_role else "General"}
Model: {model_name}

{prompt_template}

Additional Instructions:
- Provide specific, actionable feedback
- Focus on {target_role if target_role else "general professional development"}
- Consider the language and cultural context: {language}
- Prioritize suggestions that will have the highest impact
- Include quantifiable improvements where possible

Resume:
{resume_text}

Job Description:
{job_text}

Please provide a comprehensive analysis following the structure requested above.
"""
        
        # Measure response time
        start_time = time.time()
        response = llm(enhanced_prompt)
        response_time = time.time() - start_time
        
        # Add performance metadata
        model_info = llm_engine.get_model_info(model_name)
        performance_note = f"\n\n---\n*Analysis completed using {model_info.get('description', model_name)} in {response_time:.1f}s*"
        
        return response + performance_note
        
    except Exception as e:
        error_message = f"Error during analysis with {model_name}: {str(e)}"
        
        # Fallback to a different model if available
        fallback_models = ["mistral:latest", "deepseek-r1:1.5b", "llama3.1:latest"]
        for fallback_model in fallback_models:
            if fallback_model != model_name and fallback_model in llm_engine.available_models:
                try:
                    print(f"Attempting fallback to {fallback_model}...")
                    return analyze_resume(resume_text, job_text, prompt_template, fallback_model, target_role, language)
                except:
                    continue
        
        return f"⚠️ {error_message}\n\nPlease try with a different model or check your Ollama installation."

def generate_enhanced_resume(
    original_resume: str,
    feedback: str,
    model_name: str = "mistral:latest",
    target_role: str = "",
    language: str = "English",
    preserve_structure: bool = True
) -> str:
    """
    Generate an enhanced resume based on feedback.
    
    Args:
        original_resume: Original resume text
        feedback: Feedback to incorporate
        model_name: LLM model to use
        target_role: Target role for optimization
        language: Resume language
        preserve_structure: Whether to preserve original structure
    
    Returns:
        Enhanced resume text
    """
    try:
        llm = llm_engine.get_cached_model(model_name)
        
        structure_instruction = """
CRITICAL: Preserve the EXACT structure and section headings of the original resume. 
Only modify the content within each section, not the headings or overall organization.
""" if preserve_structure else ""
        
        enhancement_prompt = f"""
You are a professional resume writer specializing in {target_role if target_role else "career optimization"}.

Task: Rewrite the resume below to incorporate the feedback while maintaining professionalism and accuracy.

Language: {language}
Target Role: {target_role if target_role else "General professional development"}

{structure_instruction}

Key Requirements:
1. Incorporate feedback naturally and professionally
2. Maintain all factual information - do not fabricate experiences
3. Use strong action verbs and quantifiable achievements
4. Ensure ATS (Applicant Tracking System) compatibility
5. Keep the same overall length and format
6. Use industry-appropriate terminology for {target_role if target_role else "the field"}

Feedback to Incorporate:
{feedback}

Original Resume:
{original_resume}

Generate the improved resume:
"""
        
        enhanced_resume = llm(enhancement_prompt)
        return enhanced_resume
        
    except Exception as e:
        return f"⚠️ Error generating enhanced resume: {str(e)}\n\nOriginal resume returned unchanged.\n\n{original_resume}"

def quick_keyword_suggestions(
    resume_text: str,
    job_text: str,
    model_name: str = "deepseek-r1:1.5b"  # Use lighter model for quick suggestions
) -> str:
    """
    Get quick keyword suggestions using a lightweight model.
    """
    try:
        llm = llm_engine.get_cached_model(model_name)
        
        keyword_prompt = f"""
Analyze the job description and resume below. Provide a concise list of 5-10 important keywords/phrases from the job description that are missing from the resume.

Focus on:
- Technical skills and tools
- Industry-specific terminology
- Key qualifications and certifications
- Important soft skills mentioned in the job

Format your response as a simple bullet list.

Job Description:
{job_text[:1000]}...

Resume:
{resume_text[:800]}...

Missing Keywords:
"""
        
        return llm(keyword_prompt)
        
    except Exception as e:
        return f"⚠️ Error getting keyword suggestions: {str(e)}"

def get_model_recommendations(target_role: str = "", resume_length: str = "medium") -> Dict[str, str]:
    """
    Recommend the best model based on use case.
    
    Args:
        target_role: Target job role
        resume_length: "short", "medium", or "long"
    
    Returns:
        Dictionary with model recommendations
    """
    recommendations = {}
    
    # Technical roles
    if any(tech_term in target_role.lower() for tech_term in ['developer', 'engineer', 'programmer', 'data', 'ai', 'ml']):
        recommendations['primary'] = "codellama:7b"
        recommendations['alternative'] = "llama3.1:latest"
        recommendations['reason'] = "Technical roles benefit from CodeLlama's understanding of technical terminology"
    
    # Quick analysis
    elif resume_length == "short" or "quick" in target_role.lower():
        recommendations['primary'] = "deepseek-r1:1.5b"
        recommendations['alternative'] = "mistral:latest"
        recommendations['reason'] = "Lightweight models provide fast results for shorter content"
    
    # Detailed analysis
    elif resume_length == "long" or "senior" in target_role.lower():
        recommendations['primary'] = "llama3.1:latest"
        recommendations['alternative'] = "gemma:7b"
        recommendations['reason'] = "Larger models provide more comprehensive analysis for complex resumes"
    
    # Default
    else:
        recommendations['primary'] = "mistral:latest"
        recommendations['alternative'] = "gemma:7b"
        recommendations['reason'] = "Balanced performance for general resume optimization"
    
    return recommendations

# Backwards compatibility
def analyze_resume_legacy(resume_text, job_text, prompt_template, model_name="mistral:latest"):
    """Legacy function for backwards compatibility."""
    return analyze_resume(resume_text, job_text, prompt_template, model_name)