# llm_engine.py
from logging import log
from langchain_community.llms import Ollama
import torch
from typing import Optional, Dict, Any
import time
import gc # Import garbage collector

class EnhancedLLMEngine:
    """Enhanced LLM engine with model management and optimization."""

    def __init__(self):
        self.available_models_info = self._check_available_models() # Renamed for clarity
        self.loaded_model_name: Optional[str] = None
        self.loaded_model_instance: Optional[Ollama] = None
        # self.model_cache = {} # We will manage one loaded model at a time

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

    def get_available_model_names(self) -> list[str]:
        return list(self.available_models_info.keys())

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        return self.available_models_info.get(model_name, {})

    def unload_current_model(self):
        """Unloads the currently loaded model to free resources."""
        if self.loaded_model_instance:
            print(f"Unloading model: {self.loaded_model_name}")
            del self.loaded_model_instance
            self.loaded_model_instance = None
            self.loaded_model_name = None
            gc.collect() # Explicitly call garbage collector
            if torch.cuda.is_available():
                torch.cuda.empty_cache() # Clear PyTorch's CUDA cache
            print("Model unloaded and CUDA cache cleared (if applicable).")


    def get_llm_instance(self, model_name: str) -> Ollama:
        """Get or create a model instance. Unloads previous model if different."""
        if self.loaded_model_name == model_name and self.loaded_model_instance:
            print(f"Returning already loaded model: {model_name}")
            return self.loaded_model_instance

        # Unload previous model if it's different and loaded
        if self.loaded_model_instance and self.loaded_model_name != model_name:
            self.unload_current_model()

        print(f"Loading model: {model_name}...")
        # Add a try-except block for model loading
        try:
            self.loaded_model_instance = Ollama(
                model=model_name,
                temperature=0.1,
                top_p=0.9,
                top_k=40
            )
            self.loaded_model_name = model_name
            print(f"Model {model_name} loaded successfully.")
            return self.loaded_model_instance
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            # Optionally, try to load a default fallback model or raise the error
            if self.loaded_model_instance: # If a previous model was loaded, try to return it
                print(f"Falling back to previously loaded model: {self.loaded_model_name}")
                return self.loaded_model_instance
            raise e # Or handle more gracefully

# Global engine instance
llm_engine = EnhancedLLMEngine()



def analyze_resume(
    resume_text: str,            # Actual resume content
    job_text: str,               # Actual job description content
    prompt_template_string: str, # The raw string from ENHANCED_PROMPT_FILE
    model_name: str = "mistral:latest",
    target_role: str = "",
    language: str = "English"
) -> str:
    """
    Analyzes the resume against the job description using an LLM.
    The prompt_template_string should contain placeholders like {resume}, {job},
    {target_role}, and {language}.
    """
    try:
        llm = llm_engine.get_llm_instance(model_name)
        if not llm:
            return f"⚠️ Critical Error: Could not load LLM model {model_name}. Analysis cannot proceed."

        # Fill in the placeholders in the prompt template string
        final_llm_prompt = prompt_template_string.format(
            target_role=target_role if target_role else "Not Specified",
            language=language,
            resume=resume_text,
            job=job_text
        )
        # The 'Model:' and 'Additional Instructions' you had before are now assumed
        # to be part of the `prompt_template_string` itself if desired.
        # Or, you can add them here if they are general to all prompts:
        # final_llm_prompt = f"""
        # Context for LLM:
        # Model Being Used: {model_name}
        # Target Role: {target_role if target_role else "General"}
        # Language: {language}
        #
        # {formatted_template_content}
        # """
        # For simplicity with the new prompt structure, let's assume the template is comprehensive.

        start_time = time.time()
        response = llm(final_llm_prompt) # Send the fully prepared prompt to the LLM
        response_time = time.time() - start_time

        model_info = llm_engine.get_model_info(model_name)
        performance_note = f"\n\n---\n*Analysis completed using {model_info.get('description', model_name)} in {response_time:.1f}s. LLM: {llm_engine.loaded_model_name}*"
        return response + performance_note

    except KeyError as e:
        # This error occurs if a placeholder in prompt_template_string is not provided in .format()
        error_message = f"Error formatting prompt: Missing placeholder {e}. Please check 'ENHANCED_PROMPT_FILE'."
        print(f"Detailed error in analyze_resume (KeyError): {error_message}")
        log.error("Prompt formatting error", missing_key=str(e), prompt_template=prompt_template_string[:200]) # Log part of template
        return f"⚠️ {error_message}"
    except Exception as e:
        error_message = f"Error during analysis with {model_name}: {str(e)}"
        print(f"Detailed error in analyze_resume: {e}") # For server logs
        log.exception("LLM analysis failed", model=model_name, exc_info=True)

        # Fallback logic
        fallback_models = ["mistral:latest", "deepseek-r1:1.5b"] # Ensure these are in available_models_info
        for fallback_model in fallback_models:
            if fallback_model != model_name and fallback_model in llm_engine.get_available_model_names():
                try:
                    print(f"Attempting fallback to {fallback_model}...")
                    # Recursive call with the same prompt_template_string
                    return analyze_resume(resume_text, job_text, prompt_template_string, fallback_model, target_role, language)
                except Exception as fallback_e:
                    print(f"Fallback to {fallback_model} also failed: {fallback_e}")
                    log.warning("Fallback LLM analysis failed", model=fallback_model, error=str(fallback_e))
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
    try:
        # Get model instance using the new method
        llm = llm_engine.get_llm_instance(model_name) # Changed
        if not llm:
            return f"⚠️ Critical Error: Could not load LLM model {model_name} for resume generation."
        # ... (rest of the function is the same)
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

def get_model_recommendations(target_role: str = "", resume_length: str = "medium") -> Dict[str, str]:
    # ... (this function remains the same but ensure model names are valid keys in llm_engine.available_models_info)
    recommendations = {}
    available_models = llm_engine.get_available_model_names() # Get models actually available

    # Technical roles
    primary_rec, alt_rec = "codellama:7b", "llama3.1:latest"
    reason = "Technical roles benefit from CodeLlama's understanding of technical terminology"
    if not any(tech_term in target_role.lower() for tech_term in ['developer', 'engineer', 'programmer', 'data', 'ai', 'ml']):
        # Quick analysis
        if resume_length == "short" or "quick" in target_role.lower():
            primary_rec, alt_rec = "deepseek-r1:1.5b", "mistral:latest"
            reason = "Lightweight models provide fast results for shorter content"
        # Detailed analysis
        elif resume_length == "long" or "senior" in target_role.lower():
            primary_rec, alt_rec = "llama3.1:latest", "gemma:7b"
            reason = "Larger models provide more comprehensive analysis for complex resumes"
        # Default
        else:
            primary_rec, alt_rec = "mistral:latest", "gemma:7b"
            reason = "Balanced performance for general resume optimization"

    # Ensure recommendations are available
    recommendations['primary'] = primary_rec if primary_rec in available_models else (alt_rec if alt_rec in available_models else available_models[0])
    recommendations['alternative'] = alt_rec if alt_rec in available_models and alt_rec != recommendations['primary'] else (available_models[0] if available_models[0] != recommendations['primary'] else (available_models[1] if len(available_models) > 1 else recommendations['primary']))
    recommendations['reason'] = reason
    return recommendations

# Remove analyze_resume_legacy if it's not critically needed for other scripts
# def analyze_resume_legacy(resume_text, job_text, prompt_template, model_name="mistral:latest"):
#     """Legacy function for backwards compatibility."""
#     return analyze_resume(resume_text, job_text, prompt_template, model_name)