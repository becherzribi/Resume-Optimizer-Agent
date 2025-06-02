from sentence_transformers import SentenceTransformer
import faiss
import torch
import numpy as np
import pickle
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Model configurations for different languages
MODEL_CONFIGS = {
    'English': 'all-mpnet-base-v2',
    'French': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'Spanish': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'German': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'Other': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
}

class SemanticMatcher:
    def __init__(self, model_name: str = None, language: str = 'English', use_gpu: bool = True):
        """
        Initialize the semantic matcher with GPU optimization.
        
        Args:
            model_name: Specific model name (overrides language selection)
            language: Language for automatic model selection
            use_gpu: Whether to use GPU acceleration if available
        """
        # Determine device with GPU optimization
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = "cpu"
            print("💻 Using CPU for semantic processing")
        
        # Select appropriate model
        if model_name is None:
            model_name = MODEL_CONFIGS.get(language, MODEL_CONFIGS['English'])
        
        self.model_name = model_name
        self.language = language
        
        # Initialize model with device optimization
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            
            # GPU optimizations
            if self.device == "cuda":
                # Enable mixed precision for faster inference
                self.model.half()  # Use FP16 for faster processing
                
        except Exception as e:
            print(f"⚠️ Warning: Could not load {model_name}, falling back to default model")
            self.model = SentenceTransformer('all-mpnet-base-v2', device=self.device)
        
        self.index = None
        self.id_to_text = {}
        self.embeddings = None
        self.cache_dir = Path("cache/semantic_indexes")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, text_hash: str) -> Path:
        """Generate cache file path for a given text hash."""
        return self.cache_dir / f"index_{text_hash}_{self.model_name.replace('/', '_')}.pkl"
    
    def _compute_text_hash(self, sentences: List[str]) -> str:
        """Compute hash for caching purposes."""
        import hashlib
        text_content = "\n".join(sentences)
        return hashlib.md5(text_content.encode()).hexdigest()[:16]
    
    def build_index(self, job_sentences: List[str], force_rebuild: bool = False) -> None:
        """
        Build FAISS index from job sentences with caching support.
        
        Args:
            job_sentences: List of sentences to index
            force_rebuild: Force rebuilding even if cache exists
        """
        if not job_sentences:
            raise ValueError("Cannot build index from empty sentence list")
        
        # Check cache first
        text_hash = self._compute_text_hash(job_sentences)
        cache_path = self._get_cache_path(text_hash)
        
        if not force_rebuild and cache_path.exists():
            try:
                print("📦 Loading cached semantic index...")
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.index = cache_data['index']
                self.id_to_text = cache_data['id_to_text']
                self.embeddings = cache_data['embeddings']
                print("✅ Cached index loaded successfully!")
                return
                
            except Exception as e:
                print(f"⚠️ Cache loading failed: {e}. Rebuilding index...")
        
        print("🔨 Building new semantic index...")
        
        # Generate embeddings with batch processing for efficiency
        batch_size = 32 if self.device == "cuda" else 16
        embeddings = self.model.encode(
            job_sentences,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device,
            show_progress_bar=True
        )
        
        # Create FAISS index
        dim = embeddings.shape[1]
        
        # Use GPU index if available and beneficial
        if self.device == "cuda" and len(job_sentences) > 100:
            try:
                # GPU index for larger datasets
                res = faiss.StandardGpuResources()
                self.index = faiss.GpuIndexFlatIP(res, dim)
                print("🚀 Using GPU-accelerated FAISS index")
            except Exception as e:
                print(f"⚠️ GPU index failed: {e}. Using CPU index.")
                self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = faiss.IndexFlatIP(dim)
        
        self.index.add(embeddings)
        self.id_to_text = {i: job_sentences[i] for i in range(len(job_sentences))}
        self.embeddings = embeddings
        
        # Cache the index
        try:
            cache_data = {
                'index': self.index,
                'id_to_text': self.id_to_text,
                'embeddings': self.embeddings
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"💾 Index cached to {cache_path}")
        except Exception as e:
            print(f"⚠️ Caching failed: {e}")
        
        print("✅ Semantic index built successfully!")
    
    def query(self, resume_sentences: List[str], top_k: int = 2) -> Dict[str, List[Tuple[str, float]]]:
        """
        Query the index with resume sentences.
        
        Args:
            resume_sentences: List of resume sentences to match
            top_k: Number of top matches to return per sentence
            
        Returns:
            Dictionary mapping resume sentences to their top matches
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        if not resume_sentences:
            return {}
        
        print(f"🔍 Searching for semantic matches...")
        
        # Generate embeddings for resume sentences
        batch_size = 32 if self.device == "cuda" else 16
        resume_embeddings = self.model.encode(
            resume_sentences,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device,
            show_progress_bar=len(resume_sentences) > 50
        )
        
        # Search the index
        similarities, indices = self.index.search(resume_embeddings, top_k)
        
        # Format results
        results = {}
        for i, res_sent in enumerate(resume_sentences):
            matches = []
            for j in range(top_k):
                if j < len(indices[i]):
                    idx = int(indices[i][j])
                    score = float(similarities[i][j])
                    
                    # Only include matches above a threshold
                    if score > 0.3:  # Semantic similarity threshold
                        matches.append((self.id_to_text[idx], score))
            
            if matches:  # Only include sentences with good matches
                results[res_sent] = matches
        
        return results
    
    def get_best_matches(self, resume_text: str, job_text: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Get the best semantic matches between resume and job description.
        
        Returns:
            List of tuples (resume_sentence, job_sentence, similarity_score)
        """
        resume_sentences = [s.strip() for s in resume_text.split('.') if len(s.strip()) > 10]
        job_sentences = [s.strip() for s in job_text.split('.') if len(s.strip()) > 10]
        
        if not hasattr(self, 'index') or self.index is None:
            self.build_index(job_sentences)
        
        results = self.query(resume_sentences, top_k=1)
        
        # Flatten and sort results
        all_matches = []
        for resume_sent, matches in results.items():
            for job_sent, score in matches:
                all_matches.append((resume_sent, job_sent, score))
        
        # Sort by score and return top_k
        all_matches.sort(key=lambda x: x[2], reverse=True)
        return all_matches[:top_k]
    
    def analyze_coverage(self, resume_text: str, job_text: str, threshold: float = 0.7) -> Dict[str, any]:
        """
        Analyze how well the resume covers the job requirements semantically.
        
        Args:
            resume_text: Full resume text
            job_text: Full job description text
            threshold: Similarity threshold for considering a requirement "covered"
            
        Returns:
            Dictionary with coverage analysis
        """
        job_sentences = [s.strip() for s in job_text.split('.') if len(s.strip()) > 20]
        resume_sentences = [s.strip() for s in resume_text.split('.') if len(s.strip()) > 10]
        
        if not hasattr(self, 'index') or self.index is None:
            self.build_index(job_sentences)
        
        # Find best match for each job requirement
        job_embeddings = self.model.encode(job_sentences, convert_to_numpy=True, normalize_embeddings=True)
        resume_embeddings = self.model.encode(resume_sentences, convert_to_numpy=True, normalize_embeddings=True)
        
        # Compute similarity matrix
        similarity_matrix = np.dot(job_embeddings, resume_embeddings.T)
        
        covered_requirements = []
        uncovered_requirements = []
        
        for i, job_sent in enumerate(job_sentences):
            max_similarity = np.max(similarity_matrix[i])
            best_match_idx = np.argmax(similarity_matrix[i])
            
            if max_similarity >= threshold:
                covered_requirements.append({
                    'requirement': job_sent,
                    'best_match': resume_sentences[best_match_idx],
                    'similarity': float(max_similarity)
                })
            else:
                uncovered_requirements.append({
                    'requirement': job_sent,
                    'best_similarity': float(max_similarity)
                })
        
        coverage_percentage = len(covered_requirements) / len(job_sentences) * 100
        
        return {
            'coverage_percentage': round(coverage_percentage, 2),
            'covered_requirements': covered_requirements,
            'uncovered_requirements': uncovered_requirements,
            'total_requirements': len(job_sentences)
        }
    
    def suggest_improvements(self, resume_text: str, job_text: str) -> List[str]:
        """
        Suggest specific improvements based on semantic analysis.
        """
        coverage_analysis = self.analyze_coverage(resume_text, job_text)
        suggestions = []
        
        # Analyze uncovered requirements
        uncovered = coverage_analysis['uncovered_requirements']
        if uncovered:
            suggestions.append("🎯 **High Priority Additions:**")
            for req in uncovered[:3]:  # Top 3 uncovered requirements
                suggestions.append(f"   • Add experience related to: {req['requirement'][:100]}...")
        
        # Coverage feedback
        coverage_pct = coverage_analysis['coverage_percentage']
        if coverage_pct < 50:
            suggestions.append("🔴 **Low Coverage Alert:** Your resume covers less than 50% of job requirements. Consider major revisions.")
        elif coverage_pct < 70:
            suggestions.append("🟡 **Moderate Coverage:** Consider adding more relevant experience and skills.")
        else:
            suggestions.append("🟢 **Good Coverage:** Your resume aligns well with the job requirements.")
        
        return suggestions
    
    def clear_cache(self):
        """Clear all cached indexes."""
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                print("🗑️ Cache cleared successfully!")
        except Exception as e:
            print(f"⚠️ Cache clearing failed: {e}")

# Utility function for easy usage
def create_semantic_matcher(language: str = 'English', use_gpu: bool = True) -> SemanticMatcher:
    """
    Create a semantic matcher with optimal settings.
    
    Args:
        language: Target language for the model
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Configured SemanticMatcher instance
    """
    return SemanticMatcher(language=language, use_gpu=use_gpu)