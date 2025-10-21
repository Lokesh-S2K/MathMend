"""
embedding_pipeline.py
---------------------
Complete embedding and FAISS indexing pipeline for CalcMate.

Pipeline Sequence:
1. Load Preprocessed Data → 2. Generate Text Embeddings → 
3. Create Hybrid Vectors → 4. Build FAISS Index → 
5. Evaluate Retrieval → 6. Save Search System

Author: CalcMate Team
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple

# Add the current directory to path to import your modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your modules
from pipeline_sequence.embedder import encode_dataframe, build_embedding_records, save_embeddings_jsonl
from pipeline_sequence.indexer_faiss import FaissIndexer
from pipeline_sequence.evaluator import Evaluator

# Configuration
class Config:
    """Configuration for the embedding pipeline."""
    
    # Input/Output paths
    INPUT_DIR = "output/preprocessed"
    OUTPUT_DIR = "output/embeddings"
    LOGS_DIR = "logs"
    
    # Embedding parameters
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    BATCH_SIZE = 32
    NORMALIZE_EMBEDDINGS = True
    
    # FAISS parameters
    FAISS_INDEX_TYPE = "Flat"  # Options: "Flat", "HNSW", "IVF"
    USE_GPU = False
    
    # Vector configuration
    STRUCTURE_VECTOR_DIM = 256  # From your preprocessing
    TEXT_EMBEDDING_DIM = 384    # all-MiniLM-L6-v2 dimension
    HYBRID_VECTOR_DIM = 640     # structure_dim + text_dim
    
    # Evaluation parameters
    TEST_SPLIT_RATIO = 0.2
    TOP_K_VALUES = [1, 3, 5, 10]

# Setup logging
def setup_logging():
    """Configure logging for the pipeline."""
    os.makedirs(Config.LOGS_DIR, exist_ok=True)
    
    log_filename = f"{Config.LOGS_DIR}/embedding_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class EmbeddingPipeline:
    """
    Complete embedding and FAISS indexing pipeline for math problems.
    """
    
    def __init__(self):
        self.logger = setup_logging()
        self.df = None
        self.hybrid_vectors = None
        self.faiss_indexer = None
        self.stats = {}
        
    def run(self, input_file: str = None) -> FaissIndexer:
        """
        Execute the complete embedding pipeline.
        
        Args:
            input_file: Path to preprocessed CSV file. Auto-detects latest if None.
            
        Returns:
            FAISS indexer object for querying
        """
        self.logger.info("🚀 Starting Embedding Pipeline")
        
        try:
            # Step 1: Data Loading & Preparation
            self._load_data(input_file)
            
            # Step 2: Text Embedding Generation
            self._generate_text_embeddings()
            
            # Step 3: Hybrid Vector Creation
            self._create_hybrid_vectors()
            
            # Step 4: FAISS Index Building
            self._build_faiss_index()
            
            # Step 5: Retrieval Evaluation
            self._evaluate_retrieval()
            
            # Step 6: Save System Artifacts
            self._save_system()
            
            self.logger.info("✅ Embedding Pipeline Completed Successfully!")
            self._print_summary()
            
            return self.faiss_indexer
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {str(e)}")
            raise
    
    def _find_latest_preprocessed_file(self) -> str:
        """Find the latest preprocessed metadata file."""
        input_dir = Path(Config.INPUT_DIR)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {Config.INPUT_DIR}")
        
        metadata_files = list(input_dir.glob("metadata_*.csv"))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata files found in {Config.INPUT_DIR}")
        
        # Get the latest file by timestamp
        latest_file = max(metadata_files, key=lambda x: x.stat().st_mtime)
        self.logger.info(f"Found latest preprocessed file: {latest_file}")
        return str(latest_file)
    
    def _load_data(self, input_file: str = None):
        """Step 1: Load preprocessed data and structure vectors."""
        self.logger.info("📥 Step 1: Loading Preprocessed Data...")
        
        # Find input file
        if input_file is None:
            input_file = self._find_latest_preprocessed_file()
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Load metadata
        self.df = pd.read_csv(input_file)
        self.stats['loaded_count'] = len(self.df)
        
        # Find corresponding embeddings file
        input_path = Path(input_file)
        embeddings_pattern = input_path.name.replace("metadata_", "embeddings_").replace(".csv", ".npy")
        embeddings_file = input_path.parent / embeddings_pattern
        
        if not embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        
        # Load structure vectors
        structure_vectors = np.load(embeddings_file)
        
        # Verify dimensions match
        if len(structure_vectors) != len(self.df):
            raise ValueError(f"Structure vectors count ({len(structure_vectors)}) doesn't match DataFrame count ({len(self.df)})")
        
        if structure_vectors.shape[1] != Config.STRUCTURE_VECTOR_DIM:
            raise ValueError(f"Structure vector dimension mismatch: expected {Config.STRUCTURE_VECTOR_DIM}, got {structure_vectors.shape[1]}")
        
        # Convert string representation of vectors to actual arrays
        self.df['structure_vector'] = [np.fromstring(vec_str.strip('[]'), sep=' ') 
                                      if isinstance(vec_str, str) else vec_str
                                      for vec_str in self.df['structure_vector']]
        
        self.logger.info(f"   Loaded {len(self.df)} problems with structure vectors")
        self.logger.info(f"   Structure vector dimension: {structure_vectors.shape[1]}")
        self.logger.info(f"   Columns: {list(self.df.columns)}")
        
        # Data quality checks
        valid_structure_vectors = self.df['structure_vector'].apply(
            lambda x: isinstance(x, np.ndarray) and len(x) == Config.STRUCTURE_VECTOR_DIM
        ).sum()
        
        valid_fingerprints = self.df['symbolic_fingerprint'].notna().sum()
        
        self.stats['data_quality'] = {
            'valid_structure_vectors': int(valid_structure_vectors),
            'valid_fingerprints': int(valid_fingerprints),
            'structure_vector_success_rate': float(valid_structure_vectors / len(self.df)),
            'fingerprint_success_rate': float(valid_fingerprints / len(self.df))
        }
        
        self.logger.info(f"   Data quality: {valid_structure_vectors}/{len(self.df)} valid structure vectors")
        self.logger.info(f"   Data quality: {valid_fingerprints}/{len(self.df)} valid fingerprints")
    
    def _generate_text_embeddings(self):
        """Step 2: Generate text/reasoning embeddings using SentenceTransformer."""
        self.logger.info("🔤 Step 2: Generating Text Embeddings...")
        
        # Check if we have enough data for embeddings
        valid_count = self.stats['data_quality']['valid_fingerprints']
        if valid_count < 10:
            self.logger.warning(f"Only {valid_count} valid fingerprints - text embeddings may be poor")
        
        # Generate reasoning embeddings
        self.df = encode_dataframe(
            self.df,
            problem_col="clean_text",
            fingerprint_col="symbolic_fingerprint",
            output_col="reasoning_vector",
            model_name=Config.MODEL_NAME,
            batch_size=Config.BATCH_SIZE,
            normalize=Config.NORMALIZE_EMBEDDINGS
        )
        
        # Statistics
        valid_text_embeddings = self.df['reasoning_vector'].apply(
            lambda x: isinstance(x, np.ndarray) and len(x) > 0
        ).sum()
        
        text_embedding_dim = self.df['reasoning_vector'].iloc[0].shape[0] if valid_text_embeddings > 0 else 0
        
        self.stats['text_embeddings'] = {
            'valid_embeddings': int(valid_text_embeddings),
            'embedding_dimension': int(text_embedding_dim),
            'success_rate': float(valid_text_embeddings / len(self.df))
        }
        
        self.logger.info(f"   Generated text embeddings: {valid_text_embeddings}/{len(self.df)}")
        self.logger.info(f"   Text embedding dimension: {text_embedding_dim}")
        
        if valid_text_embeddings > 0:
            sample_vec = self.df['reasoning_vector'].iloc[0]
            self.logger.info(f"   Sample text embedding stats: mean={sample_vec.mean():.4f}, std={sample_vec.std():.4f}")
    
    def _create_hybrid_vectors(self):
        """Step 3: Create hybrid vectors by concatenating structure and text embeddings."""
        self.logger.info("🔗 Step 3: Creating Hybrid Vectors...")
        
        hybrid_vectors = []
        valid_hybrid_count = 0
        
        for idx, row in self.df.iterrows():
            structure_vec = row.get('structure_vector')
            reasoning_vec = row.get('reasoning_vector')
            
            if (isinstance(structure_vec, np.ndarray) and 
                isinstance(reasoning_vec, np.ndarray) and
                len(structure_vec) == Config.STRUCTURE_VECTOR_DIM and
                len(reasoning_vec) == Config.TEXT_EMBEDDING_DIM):
                
                # Concatenate structure and reasoning vectors
                hybrid_vec = np.concatenate([structure_vec, reasoning_vec])
                hybrid_vectors.append(hybrid_vec)
                valid_hybrid_count += 1
            else:
                # Create zero vector for invalid cases
                hybrid_vec = np.zeros(Config.HYBRID_VECTOR_DIM)
                hybrid_vectors.append(hybrid_vec)
        
        self.hybrid_vectors = np.array(hybrid_vectors, dtype=np.float32)
        
        self.stats['hybrid_vectors'] = {
            'valid_hybrid_vectors': valid_hybrid_count,
            'hybrid_vector_dimension': Config.HYBRID_VECTOR_DIM,
            'success_rate': float(valid_hybrid_count / len(self.df))
        }
        
        self.logger.info(f"   Created hybrid vectors: {valid_hybrid_count}/{len(self.df)}")
        self.logger.info(f"   Hybrid vector dimension: {Config.HYBRID_VECTOR_DIM}")
        
        if valid_hybrid_count > 0:
            sample_hybrid = self.hybrid_vectors[0]
            self.logger.info(f"   Sample hybrid vector stats: mean={sample_hybrid.mean():.4f}, std={sample_hybrid.std():.4f}")
            self.logger.info(f"   Vector composition: {Config.STRUCTURE_VECTOR_DIM}D structure + {Config.TEXT_EMBEDDING_DIM}D text")
    
    def _build_faiss_index(self):
        """Step 4: Build FAISS index with hybrid vectors."""
        self.logger.info("🏗️ Step 4: Building FAISS Index...")
        
        if self.hybrid_vectors is None or len(self.hybrid_vectors) == 0:
            raise ValueError("No hybrid vectors available for indexing")
        
        # Prepare metadata for FAISS index
        ids = self.df['problem_id'].astype(str).tolist()
        
        metadata_list = []
        for _, row in self.df.iterrows():
            metadata = {
                'problem_text': row.get('problem_text', ''),
                'clean_text': row.get('clean_text', ''),
                'symbolic_fingerprint': row.get('symbolic_fingerprint', ''),
                'problem_type': row.get('problem_type', ''),
                'difficulty_level': row.get('difficulty_level', ''),
                'equation_count': row.get('equation_count', 0),
                'variable_count': row.get('variable_count', 0),
                'extracted_equations': row.get('extracted_equations', [])
            }
            metadata_list.append(metadata)
        
        # Create and build FAISS index
        self.faiss_indexer = FaissIndexer(
            dim=Config.HYBRID_VECTOR_DIM,
            index_type=Config.FAISS_INDEX_TYPE,
            normalize=Config.NORMALIZE_EMBEDDINGS,
            use_gpu=Config.USE_GPU
        )
        
        self.faiss_indexer.build(
            vectors=self.hybrid_vectors,
            ids=ids,
            metadata_list=metadata_list
        )
        
        self.stats['faiss_index'] = {
            'index_type': Config.FAISS_INDEX_TYPE,
            'vector_count': len(self.hybrid_vectors),
            'dimension': Config.HYBRID_VECTOR_DIM,
            'normalized': Config.NORMALIZE_EMBEDDINGS,
            'gpu_enabled': Config.USE_GPU
        }
        
        self.logger.info(f"   Built FAISS index with {len(self.hybrid_vectors)} vectors")
        self.logger.info(f"   Index type: {Config.FAISS_INDEX_TYPE}")
        self.logger.info(f"   Vector dimension: {Config.HYBRID_VECTOR_DIM}")
    
    def _evaluate_retrieval(self):
        """Step 5: Evaluate retrieval performance."""
        self.logger.info("📊 Step 5: Evaluating Retrieval Performance...")
        
        if self.faiss_indexer is None:
            self.logger.warning("No FAISS index available for evaluation")
            return
        
        # Create test queries from the dataset itself
        test_indices = np.random.choice(
            len(self.df), 
            size=min(20, len(self.df) // 5),  # 20 or 20% of data, whichever smaller
            replace=False
        )
        
        test_vectors = self.hybrid_vectors[test_indices]
        test_ids = self.df.iloc[test_indices]['problem_id'].astype(str).tolist()
        
        # Create evaluator
        id_to_problem = {
            str(row['problem_id']): row['problem_text'] 
            for _, row in self.df.iterrows()
        }
        
        evaluator = Evaluator(
            faiss_index=self.faiss_indexer.index,
            embeddings=self.hybrid_vectors,
            metadata_df=self.df,
            id_to_problem=id_to_problem
        )
        
        # Evaluate for different top_k values
        evaluation_results = {}
        for k in Config.TOP_K_VALUES:
            precision = evaluator.precision_at_k(test_vectors, test_indices, k=k)
            recall = evaluator.recall_at_k(test_vectors, test_indices, k=k)
            
            evaluation_results[f'precision@{k}'] = precision
            evaluation_results[f'recall@{k}'] = recall
            
            self.logger.info(f"   Precision@{k}: {precision:.3f}, Recall@{k}: {recall:.3f}")
        
        self.stats['evaluation'] = evaluation_results
        self.stats['evaluation_details'] = {
            'test_set_size': len(test_indices),
            'top_k_values': Config.TOP_K_VALUES
        }
        
        # Save failure analysis
        evaluator.analyze_failures(test_vectors, test_indices, k=5)
    
    def _save_system(self):
        """Step 6: Save the complete search system."""
        self.logger.info("💾 Step 6: Saving Search System...")
        
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save FAISS index and ID mapping
        index_path = f"{Config.OUTPUT_DIR}/faiss_index_{timestamp}.bin"
        idmap_path = f"{Config.OUTPUT_DIR}/faiss_id_map_{timestamp}.json"
        
        self.faiss_indexer.save(index_path, idmap_path)
        
        # Save embedding records
        embedding_records = build_embedding_records(
            self.df,
            id_col="problem_id",
            vector_col="reasoning_vector",
            fingerprint_col="symbolic_fingerprint",
            metadata_cols=["problem_text", "problem_type", "difficulty_level"]
        )
        
        embeddings_path = f"{Config.OUTPUT_DIR}/embedding_records_{timestamp}.jsonl"
        save_embeddings_jsonl(embedding_records, embeddings_path)
        
        # Save pipeline statistics
        stats_path = f"{Config.OUTPUT_DIR}/pipeline_stats_{timestamp}.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Save search system configuration
        system_config = {
            'timestamp': timestamp,
            'model_name': Config.MODEL_NAME,
            'faiss_index_type': Config.FAISS_INDEX_TYPE,
            'vector_dimensions': {
                'structure': Config.STRUCTURE_VECTOR_DIM,
                'text': Config.TEXT_EMBEDDING_DIM,
                'hybrid': Config.HYBRID_VECTOR_DIM
            },
            'file_paths': {
                'faiss_index': index_path,
                'id_mapping': idmap_path,
                'embedding_records': embeddings_path,
                'pipeline_stats': stats_path
            }
        }
        
        config_path = f"{Config.OUTPUT_DIR}/system_config_{timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump(system_config, f, indent=2)
        
        self.stats['output_files'] = system_config['file_paths']
        
        self.logger.info(f"   Saved FAISS index to: {index_path}")
        self.logger.info(f"   Saved ID mapping to: {idmap_path}")
        self.logger.info(f"   Saved embedding records to: {embeddings_path}")
        self.logger.info(f"   Saved system config to: {config_path}")
    
    def _print_summary(self):
        """Print a comprehensive summary of the pipeline execution."""
        self.logger.info("\n" + "="*60)
        self.logger.info("📈 EMBEDDING PIPELINE EXECUTION SUMMARY")
        self.logger.info("="*60)
        
        self.logger.info(f"📊 Dataset: {self.stats['loaded_count']} problems")
        
        if 'data_quality' in self.stats:
            dq = self.stats['data_quality']
            self.logger.info(f"🧪 Data Quality: {dq['structure_vector_success_rate']:.1%} structure vectors")
            self.logger.info(f"                  {dq['fingerprint_success_rate']:.1%} valid fingerprints")
        
        if 'text_embeddings' in self.stats:
            te = self.stats['text_embeddings']
            self.logger.info(f"🔤 Text Embeddings: {te['success_rate']:.1%} success rate")
        
        if 'hybrid_vectors' in self.stats:
            hv = self.stats['hybrid_vectors']
            self.logger.info(f"🔗 Hybrid Vectors: {hv['success_rate']:.1%} success rate")
        
        if 'faiss_index' in self.stats:
            fi = self.stats['faiss_index']
            self.logger.info(f"🏗️ FAISS Index: {fi['vector_count']} vectors, {fi['dimension']}D")
        
        if 'evaluation' in self.stats:
            eval_stats = self.stats['evaluation']
            self.logger.info("📊 Retrieval Performance:")
            for k in Config.TOP_K_VALUES:
                prec_key = f'precision@{k}'
                rec_key = f'recall@{k}'
                if prec_key in eval_stats and rec_key in eval_stats:
                    self.logger.info(f"   Top-{k}: P={eval_stats[prec_key]:.3f}, R={eval_stats[rec_key]:.3f}")
        
        if 'output_files' in self.stats:
            self.logger.info("💾 Output Files:")
            for file_type, file_path in self.stats['output_files'].items():
                self.logger.info(f"   - {file_type}: {file_path}")

def main():
    """Main function to run the embedding pipeline."""
    parser = argparse.ArgumentParser(description='Run the embedding pipeline')
    parser.add_argument('--input', type=str, help='Path to preprocessed CSV file')
    parser.add_argument('--model', type=str, default=Config.MODEL_NAME, help='SentenceTransformer model name')
    parser.add_argument('--index-type', type=str, default=Config.FAISS_INDEX_TYPE, 
                       choices=['Flat', 'HNSW', 'IVF'], help='FAISS index type')
    
    args = parser.parse_args()
    
    # Update config from command line
    if args.model:
        Config.MODEL_NAME = args.model
    if args.index_type:
        Config.FAISS_INDEX_TYPE = args.index_type
    
    try:
        pipeline = EmbeddingPipeline()
        faiss_indexer = pipeline.run(args.input)
        
        print("\n🎉 EMBEDDING PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"📁 Output files saved in: {Config.OUTPUT_DIR}")
        print(f"📊 Indexed {pipeline.stats['loaded_count']} problems")
        print(f"🔍 FAISS index ready for querying!")
        
        return faiss_indexer
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()