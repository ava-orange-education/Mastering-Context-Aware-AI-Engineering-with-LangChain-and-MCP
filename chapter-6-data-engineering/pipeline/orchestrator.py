"""
Pipeline orchestration with Apache Airflow.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrate data engineering pipeline"""
    
    def __init__(self, pipeline_config: Dict[str, Any]):
        """
        Initialize pipeline orchestrator
        
        Args:
            pipeline_config: Pipeline configuration
        """
        self.config = pipeline_config
        self.pipeline_id = pipeline_config.get('pipeline_id', 'data_pipeline')
        
        # Pipeline components
        self.ingestion_connectors = []
        self.preprocessing_steps = []
        self.embedding_manager = None
        self.vector_store = None
        self.graph_builder = None
    
    def register_ingestion_connector(self, connector):
        """Register data ingestion connector"""
        self.ingestion_connectors.append(connector)
        logger.info(f"Registered ingestion connector: {connector.source.source_id}")
    
    def register_preprocessing_step(self, step):
        """Register preprocessing step"""
        self.preprocessing_steps.append(step)
        logger.info(f"Registered preprocessing step")
    
    def set_embedding_manager(self, embedding_manager):
        """Set embedding manager"""
        self.embedding_manager = embedding_manager
        logger.info("Set embedding manager")
    
    def set_vector_store(self, vector_store):
        """Set vector store"""
        self.vector_store = vector_store
        logger.info("Set vector store")
    
    def set_graph_builder(self, graph_builder):
        """Set graph builder"""
        self.graph_builder = graph_builder
        logger.info("Set graph builder")
    
    def run_pipeline(self, checkpoint_store: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute complete pipeline
        
        Args:
            checkpoint_store: Checkpoint data for incremental runs
            
        Returns:
            Pipeline execution statistics
        """
        checkpoint_store = checkpoint_store or {}
        
        stats = {
            'start_time': datetime.now().isoformat(),
            'records_ingested': 0,
            'records_processed': 0,
            'embeddings_generated': 0,
            'vectors_stored': 0,
            'entities_extracted': 0,
            'relations_extracted': 0,
            'errors': []
        }
        
        try:
            # Step 1: Data Ingestion
            logger.info("Starting data ingestion...")
            ingested_records = self._run_ingestion(checkpoint_store)
            stats['records_ingested'] = len(ingested_records)
            
            # Step 2: Preprocessing
            logger.info("Starting preprocessing...")
            processed_records = self._run_preprocessing(ingested_records)
            stats['records_processed'] = len(processed_records)
            
            # Step 3: Embedding Generation
            if self.embedding_manager:
                logger.info("Generating embeddings...")
                embedded_records = self._run_embedding(processed_records)
                stats['embeddings_generated'] = len(embedded_records)
            else:
                embedded_records = processed_records
            
            # Step 4: Vector Storage
            if self.vector_store and self.embedding_manager:
                logger.info("Storing vectors...")
                self._run_vector_storage(embedded_records)
                stats['vectors_stored'] = len(embedded_records)
            
            # Step 5: Knowledge Graph Construction
            if self.graph_builder:
                logger.info("Building knowledge graph...")
                graph_stats = self._run_graph_construction(processed_records)
                stats.update(graph_stats)
            
            stats['end_time'] = datetime.now().isoformat()
            stats['status'] = 'success'
            
            logger.info(f"Pipeline completed successfully: {stats}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            stats['status'] = 'failed'
            stats['errors'].append(str(e))
        
        return stats
    
    def _run_ingestion(self, checkpoint_store: Dict) -> List[Dict[str, Any]]:
        """Run data ingestion"""
        all_records = []
        
        for connector in self.ingestion_connectors:
            try:
                checkpoint = checkpoint_store.get(connector.source.source_id)
                
                if not connector.connect():
                    logger.error(f"Failed to connect: {connector.source.source_id}")
                    continue
                
                for record in connector.extract(checkpoint):
                    all_records.append({
                        'record_id': record.record_id,
                        'source_id': record.source_id,
                        'content': record.content,
                        'metadata': record.metadata,
                        'timestamp': record.timestamp,
                        'checksum': record.checksum
                    })
                
                # Update checkpoint
                checkpoint_store[connector.source.source_id] = connector.get_checkpoint()
                
                connector.close()
                
            except Exception as e:
                logger.error(f"Ingestion error for {connector.source.source_id}: {e}")
        
        return all_records
    
    def _run_preprocessing(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run preprocessing steps"""
        processed_records = []
        
        for record in records:
            try:
                processed_content = record['content']
                
                # Apply each preprocessing step
                for step in self.preprocessing_steps:
                    processed_content = step.process(processed_content)
                
                processed_records.append({
                    **record,
                    'processed_content': processed_content
                })
                
            except Exception as e:
                logger.error(f"Preprocessing error for record {record['record_id']}: {e}")
        
        return processed_records
    
    def _run_embedding(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings"""
        embedded_records = []
        
        # Extract texts to embed
        texts = []
        for record in records:
            content = record.get('processed_content', record.get('content', ''))
            
            if isinstance(content, dict):
                text = content.get('full_text', content.get('content', ''))
            else:
                text = str(content)
            
            texts.append(text)
        
        # Generate embeddings in batch
        if texts:
            try:
                results = self.embedding_manager.embed_batch(texts)
                
                for record, result in zip(records, results):
                    embedded_records.append({
                        **record,
                        'embedding': result.embedding,
                        'embedding_model': result.model_name
                    })
            
            except Exception as e:
                logger.error(f"Embedding generation error: {e}")
        
        return embedded_records
    
    def _run_vector_storage(self, records: List[Dict[str, Any]]):
        """Store vectors in vector database"""
        vectors = []
        metadata = []
        ids = []
        
        for record in records:
            if 'embedding' in record:
                vectors.append(record['embedding'])
                
                # Prepare metadata (exclude embedding)
                meta = {
                    'record_id': record['record_id'],
                    'source_id': record['source_id'],
                    'text': str(record.get('processed_content', record.get('content', ''))),
                    **record.get('metadata', {})
                }
                metadata.append(meta)
                ids.append(record['record_id'])
        
        if vectors:
            self.vector_store.add_vectors(vectors, metadata, ids)
    
    def _run_graph_construction(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build knowledge graph"""
        documents = []
        
        for record in records:
            content = record.get('processed_content', record.get('content', ''))
            
            if isinstance(content, dict):
                text = content.get('full_text', content.get('content', ''))
            else:
                text = str(content)
            
            documents.append({
                'id': record['record_id'],
                'content': text,
                'metadata': record.get('metadata', {})
            })
        
        graph = self.graph_builder.build_from_documents(documents)
        
        return {
            'entities_extracted': len(graph.entities),
            'relations_extracted': len(graph.relations)
        }


class AirflowDAGGenerator:
    """Generate Airflow DAG from pipeline configuration"""
    
    def __init__(self, pipeline_config: Dict[str, Any]):
        """
        Initialize DAG generator
        
        Args:
            pipeline_config: Pipeline configuration
        """
        self.config = pipeline_config
    
    def generate_dag(self) -> str:
        """
        Generate Airflow DAG Python code
        
        Returns:
            DAG code as string
        """
        dag_id = self.config.get('pipeline_id', 'data_pipeline')
        schedule = self.config.get('schedule', '@daily')
        
        dag_code = f"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {{
    'owner': 'data_engineering',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}}

dag = DAG(
    '{dag_id}',
    default_args=default_args,
    description='Data engineering pipeline',
    schedule_interval='{schedule}',
    catchup=False,
)

def ingest_data(**context):
    from pipeline.orchestrator import PipelineOrchestrator
    # Ingestion logic here
    pass

def preprocess_data(**context):
    # Preprocessing logic here
    pass

def generate_embeddings(**context):
    # Embedding generation logic here
    pass

def store_vectors(**context):
    # Vector storage logic here
    pass

def build_graph(**context):
    # Graph construction logic here
    pass

# Define tasks
ingest_task = PythonOperator(
    task_id='ingest_data',
    python_callable=ingest_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

embed_task = PythonOperator(
    task_id='generate_embeddings',
    python_callable=generate_embeddings,
    dag=dag,
)

store_task = PythonOperator(
    task_id='store_vectors',
    python_callable=store_vectors,
    dag=dag,
)

graph_task = PythonOperator(
    task_id='build_graph',
    python_callable=build_graph,
    dag=dag,
)

# Define task dependencies
ingest_task >> preprocess_task >> embed_task >> store_task
preprocess_task >> graph_task
"""
        
        return dag_code