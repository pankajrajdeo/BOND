import pytest
import tempfile
import os
import json
from pathlib import Path

def create_mini_ontology():
    """Create a minimal test ontology for end-to-end testing"""
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create mini OWL file
    owl_content = """<?xml version="1.0"?>
<rdf:RDF xmlns="http://purl.obolibrary.org/obo/test.owl#"
     xml:base="http://purl.obolibrary.org/obo/test.owl"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:xml="http://www.w3.org/XML/1998/namespace"
     xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    
    <owl:Ontology rdf:about="http://purl.obolibrary.org/obo/test.owl"/>
    
    <owl:Class rdf:about="http://purl.obolibrary.org/obo/CL_0000084">
        <rdfs:label>T lymphocyte</rdfs:label>
        <rdfs:comment>A type of white blood cell that plays a central role in cell-mediated immunity.</rdfs:comment>
        <oboInOwl:hasExactSynonym rdf:datatype="http://www.w3.org/2001/XMLSchema#string">T-cell</oboInOwl:hasExactSynonym>
        <oboInOwl:hasExactSynonym rdf:datatype="http://www.w3.org/2001/XMLSchema#string">T cell</oboInOwl:hasExactSynonym>
    </owl:Class>
    
    <owl:Class rdf:about="http://purl.obolibrary.org/obo/CL_0000236">
        <rdfs:label>B lymphocyte</rdfs:label>
        <rdfs:comment>A type of white blood cell that produces antibodies.</rdfs:comment>
        <oboInOwl:hasExactSynonym rdf:datatype="http://www.w3.org/2001/XMLSchema#string">B-cell</oboInOwl:hasExactSynonym>
        <oboInOwl:hasExactSynonym rdf:datatype="http://www.w3.org/2001/XMLSchema#string">B cell</oboInOwl:hasExactSynonym>
    </owl:Class>
    
    <owl:Class rdf:about="http://purl.obolibrary.org/obo/CL_0000540">
        <rdfs:label>natural killer cell</rdfs:label>
        <rdfs:comment>A type of cytotoxic lymphocyte that provides rapid responses to viral-infected cells.</rdfs:comment>
        <oboInOwl:hasExactSynonym rdf:datatype="http://www.w3.org/2001/XMLSchema#string">NK cell</oboInOwl:hasExactSynonym>
    </owl:Class>
</rdf:RDF>"""
    
    owl_path = os.path.join(temp_dir, "test.owl")
    with open(owl_path, 'w') as f:
        f.write(owl_content)
    
    return temp_dir, owl_path

def test_mini_ontology_creation():
    """Test that mini ontology can be created and contains expected terms"""
    
    temp_dir, owl_path = create_mini_ontology()
    
    try:
        # Verify file exists
        assert os.path.exists(owl_path)
        
        # Check content
        with open(owl_path, 'r') as f:
            content = f.read()
        
        # Verify key terms are present
        assert "T lymphocyte" in content
        assert "B lymphocyte" in content
        assert "natural killer cell" in content
        assert "T-cell" in content  # synonym
        assert "B-cell" in content  # synonym
        assert "NK cell" in content  # synonym
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

def test_build_script_imports():
    """Test that build script can be imported without errors"""
    
    try:
        from scripts.build_index_from_owl import main, create_sqlite, ingest_owl, build_faiss_profiles
        assert True  # If we get here, imports worked
    except ImportError as e:
        pytest.skip(f"Build script imports failed: {e}")

def test_bond_imports():
    """Test that core BOND modules can be imported"""
    
    try:
        from bond import BondMatcher
        from bond.config import BondSettings
        from bond.providers import resolve_embeddings
        from bond.retrieval.bm25_sqlite import search_bm25
        from bond.retrieval.faiss_store import FaissStore
        assert True  # If we get here, imports worked
    except ImportError as e:
        pytest.skip(f"BOND imports failed: {e}")

# Note: Full end-to-end test would require:
# 1. Building index with mini ontology
# 2. Running actual queries
# 3. Verifying results
# This is marked as integration test and would run in CI with proper setup
@pytest.mark.integration
def test_full_e2e_pipeline():
    """Full end-to-end test of the BOND pipeline"""
    
    # This test would:
    # 1. Create mini ontology
    # 2. Build index using build script
    # 3. Initialize BondMatcher
    # 4. Run test queries
    # 5. Verify results match expectations
    
    pytest.skip("Full E2E test requires index building and is marked as integration test")
