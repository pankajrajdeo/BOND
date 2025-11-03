#!/usr/bin/env python3
"""
Simple metadata downloader for CellxGene datasets.
Downloads H5AD for given dataset IDs, extracts metadata, saves as CSV.gz, and deletes H5AD.
"""

import os
import sys
import gc
import argparse
import anndata
import cellxgene_census
from tqdm import tqdm


def download_metadata(dataset_id: str, output_dir: str, census_version: str = "2025-01-30") -> bool:
    """
    Download H5AD for a dataset ID, extract metadata, save as CSV.gz, and delete H5AD.
    
    Args:
        dataset_id: The dataset ID to download
        output_dir: Directory to save the metadata CSV.gz file
        census_version: Census version to use
        
    Returns:
        True if successful, False otherwise
    """
    output_path = os.path.join(output_dir, f"{dataset_id}_metadata.csv.gz")
    tmp_h5ad = f"{dataset_id}.h5ad"
    
    # Skip if file already exists
    if os.path.exists(output_path):
        print(f"⏭️  Skipping {dataset_id}: file already exists")
        return True
    
    try:
        print(f"📥 Downloading {dataset_id}...")
        
        # Download H5AD file
        cellxgene_census.download_source_h5ad(dataset_id, to_path=tmp_h5ad)
        
        # Read H5AD and extract metadata
        print(f"📖 Reading metadata from {dataset_id}...")
        ad = anndata.read_h5ad(tmp_h5ad)
        
        # Save metadata as CSV.gz
        print(f"💾 Saving metadata to {output_path}...")
        ad.obs.to_csv(output_path, index=False, compression="gzip")
        
        print(f"✅ Successfully processed {dataset_id}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR processing {dataset_id}: {e}")
        return False
        
    finally:
        # Clean up
        try:
            if 'ad' in locals():
                del ad
            gc.collect()
        except Exception:
            pass
        
        try:
            if os.path.exists(tmp_h5ad):
                os.remove(tmp_h5ad)
                print(f"🗑️  Deleted temporary H5AD file for {dataset_id}")
        except Exception as e:
            print(f"⚠️  Warning: Could not delete temporary file {tmp_h5ad}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Simple metadata downloader for CellxGene datasets")
    parser.add_argument("--dataset-ids", required=True, help="File containing dataset IDs (one per line)")
    parser.add_argument("--output-dir", default="/Users/rajlq7/Downloads/Terms/BOND/Miscellaneous/CellxGene_Benchmark/Miscellaneous/CellxGene_Benchmark/benchmark_data/metadata_schemas_new", 
                       help="Output directory for metadata CSV.gz files")
    parser.add_argument("--census-version", default="2025-01-30", help="CellxGene Census version")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read dataset IDs
    try:
        with open(args.dataset_ids, 'r') as f:
            dataset_ids = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ Error: File {args.dataset_ids} not found")
        sys.exit(1)
    
    print(f"📋 Found {len(dataset_ids)} dataset IDs to process")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🔧 Census version: {args.census_version}")
    print("-" * 50)
    
    # Process each dataset ID
    successful = 0
    failed = 0
    
    for dataset_id in tqdm(dataset_ids, desc="Processing datasets", unit="dataset"):
        if download_metadata(dataset_id, args.output_dir, args.census_version):
            successful += 1
        else:
            failed += 1
    
    print("-" * 50)
    print(f"🎉 Download complete!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total processed: {len(dataset_ids)}")


if __name__ == "__main__":
    main()
