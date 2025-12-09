"""
Main script to process raw data and save cleaned data.
"""
import pandas as pd
from pathlib import Path
from typing import Optional, List
import json
from datetime import datetime

from .data_loader import DataLoader
from .data_cleaner import DataCleaner


def process_and_save(
    symbols: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    interpolation_method: str = 'time',
    drop_missing: bool = True,
    save_format: List[str] = ['parquet', 'csv']
):
    """
    Process raw data and save cleaned data to processed directory.
    
    Args:
        symbols: List of stock symbols to process. If None, processes all available stocks.
        output_dir: Directory to save processed data. If None, uses default processed directory.
        interpolation_method: Method for handling missing values.
        drop_missing: Whether to drop rows with missing values after interpolation.
        save_format: List of formats to save ('parquet', 'csv', or both).
    """
    # Initialize paths
    # __file__ is: implementation_03/src/data/process_data.py
    # We need to go up 3 levels to reach implementation_03
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent  # implementation_03
    
    if output_dir is None:
        output_dir = project_root / "data" / "processed"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("=" * 60)
    print("DATA PROCESSING PIPELINE")
    print("=" * 60)
    
    loader = DataLoader()
    
    if symbols is None:
        symbols = loader.get_available_symbols()
        print(f"\nProcessing all available stocks: {symbols}")
    else:
        print(f"\nProcessing specified stocks: {symbols}")
    
    print(f"\nLoading raw data from: {loader.raw_data_path}")
    data_dict = loader.load_multiple_stocks(symbols)
    
    # Clean data
    cleaner = DataCleaner(
        interpolation_method=interpolation_method,
        drop_missing=drop_missing
    )
    cleaned_data = cleaner.clean(data_dict)
    
    # Save individual stock files
    print("\n" + "=" * 60)
    print("SAVING PROCESSED DATA")
    print("=" * 60)
    
    for symbol, df in cleaned_data.items():
        print(f"\nSaving {symbol}...")
        
        if 'parquet' in save_format:
            parquet_path = output_dir / f"{symbol}_processed.parquet"
            df.to_parquet(parquet_path, index=True)
            print(f"  Saved: {parquet_path}")
        
        if 'csv' in save_format:
            csv_path = output_dir / f"{symbol}_processed.csv"
            df.to_csv(csv_path, index=True)
            print(f"  Saved: {csv_path}")
    
    # Save combined panel data (wide format: Date x Symbol for each column)
    print("\nSaving combined panel data...")
    
    panel_data = {}
    for col in ['Open', 'High', 'Low', 'Close', 'Price', 'Volume']:
        if col in list(cleaned_data.values())[0].columns:
            panel_df = pd.DataFrame({
                symbol: df[col] for symbol, df in cleaned_data.items()
            })
            panel_data[col] = panel_df
            
            if 'parquet' in save_format:
                panel_path = output_dir / f"panel_{col.lower()}.parquet"
                panel_df.to_parquet(panel_path, index=True)
                print(f"  Saved panel {col}: {panel_path}")
            
            if 'csv' in save_format:
                panel_path = output_dir / f"panel_{col.lower()}.csv"
                panel_df.to_csv(panel_path, index=True)
                print(f"  Saved panel {col}: {panel_path}")
    
    # Save metadata
    metadata = {
        "dataset_name": "Processed Stock Data - 10 Indian Stocks",
        "description": "Cleaned and aligned daily stock price data for 10 major Indian companies",
        "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stocks": list(cleaned_data.keys()),
        "date_range": {
            "start_date": str(list(cleaned_data.values())[0].index.min()),
            "end_date": str(list(cleaned_data.values())[0].index.max()),
            "total_trading_days": len(list(cleaned_data.values())[0])
        },
        "processing_parameters": {
            "interpolation_method": interpolation_method,
            "drop_missing": drop_missing
        },
        "data_structure": {
            "format": "Panel data (Date x Symbol)",
            "columns": list(list(cleaned_data.values())[0].columns),
            "files": {
                "individual_stocks": [f"{symbol}_processed.parquet" for symbol in cleaned_data.keys()],
                "panel_data": [f"panel_{col.lower()}.parquet" for col in panel_data.keys()]
            }
        },
        "statistics": {}
    }
    
    # Add statistics for each stock
    for symbol, df in cleaned_data.items():
        metadata["statistics"][symbol] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "date_range": {
                "start": str(df.index.min()),
                "end": str(df.index.max())
            },
            "summary_stats": {}
        }
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                metadata["statistics"][symbol]["summary_stats"][col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "median": float(df[col].median())
                }
    
    metadata_path = output_dir / "processing_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\nSaved metadata: {metadata_path}")
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nProcessed data saved to: {output_dir}")
    print(f"Total stocks processed: {len(cleaned_data)}")
    print(f"Total trading days: {len(list(cleaned_data.values())[0])}")
    
    return cleaned_data, metadata


def main():
    """Main entry point for data processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process raw stock data')
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Stock symbols to process (e.g., RELIANCE TCS). If not specified, processes all available stocks.'
    )
    parser.add_argument(
        '--interpolation',
        choices=['time', 'linear', 'forward_fill', 'backward_fill'],
        default='time',
        help='Interpolation method for missing values'
    )
    parser.add_argument(
        '--keep-missing',
        action='store_true',
        help='Keep rows with missing values (default: drop them)'
    )
    parser.add_argument(
        '--format',
        nargs='+',
        choices=['parquet', 'csv'],
        default=['parquet', 'csv'],
        help='Output formats to save'
    )
    
    args = parser.parse_args()
    
    process_and_save(
        symbols=args.symbols,
        interpolation_method=args.interpolation,
        drop_missing=not args.keep_missing,
        save_format=args.format
    )


if __name__ == "__main__":
    main()

