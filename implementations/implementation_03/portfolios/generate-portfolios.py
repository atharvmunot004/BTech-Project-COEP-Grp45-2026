"""
Generate 100,000 portfolios for 10 stocks using Dirichlet distribution.
Portfolios can have different sizes by assigning 0 weight to some stocks.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Stock symbols (10 stocks)
STOCKS = [
    "RELIANCE",
    "HDFCBANK",
    "TCS",
    "INFY",
    "ICICIBANK",
    "HINDUNILVR",
    "ASIANPAINT",
    "LT",
    "BAJFINANCE",
    "NTPC"
]

NUM_PORTFOLIOS = 100_000
NUM_STOCKS = len(STOCKS)


def generate_portfolios(num_portfolios=NUM_PORTFOLIOS, num_stocks=NUM_STOCKS, random_seed=42):
    """
    Generate portfolios with varying sizes using Dirichlet distribution.
    
    Parameters:
    -----------
    num_portfolios : int
        Number of portfolios to generate (default: 100,000)
    num_stocks : int
        Total number of stocks (default: 10)
    random_seed : int
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with shape (num_portfolios, num_stocks) where each row is a portfolio
    """
    np.random.seed(random_seed)
    
    # Initialize portfolio matrix
    portfolios = np.zeros((num_portfolios, num_stocks))
    
    for i in range(num_portfolios):
        # Randomly decide how many stocks to include in this portfolio (between 2 and 10)
        # This creates portfolios of different sizes
        num_active_stocks = np.random.randint(2, num_stocks + 1)
        
        # Randomly select which stocks to include
        active_indices = np.random.choice(num_stocks, size=num_active_stocks, replace=False)
        
        # Generate Dirichlet weights for active stocks
        # Using alpha=1.0 for uniform Dirichlet (can be adjusted for different concentration)
        alpha = np.ones(num_active_stocks)
        active_weights = np.random.dirichlet(alpha)
        
        # Assign weights to active stocks, leave others as 0
        portfolios[i, active_indices] = active_weights
    
    # Create DataFrame
    portfolios_df = pd.DataFrame(portfolios, columns=STOCKS)
    
    # Ensure weights sum to 1 (should already be true, but verify)
    assert np.allclose(portfolios_df.sum(axis=1), 1.0), "Portfolio weights must sum to 1"
    
    return portfolios_df


def calculate_statistics(portfolios_df):
    """
    Calculate comprehensive statistics for the generated portfolios.
    
    Parameters:
    -----------
    portfolios_df : pd.DataFrame
        DataFrame containing portfolio weights
    
    Returns:
    --------
    dict
        Dictionary containing all portfolio statistics
    """
    # Portfolio size statistics (number of non-zero weights)
    portfolio_sizes = portfolios_df.astype(bool).sum(axis=1)
    size_dist = portfolio_sizes.value_counts().sort_index()
    
    # Weight statistics per stock
    weight_stats = {}
    for stock in STOCKS:
        stock_weights = portfolios_df[stock]
        non_zero_weights = stock_weights[stock_weights > 0]
        weight_stats[stock] = {
            "mean": float(stock_weights.mean()),
            "std": float(stock_weights.std()),
            "min": float(stock_weights.min()),
            "max": float(stock_weights.max()),
            "median": float(stock_weights.median()),
            "non_zero_count": int((stock_weights > 0).sum()),
            "non_zero_percentage": float((stock_weights > 0).sum() / len(portfolios_df) * 100),
            "non_zero_mean": float(non_zero_weights.mean()) if len(non_zero_weights) > 0 else 0.0,
            "non_zero_std": float(non_zero_weights.std()) if len(non_zero_weights) > 0 else 0.0
        }
    
    # Overall statistics
    stats = {
        "dataset_name": "Portfolio Weights Dataset - 100,000 Portfolios",
        "description": "100,000 randomly generated portfolios for 10 Indian stocks using Dirichlet distribution. Portfolios have varying sizes (2-10 stocks) with some stocks assigned zero weight to create diverse portfolio configurations.",
        "generation_method": {
            "distribution": "Dirichlet",
            "alpha_parameter": 1.0,
            "description": "Uniform Dirichlet distribution (alpha=1.0) used to generate weights for active stocks in each portfolio"
        },
        "portfolio_configuration": {
            "total_portfolios": int(len(portfolios_df)),
            "total_stocks": NUM_STOCKS,
            "stocks": STOCKS,
            "min_portfolio_size": int(portfolio_sizes.min()),
            "max_portfolio_size": int(portfolio_sizes.max()),
            "mean_portfolio_size": float(portfolio_sizes.mean()),
            "median_portfolio_size": float(portfolio_sizes.median()),
            "std_portfolio_size": float(portfolio_sizes.std())
        },
        "portfolio_size_distribution": {
            str(int(size)): {
                "count": int(count),
                "percentage": float(count / len(portfolios_df) * 100)
            }
            for size, count in size_dist.items()
        },
        "weight_statistics": weight_stats,
        "overall_weight_statistics": {
            "mean": float(portfolios_df.values.mean()),
            "std": float(portfolios_df.values.std()),
            "min": float(portfolios_df.values.min()),
            "max": float(portfolios_df.values.max()),
            "median": float(np.median(portfolios_df.values)),
            "zero_weight_percentage": float((portfolios_df.values == 0).sum() / portfolios_df.size * 100),
            "non_zero_mean": float(portfolios_df.values[portfolios_df.values > 0].mean()) if (portfolios_df.values > 0).any() else 0.0
        },
        "data_structure": {
            "file_format": ["CSV", "Parquet"],
            "file_location": "portfolios/",
            "columns": [
                {
                    "name": stock,
                    "description": f"Weight allocation for {stock} in the portfolio",
                    "data_type": "float",
                    "range": [0.0, 1.0],
                    "constraint": "Sum of all weights in a portfolio equals 1.0"
                }
                for stock in STOCKS
            ],
            "rows": int(len(portfolios_df)),
            "constraints": [
                "Each portfolio's weights sum to 1.0",
                "All weights are non-negative (>= 0)",
                "Portfolios can have 2-10 active stocks (non-zero weights)"
            ]
        },
        "generation_parameters": {
            "random_seed": 42,
            "num_portfolios": NUM_PORTFOLIOS,
            "num_stocks": NUM_STOCKS,
            "min_active_stocks": 2,
            "max_active_stocks": NUM_STOCKS
        },
        "notes": [
            "Portfolios are generated using Dirichlet distribution with alpha=1.0 (uniform)",
            "Each portfolio randomly selects 2-10 stocks to include",
            "Weights for non-selected stocks are set to 0",
            "All portfolio weights sum to 1.0",
            "Suitable for portfolio optimization, risk analysis, and backtesting",
            "Portfolios represent diverse allocation strategies from concentrated to diversified"
        ],
        "metadata_version": "1.0",
        "last_updated": datetime.now().strftime("%Y-%m-%d")
    }
    
    return stats


def main():
    """Main function to generate and save portfolios."""
    print(f"Generating {NUM_PORTFOLIOS:,} portfolios for {NUM_STOCKS} stocks...")
    
    # Generate portfolios
    portfolios_df = generate_portfolios()
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_path = output_dir / "portfolios.csv"
    portfolios_df.to_csv(csv_path, index=False)
    print(f"Saved portfolios to: {csv_path}")
    
    # Save as Parquet (more efficient for large datasets)
    parquet_path = output_dir / "portfolios.parquet"
    portfolios_df.to_parquet(parquet_path, index=False)
    print(f"Saved portfolios to: {parquet_path}")
    
    # Calculate statistics
    print("\nCalculating portfolio statistics...")
    statistics = calculate_statistics(portfolios_df)
    
    # Save statistics to llm.json
    llm_json_path = output_dir / "llm.json"
    with open(llm_json_path, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    print(f"Saved statistics to: {llm_json_path}")
    
    # Print statistics
    print("\nPortfolio Statistics:")
    print(f"  Total portfolios: {statistics['portfolio_configuration']['total_portfolios']:,}")
    print(f"  Number of stocks: {statistics['portfolio_configuration']['total_stocks']}")
    print(f"  Average portfolio size (non-zero weights): {statistics['portfolio_configuration']['mean_portfolio_size']:.2f}")
    print(f"  Min portfolio size: {statistics['portfolio_configuration']['min_portfolio_size']}")
    print(f"  Max portfolio size: {statistics['portfolio_configuration']['max_portfolio_size']}")
    print(f"  Portfolio size distribution:")
    for size, info in statistics['portfolio_size_distribution'].items():
        print(f"    {size} stocks: {info['count']:,} portfolios ({info['percentage']:.1f}%)")
    
    print("\nSample portfolios (first 5):")
    print(portfolios_df.head())
    
    print("\nPortfolio weight statistics:")
    print(portfolios_df.describe())


if __name__ == "__main__":
    main()

