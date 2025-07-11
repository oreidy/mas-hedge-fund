#!/usr/bin/env python3
"""
Script to analyze the company news database and provide statistics.
"""

import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

def parse_time_published(time_str):
    """Parse Alpha Vantage time format to datetime object."""
    try:
        # Format: 20250304T141511
        return datetime.strptime(time_str, "%Y%m%dT%H%M%S")
    except:
        try:
            # Try alternative format without seconds
            return datetime.strptime(time_str, "%Y%m%dT%H%M")
        except:
            return None

def analyze_ticker_file(ticker_file):
    """Analyze a single ticker's news file."""
    try:
        with open(ticker_file, 'r') as f:
            data = json.load(f)
        
        ticker = ticker_file.stem.replace('_news', '')
        
        # Get basic info
        total_articles = int(data.get('items', 0))
        
        # Analyze articles
        articles = data.get('feed', [])
        article_count = len(articles)
        
        if articles:
            # Get dates
            dates = []
            for article in articles:
                time_pub = article.get('time_published', '')
                if time_pub:
                    parsed_date = parse_time_published(time_pub)
                    if parsed_date:
                        dates.append(parsed_date)
            
            if dates:
                first_date = min(dates)
                last_date = max(dates)
                date_span_days = (last_date - first_date).days
            else:
                first_date = last_date = None
                date_span_days = 0
            
            # Analyze sentiment
            sentiment_scores = []
            for article in articles:
                ticker_sentiments = article.get('ticker_sentiment', [])
                for ts in ticker_sentiments:
                    if ts.get('ticker') == ticker:
                        score = ts.get('ticker_sentiment_score')
                        if score is not None:
                            try:
                                sentiment_scores.append(float(score))
                            except:
                                pass
            
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else None
            
            return {
                'ticker': ticker,
                'total_articles': total_articles,
                'actual_articles': article_count,
                'first_date': first_date,
                'last_date': last_date,
                'date_span_days': date_span_days,
                'avg_sentiment': avg_sentiment,
                'sentiment_count': len(sentiment_scores)
            }
        else:
            return {
                'ticker': ticker,
                'total_articles': total_articles,
                'actual_articles': 0,
                'first_date': None,
                'last_date': None,
                'date_span_days': 0,
                'avg_sentiment': None,
                'sentiment_count': 0
            }
    
    except Exception as e:
        print(f"Error analyzing {ticker_file}: {e}")
        return None

def main():
    print("Analyzing Company News Database")
    print("=" * 50)
    
    # Find all news files
    cache_dir = Path("cache/company_news")
    if not cache_dir.exists():
        print(f"Cache directory {cache_dir} does not exist!")
        return
    
    news_files = list(cache_dir.glob("*_news.json"))
    print(f"Found {len(news_files)} ticker files in database\n")
    
    if not news_files:
        print("No news files found!")
        return
    
    # Analyze each file
    results = []
    for news_file in news_files:
        result = analyze_ticker_file(news_file)
        if result:
            results.append(result)
    
    # Sort by ticker
    results.sort(key=lambda x: x['ticker'])
    
    # Create summary table
    print("TICKER SUMMARY")
    print("-" * 90)
    print(f"{'Ticker':<8} {'Articles':<8} {'First Date':<12} {'Last Date':<12} {'Span (Days)':<12} {'Avg Sentiment':<12}")
    print("-" * 90)
    
    total_articles = 0
    tickers_with_data = 0
    tickers_no_data = 0
    all_first_dates = []
    all_last_dates = []
    all_sentiments = []
    
    for result in results:
        ticker = result['ticker']
        articles = result['actual_articles']
        first_date = result['first_date']
        last_date = result['last_date']
        span_days = result['date_span_days']
        avg_sentiment = result['avg_sentiment']
        
        total_articles += articles
        
        if articles > 0:
            tickers_with_data += 1
            if first_date:
                all_first_dates.append(first_date)
            if last_date:
                all_last_dates.append(last_date)
            if avg_sentiment is not None:
                all_sentiments.append(avg_sentiment)
        else:
            tickers_no_data += 1
        
        # Format dates
        first_str = first_date.strftime("%Y-%m-%d") if first_date else "No data"
        last_str = last_date.strftime("%Y-%m-%d") if last_date else "No data"
        sentiment_str = f"{avg_sentiment:.3f}" if avg_sentiment is not None else "No data"
        
        print(f"{ticker:<8} {articles:<8} {first_str:<12} {last_str:<12} {span_days:<12} {sentiment_str:<12}")
    
    # Overall statistics
    print("\n" + "=" * 50)
    print("OVERALL STATISTICS")
    print("=" * 50)
    print(f"Total tickers in database: {len(results)}")
    print(f"Tickers with news data: {tickers_with_data}")
    print(f"Tickers with no data: {tickers_no_data}")
    print(f"Total articles: {total_articles:,}")
    print(f"Average articles per ticker (with data): {total_articles/tickers_with_data:.1f}" if tickers_with_data > 0 else "N/A")
    
    if all_first_dates:
        earliest_date = min(all_first_dates)
        latest_date = max(all_last_dates) if all_last_dates else None
        print(f"Earliest article date: {earliest_date.strftime('%Y-%m-%d')}")
        print(f"Latest article date: {latest_date.strftime('%Y-%m-%d')}" if latest_date else "N/A")
        if latest_date:
            total_span = (latest_date - earliest_date).days
            print(f"Total date span: {total_span} days ({total_span/365:.1f} years)")
    
    if all_sentiments:
        avg_sentiment_overall = sum(all_sentiments) / len(all_sentiments)
        print(f"Average sentiment across all tickers: {avg_sentiment_overall:.3f}")
    
    # Top 10 tickers by article count
    print("\n" + "=" * 50)
    print("TOP 10 TICKERS BY ARTICLE COUNT")
    print("=" * 50)
    top_tickers = sorted([r for r in results if r['actual_articles'] > 0], 
                        key=lambda x: x['actual_articles'], reverse=True)[:10]
    
    for i, result in enumerate(top_tickers, 1):
        print(f"{i:2d}. {result['ticker']:<8} {result['actual_articles']:,} articles")
    
    # Tickers with no data
    no_data_tickers = [r['ticker'] for r in results if r['actual_articles'] == 0]
    if no_data_tickers:
        print(f"\nTickers with no data ({len(no_data_tickers)}):")
        print(", ".join(no_data_tickers))
    
    # Date range analysis
    print("\n" + "=" * 50)
    print("DATE RANGE ANALYSIS")
    print("=" * 50)
    
    span_ranges = {
        "< 1 year": 0,
        "1-2 years": 0,
        "2-3 years": 0,
        "3-4 years": 0,
        "4+ years": 0
    }
    
    for result in results:
        if result['date_span_days'] > 0:
            days = result['date_span_days']
            if days < 365:
                span_ranges["< 1 year"] += 1
            elif days < 730:
                span_ranges["1-2 years"] += 1
            elif days < 1095:
                span_ranges["2-3 years"] += 1
            elif days < 1460:
                span_ranges["3-4 years"] += 1
            else:
                span_ranges["4+ years"] += 1
    
    for range_name, count in span_ranges.items():
        print(f"{range_name}: {count} tickers")

if __name__ == "__main__":
    main()