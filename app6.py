import streamlit as st
import requests
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from duckduckgo_search import DDGS
import random
# from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pytrends.request import TrendReq
from bs4 import BeautifulSoup
import re,os
from collections import Counter
from textblob import TextBlob
from dotenv import load_dotenv
load_dotenv()
import logging
logging.basicConfig(level=logging.DEBUG,  # Set the logging level
                    format='%(asctime)s - %(levelname)s - %(message)s')  # Set the format of the log messages
# Download NLTK resources (first-time only)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

# Set your OpenRouter API Key (don't prompt user)
OPENROUTER_API_KEY = st.secrets["API_KEY"]  # Replace with environment variable in production

# Add this to your imports
# from serpapi import GoogleSearch
import time

def search_web(query, max_results=200):
    try:
        # Primary search using DDGS
        results = []
        with DDGS() as ddgs:
            try:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "body": r.get("body", ""),
                        "href": r.get("href", "")
                    })
            except Exception as e:
                logging.debug(f"Error during DDGS search: {e}")

                # Continue with fallback if DDGS fails
        
        # If DDGS returned results, use them
        if results:
            logging.debug(f'RESULTS FROM PRIMARY ENGINE (DDGS): {len(results)} found')
            return results
        
        # If DDGS returned no results, try fallback
        logging.debug('DDGS RETURNED NO RESULTS, TRYING FALLBACK')
        return fallback_search(query, max_results)
        
    except Exception as e:
        logging.debug(f"Primary search error: {e}")
        logging.debug('TRYING FALLBACK SEARCH ENGINE')
        return fallback_search(query, max_results)

# Improved fallback search
def fallback_search(query, max_results=200):
    try:
        results = []
        # Using requests with custom headers to get regular search results
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Try multiple search engines
        search_urls = [
            f"https://www.google.com/search?q={query.replace(' ', '+')}&num=100",
            f"https://www.bing.com/search?q={query.replace(' ', '+')}&count=100",
            f"https://search.yahoo.com/search?p={query.replace(' ', '+')}&n=100"
        ]
        
        for search_url in search_urls:
            try:
                response = requests.get(search_url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    # Simple regex-based extraction
                    title_pattern = r'<h3[^>]*>(.*?)</h3>'
                    title_matches = re.findall(title_pattern, response.text)
                    
                    snippet_pattern = r'<div class="[^"]*"[^>]*>(.*?)</div>'
                    snippet_matches = re.findall(snippet_pattern, response.text)
                    
                    url_pattern = r'<a href="(https?://[^"]+)"'
                    url_matches = re.findall(url_pattern, response.text)
                    
                    # Extract and clean at least some results
                    for i in range(min(len(title_matches), len(snippet_matches), len(url_matches), max_results)):
                        # Clean HTML tags from the results
                        title = re.sub(r'<.*?>', '', title_matches[i])
                        body = re.sub(r'<.*?>', '', snippet_matches[i])
                        href = url_matches[i]
                        
                        # Only add if we have meaningful content
                        if len(title.strip()) > 0 and len(body.strip()) > 0:
                            results.append({
                                "title": title,
                                "body": body,
                                "href": href
                            })
                    
                    # If we got at least some results, break the loop
                    if len(results) > 10:
                        break
            except Exception as e:
                logging.debug(f"Error with search engine {search_url}: {e}")
                continue
        
        # If all search engines failed, return a helpful placeholder result
        if not results:
            logging.debug("All search engines failed, returning placeholder result")
            return [{
                "title": f"Information about {query}",
                "body": f"We're collecting information about {query}. Try refining your search or check back in a moment.",
                "href": "https://www.example.com"
            }]
        
        logging.debug(f'RESULTS FROM FALLBACK ENGINE: {len(results)} found')
        return results
        
    except Exception as e:
        logging.debug(f"All fallback search methods failed: {e}")
        # Return a minimal placeholder to avoid breaking the application
        return [{
            "title": f"Information about {query}",
            "body": f"We're collecting information about {query}. Try refining your search or check back in a moment.",
            "href": "https://www.example.com"
        }]

# Improved analyze_keyword_structured function with better JSON parsing
def analyze_keyword_structured(keyword, search_results):
    # Combine search result content for context
    context = "\n".join([f"Title: {result['title']}\nContent: {result['body']}" for result in search_results[:15]])
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""
    Analyze the keyword '{keyword}' based on these search results:
    
    {context}
    
    Provide a structured analysis with these sections:
    1. TRENDING: What's currently trending about this keyword?
    2. POSITIVE: Key positive aspects or reviews
    3. NEGATIVE: Key negative aspects or criticisms
    4. CURRENT_TOPICS: Current discussions or news
    5. KEY_METRICS: Any numbers, statistics, or metrics mentioned (like prices, ratings, etc.)
    6. SENTIMENT_SCORE: Overall sentiment on a scale of -10 (very negative) to +10 (very positive)
    
    Format your response as JSON:
    {{
        "trending": ["trend1", "trend2", "trend3"],
        "positive_aspects": ["positive1", "positive2", "positive3"],
        "negative_aspects": ["negative1", "negative2", "negative3"],
        "current_topics": ["topic1", "topic2", "topic3"],
        "key_metrics": {{"metric_name1": "value1", "metric_name2": "value2"}},
        "sentiment_score": 5,
        "summary": "Brief summary of overall findings."
    }}
    
    IMPORTANT: Return ONLY valid JSON without any markdown formatting, code blocks, or explanations. Do not include ```json at the beginning or ``` at the end. Only return a valid JSON object.
    """
    
    payload = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": [
            {"role": "system", "content": "You are an expert analyst who provides structured JSON outputs. Return ONLY valid JSON without any markdown formatting, code blocks, or explanations."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    }
    
    # Create default fallback response
    default_response = {
        "trending": ["Analysis in progress..."],
        "positive_aspects": ["Analysis in progress..."],
        "negative_aspects": ["Analysis in progress..."],
        "current_topics": ["Analysis in progress..."],
        "key_metrics": {},
        "sentiment_score": 0,
        "summary": f"Analysis of '{keyword}' is being processed. Results will appear shortly."
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        logging.debug('STATUS CODE', response.status_code)
        
        if response.status_code == 200:
            try:
                # First try the proper API response format
                content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "{}")
                logging.debug('RAW CONTENT:', content[:500])
                
                # Try multiple approaches to parse the JSON
                try:
                    # Method 1: Direct parsing
                    result = json.loads(content)
                    
                    # Verify we have the expected fields
                    if isinstance(result, dict) and "trending" in result and "positive_aspects" in result:
                        return result
                    else:
                        # Missing expected fields, fall back to default
                        logging.debug("Response missing expected fields, using default")
                        return default_response
                        
                except json.JSONDecodeError:
                    # Method 2: Clean and try again
                    # Remove any possible markdown code blocks, backslashes, and other problematic characters
                    cleaned_content = re.sub(r'^```json\s*|\s*```$', '', content.strip())
                    cleaned_content = re.sub(r'\\([^\\])', r'\1', cleaned_content)
                    
                    try:
                        result = json.loads(cleaned_content)
                        if isinstance(result, dict) and "trending" in result:
                            return result
                    except json.JSONDecodeError:
                        # Method 3: More aggressive regex extraction
                        pattern = r'({[\s\S]*})'
                        matches = re.findall(pattern, cleaned_content)
                        if matches:
                            for potential_json in matches:
                                try:
                                    result = json.loads(potential_json)
                                    if isinstance(result, dict) and "trending" in result:
                                        return result
                                except:
                                    continue
                        
                        # Method 4: Fall back to manual parsing if all else fails
                        try:
                            # Extract data manually using regex patterns
                            manual_result = default_response.copy()
                            
                            # Try to extract trending items
                            trending_match = re.search(r'"trending"\s*:\s*\[(.*?)\]', cleaned_content, re.DOTALL)
                            if trending_match:
                                trending_items = re.findall(r'"([^"]+)"', trending_match.group(1))
                                if trending_items:
                                    manual_result["trending"] = trending_items
                            
                            # Extract positive aspects
                            positive_match = re.search(r'"positive_aspects"\s*:\s*\[(.*?)\]', cleaned_content, re.DOTALL)
                            if positive_match:
                                positive_items = re.findall(r'"([^"]+)"', positive_match.group(1))
                                if positive_items:
                                    manual_result["positive_aspects"] = positive_items
                            
                            # Extract negative aspects
                            negative_match = re.search(r'"negative_aspects"\s*:\s*\[(.*?)\]', cleaned_content, re.DOTALL)
                            if negative_match:
                                negative_items = re.findall(r'"([^"]+)"', negative_match.group(1))
                                if negative_items:
                                    manual_result["negative_aspects"] = negative_items
                            
                            # Extract current topics
                            topics_match = re.search(r'"current_topics"\s*:\s*\[(.*?)\]', cleaned_content, re.DOTALL)
                            if topics_match:
                                topic_items = re.findall(r'"([^"]+)"', topics_match.group(1))
                                if topic_items:
                                    manual_result["current_topics"] = topic_items
                            
                            # Extract sentiment score
                            sentiment_match = re.search(r'"sentiment_score"\s*:\s*(-?\d+)', cleaned_content)
                            if sentiment_match:
                                manual_result["sentiment_score"] = int(sentiment_match.group(1))
                            
                            # Extract summary
                            summary_match = re.search(r'"summary"\s*:\s*"([^"]+)"', cleaned_content)
                            if summary_match:
                                manual_result["summary"] = summary_match.group(1)
                            
                            return manual_result
                        except Exception as e:
                            logging.debug(f"Manual parsing failed: {e}")
                            return default_response
                    
                # If we get here, all parsing methods failed
                logging.debug("All JSON parsing methods failed, using default response")
                return default_response
            except Exception as e:
                logging.debug(f"Response processing error: {e}")
                return default_response
        else:
            logging.debug(f"API Error: {response.status_code}")
            return default_response
    except Exception as e:
        logging.debug(f"Analysis request error: {e}")
        return default_response
    
def scrape_reviews(keyword, search_results, max_reviews=20):
    all_reviews = []
    
    search_results = cached_search(f'reviews for {keyword}')
    # First, try to get review pages from the search results
    review_pages = [result for result in search_results 
                   if 'review' in result['title'].lower() or 'review' in result['body'].lower()]
    
    print('REVIEW PAGES',review_pages)
    
    # If we have potential review pages, try to extract reviews from them
    for page in review_pages[:3]:  # Limit to 3 pages to avoid too many requests
        try:
            # Get the page content
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(page['href'], headers=headers, timeout=5)
            
            if response.status_code == 200:
                # Parse with BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for review elements - this is site-specific so we'll use general patterns
                # Common review patterns
                review_patterns = [
                    # Look for divs with review-like class names
                    soup.find_all('div', class_=lambda c: c and ('review' in c.lower() or 'comment' in c.lower())),
                    # Look for paragraphs inside review-like containers
                    soup.find_all('p', class_=lambda c: c and ('review' in c.lower() or 'comment' in c.lower())),
                    # Look for review text in specific sites like Amazon
                    soup.find_all('span', {'data-hook': 'review-body'})
                ]
                
                # Flatten the list of review elements
                review_elements = [item for sublist in review_patterns for item in sublist]
                
                # Extract text from review elements
                for element in review_elements:
                    review_text = element.get_text().strip()
                    if len(review_text) > 20:  # Only include reviews of meaningful length
                        # Clean the review text
                        review_text = re.sub(r'\s+', ' ', review_text)
                        all_reviews.append({
                            'text': review_text,
                            'source': page['href']
                        })
                        
                        # Break if we have enough reviews
                        if len(all_reviews) >= max_reviews:
                            break
        
        except Exception as e:
            logging.debug(f"Error scraping reviews from {page['href']}: {e}")
    
    # If we still need more reviews, use the search results content as fallback
    if len(all_reviews) < max_reviews:
        for result in search_results:
            # Only include results that might contain reviews
            if 'review' in result['title'].lower() or 'review' in result['body'].lower():
                review_text = result['body']
                if len(review_text) > 20:
                    all_reviews.append({
                        'text': review_text,
                        'source': result['href']
                    })
                    
                    # Break if we have enough reviews
                    if len(all_reviews) >= max_reviews:
                        break
    
    # Ensure we don't have duplicates
    unique_reviews = []
    seen_texts = set()
    
    for review in all_reviews:
        if review['text'] not in seen_texts:
            seen_texts.add(review['text'])
            unique_reviews.append(review)
    
    # Add sentiment analysis to each review
    for review in unique_reviews:
        # Use TextBlob for sentiment analysis
        blob = TextBlob(review['text'])
        review['sentiment'] = blob.sentiment.polarity
        
        # Categorize the sentiment
        if review['sentiment'] > 0.2:
            review['category'] = 'Positive'
        elif review['sentiment'] < -0.2:
            review['category'] = 'Negative'
        else:
            review['category'] = 'Neutral'
    
    return unique_reviews   

def analyze_sentiment2(search_results, neg_threshold=0.1, compound_threshold=0):
    sia = SentimentIntensityAnalyzer()
    
    sentiments = []
    for result in search_results:
        text = f"{result['title']} {result['body']}"
        sentiment = sia.polarity_scores(text)
        
        # Determine sentiment category based on compound score
        if sentiment['compound'] >= compound_threshold:
            overall_sentiment = "Positive" if sentiment['compound'] > 0.25 else "Neutral"
        else:
            overall_sentiment = "Negative"
            
        # Only count negative if it's above a meaningful threshold
        adjusted_negative = sentiment['neg'] if sentiment['neg'] >= neg_threshold else 0.0
        
        sentiments.append({
            "title": result['title'][:50] + "..." if len(result['title']) > 50 else result['title'],
            "compound": sentiment['compound'],
            "positive": sentiment['pos'],
            "negative": adjusted_negative,  # Use adjusted negative score
            "neutral": sentiment['neu'],
            "sentiment": overall_sentiment,
            "has_negativity": adjusted_negative > 0  # Boolean flag for presence of significant negativity
        })
    
    df = pd.DataFrame(sentiments)
    logging.debug('SENTIMENT ANALYSIS RESULTS:')
    logging.debug(df)
    return df

# Function to perform local sentiment analysis on search results
def analyze_sentiment(search_results):
    sia = SentimentIntensityAnalyzer()
    
    sentiments = []
    for result in search_results:
        text = f"{result['title']} {result['body']}"
        sentiment = sia.polarity_scores(text)
        sentiments.append({
            "title": result['title'][:50] + "..." if len(result['title']) > 50 else result['title'],
            "compound": sentiment['compound'],
            "positive": sentiment['pos'],
            "negative": sentiment['neg'],
            "neutral": sentiment['neu']
        })
    logging.debug('COLLECTING IT AS PANDAS DATAFRAMES')
    logging.debug(pd.DataFrame(sentiments))
    return pd.DataFrame(sentiments)

# def get_trend_data(keyword, timeframe='now 7-d'):
#     url = f"https://en.wikipedia.org/wiki/{keyword.replace(' ', '_')}"
#     logging.debug(url)
#     response = requests.get(url,timeout=30)
#     logging.debug('RESPONSE',response.status_code)
    
#     if response.status_code != 200:
#         return None
    
#     soup = BeautifulSoup(response.text, 'html.parser')
    
#     # Find the page views section (this is a hypothetical example)
#     # You may need to adjust the selectors based on the actual HTML structure
#     views_data = []
#     for row in soup.find_all('tr'):
#         cells = row.find_all('td')
#         if len(cells) > 1:
#             date = cells[0].get_text(strip=True)
#             views = cells[1].get_text(strip=True)
#             views_data.append((date, int(views.replace(',', ''))))  # Convert views to integer
    
#     return pd.DataFrame(views_data, columns=['Date', 'Views'])

# # Test with a common keyword
# data = get_trend_data('python')

# Updated chart creation function with robust error handling
# def create_trend_chart(keyword, search_results=None):
#     try:
#         # Get actual trend data
#         trend_df = get_trend_data(keyword, timeframe='now 4-H')  # Last 4 hours
    
#         if trend_df.empty or len(trend_df) < 2:
#             return None, "No real-time interest data available for this keyword"
        
#         # Create the chart with Plotly
#         fig = px.line(trend_df, x='date', y='views', 
#                     title=f'Real-time interest in "{keyword}"')
        
#         # Add interactive features
#         fig.update_layout(
#             xaxis_title='date', 
#             yaxis_title='views',
#             hovermode='x unified',
#             updatemenus=[{
#                 'buttons': [
#                     {'args': [{'visible': [True, False, False]}, 
#                             {'title': f'Real-time interest in "{keyword}"'}],
#                     'label': '4 Hours',
#                     'method': 'update'},
#                     {'args': [{'visible': [False, True, False]}, 
#                             {'title': f'Real-time interest in "{keyword}"'}],
#                     'label': '24 Hours',
#                     'method': 'update'},
#                     {'args': [{'visible': [False, False, True]}, 
#                             {'title': f'Real-time interest in "{keyword}"'}],
#                     'label': '7 Days',
#                     'method': 'update'}
#                 ],
#                 'direction': 'down',
#                 'showactive': True,
#                 'x': 0.1,
#                 'y': 1.15
#             }]
#         )
        
#         return fig, None
#     except Exception as e:
#         logging.debug(f"Error creating trend chart: {e}")
#         # Return a None chart and a message
#         return None, "Unable to display interest data for this keyword"

# Function to create a sentiment distribution chart
def create_sentiment_chart(sentiment_df):
    if sentiment_df.empty:
        return None
    
    # Calculate average sentiment values
    avg_positive = sentiment_df['positive'].mean()
    avg_negative = sentiment_df['negative'].mean()
    avg_neutral = sentiment_df['neutral'].mean()
    
    # Create a donut chart
    fig = go.Figure(data=[go.Pie(
        labels=['Positive', 'Neutral', 'Negative'],
        values=[avg_positive, avg_neutral, avg_negative],
        hole=.3,
        marker_colors=['#2ca02c', '#7f7f7f', '#d62728'],
        textinfo='label+percent',
        hoverinfo='label+percent+value'
    )])
    
    fig.update_layout(
        title_text='Sentiment Distribution',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    return fig
# Set up the cache for expensive operations
# @st.cache_data(ttl=1800)  # Cache expires after 30 minutes
def cached_search(query):
    return search_web(query, max_results=10)

# @st.cache_data(ttl=1800)
def cached_analysis(keyword, search_results):
    return analyze_keyword_structured(keyword, search_results)

# Streamlit UI with improved layout
st.set_page_config(layout="wide", page_title="Keyword Analyzer Dashboard", page_icon="🔍")

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 1rem;
    }
    .search-box {
        padding: 10px;
        background-color: #F3F4F6;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<div class="main-header">🔍vSaaS Keyword Trend & Sentiment Analyzer</div>', unsafe_allow_html=True)

# Help text in the sidebar
with st.sidebar:
    st.image("logo.png")
    st.header("About This Tool")
    st.markdown("""This dashboard analyzes any keyword to provide insights on:
        * 📈 Current trends and interest
        * 💬 Public sentiment and opinions""")
# Main search box
st.markdown('<div class="search-box">', unsafe_allow_html=True)
keyword = st.text_input("", placeholder="Enter a keyword to analyze (product, brand, location, etc.)", 
                      help="Try entering a specific brand, product, place, or topic")
col1, col2 = st.columns([6, 1])
with col2:
    search_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
# Handle search
if search_button and keyword:
    # Show spinner while processing
    with st.spinner(f"Analyzing '{keyword}'..."):
            # Get search results using caching
            search_results = cached_search(keyword)
            
            # Always continue - if search_results is empty, we'll have a placeholder
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            # Update progress
            progress_text.text("Collecting search data...")
            progress_bar.progress(25)
            
            # Perform local sentiment analysis
            sentiment_df = analyze_sentiment2(search_results)
            progress_bar.progress(50)
            
            # Perform AI analysis
            progress_text.text("Analyzing Content")
            analysis = cached_analysis(keyword, search_results)
            progress_bar.progress(75)
            
            # Generate visualizations
            progress_text.text("Creating visualizations...")
            # trend_chart, trend_message = create_trend_chart(keyword)
            
            progress_bar.progress(100)
            progress_text.text("Analysis complete!")
            
            # Clear progress indicators after 1 second
            import time
            time.sleep(1)
            progress_text.empty()
            progress_bar.empty()
            
            # Display results in a dashboard layout
            st.markdown(f'<div class="subheader">Analysis Results for "{keyword}"</div>', unsafe_allow_html=True)
            
            # Summary section
            if analysis and "summary" in analysis:
                st.info(analysis["summary"])
            
            # Create tabs for different aspects of the analysis
            tab1, tab2, tab3, tab4= st.tabs(["📈 Trends & Topics", "😊 Sentiment","Reviews","🔍 Scraped Data"])
            
            with tab1:
                col1, col2 = st.columns([2,1])
                
                with col1:
                    # Trending topics
                    st.subheader("🔥 Trending Topics")
                    if analysis and "trending" in analysis and analysis["trending"]:
                        for item in analysis["trending"]:
                            st.markdown(f"- {item}")
                    else:
                        st.info("No trending topics found for this keyword.")
                    
                    # Current topics
                    st.subheader("📰 Current Discussions")
                    if analysis and "current_topics" in analysis and analysis["current_topics"]:
                        for item in analysis["current_topics"]:
                            st.markdown(f"- {item}")
                    else:
                        st.info("No current discussions found for this keyword.")
                
                # with col2:
                #     # Trend chart
                #     if trend_chart:
                #         st.plotly_chart(trend_chart, use_container_width=True)
                #     elif trend_message:
                #         st.info(trend_message)       
            with tab2:
                col1, col2,col3 = st.columns([1,2,1])
                
                with col2:
                    # Positive aspects
                    st.subheader("✅ Positive Aspects")
                    if analysis and "positive_aspects" in analysis:
                        for item in analysis["positive_aspects"]:
                            st.markdown(f"- {item}")
                    else:
                        st.markdown('no positive aspects')
                    
                    # Negative aspects
                    st.subheader("❌ Negative Aspects")
                    if analysis and "negative_aspects" in analysis:
                        for item in analysis["negative_aspects"]:
                            st.markdown(f"- {item}")
                    else:
                        st.markdown('no negative aspects')
            
            with tab3:
                st.subheader("Reviews Analysis")
                
                try:
                    with st.spinner("Scraping and analyzing reviews..."):
                        reviews = scrape_reviews(f'reviews for {keyword}', search_results)
                        
                        if reviews and len(reviews) > 0:
                            st.write(f"Found {len(reviews)} reviews related to '{keyword}'")
                            st.subheader("Review Samples")
                            if len(reviews) > 0:
                                for i, review in enumerate(reviews[:10]):  # Limit to 10 displayed reviews
                                    with st.expander(f"Review {i+1}"):
                                    # with st.expander(f"Review {i+1} ({review['category']})"):
                                        st.write(review['text'])
                                        st.caption(f"Source: {review['source']}")
                            else:
                                st.info("No detailed review content was found.")
                        else:
                            st.info("No reviews found for this keyword. Try a product or brand name for better results.")
                except Exception as e:
                    logging.debug(f"Error in reviews tab: {e}")
                    st.info("Review analysis is not available for this keyword.")            
            with tab4:
                # Raw search results
                st.subheader("Scraped Data")
                for i, result in enumerate(search_results, 1):
                    with st.expander(f"{i}. {result['title']}"):
                        st.write(result['body'])
                        st.write(f"Source: [{result['href']}]({result['href']})")
                
                # Raw analysis data
                if analysis:
                    with st.expander("Raw Analysis Data"):
                        st.json(analysis)
                        
                # Download options
                st.subheader("Download Data")
                col1, col2 = st.columns(2)
                with col1:
                    # Convert search results to CSV for download
                    search_df = pd.DataFrame(search_results)
                    csv = search_df.to_csv(index=False)
                    st.download_button(
                        label="Download Search Results",
                        data=csv,
                        file_name=f"{keyword}_search_results.csv",
                        mime="text/csv",
                    )
                with col2:
                    # Convert analysis to JSON for download
                    if analysis:
                        json_str = json.dumps(analysis, indent=2)
                        st.download_button(
                            label="Download Analysis Data",
                            data=json_str,
                            file_name=f"{keyword}_analysis.json",
                            mime="application/json",
                        )
else:
    # Display welcome screen with instructions
    st.info("👆 Enter a keyword above and click 'Analyze' to get started.")    
    # Example dashboard preview (static image or placeholder)
    st.image("https://img.icons8.com/color/452/dashboard-layout.png", width=300)
