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
import re
from collections import Counter
from textblob import TextBlob

# Download NLTK resources (first-time only)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

# Set your OpenRouter API Key (don't prompt user)
OPENROUTER_API_KEY = "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"  # Replace with environment variable in production

# Add this to your imports
from serpapi import GoogleSearch
import time

# Update your search_web function with fallback
def search_web(query, max_results=10):
    try:
        # Primary search using DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "href": r.get("href", "")
                })
        
        # If DDGS returned results, use them
        if results:
            print('RESULTS FROM PRIMARY ENGINE (DDGS)')
            return results
        
        # If DDGS returned no results, try fallback
        print('DDGS RETURNED NO RESULTS, TRYING FALLBACK')
        return fallback_search(query, max_results)
        
    except Exception as e:
        print(f"Primary search error: {e}")
        print('TRYING FALLBACK SEARCH ENGINE')
        return fallback_search(query, max_results)

# Add fallback search function
def fallback_search(query, max_results=10):
    try:
        results = []
        # Using requests with custom headers to get regular search results
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        
        response = requests.get(search_url, headers=headers)
        
        # Simple regex-based extraction (very basic but works without API keys)
        title_pattern = r'<h3 class="[^"]*">(.*?)</h3>'
        title_matches = re.findall(title_pattern, response.text)
        
        snippet_pattern = r'<div class="[^"]*" data-sncf="[^"]*" data-snf="[^"]*">(.*?)</div>'
        snippet_matches = re.findall(snippet_pattern, response.text)
        
        url_pattern = r'<a href="(https?://[^"]+)"'
        url_matches = re.findall(url_pattern, response.text)
        
        # Combine the results
        for i in range(min(len(title_matches), len(snippet_matches), len(url_matches), max_results)):
            # Clean HTML tags from the results
            title = re.sub(r'<.*?>', '', title_matches[i])
            body = re.sub(r'<.*?>', '', snippet_matches[i])
            href = url_matches[i]
            
            results.append({
                "title": title,
                "body": body,
                "href": href
            })
        
        print('RESULTS FROM FALLBACK ENGINE')
        return results
        
    except Exception as e:
        st.error(f"Fallback search error: {e}")
        return []

# Function to analyze keyword using OpenRouter AI with structured output
def analyze_keyword_structured(keyword, search_results):
    # Combine search result content for context
    context = "\n".join([f"Title: {result['title']}\nContent: {result['body']}" for result in search_results[:5]])
    
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
    """
    
    payload = {
        "model": "deepseek/deepseek-r1-zero:free",
        "messages": [
            {"role": "system", "content": "You are an expert analyst who provides structured JSON outputs."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print('STATUS CODE', response.status_code)
        print('RESPONSE HEADERS:', response.headers)
        
        if response.status_code == 200:
            try:
                # First try the proper API response format
                content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "{}")
                print('RAW CONTENT:', content[:500])
                
                # Try to parse the JSON response
                try:
                    # Clean the content in case there's any markdown formatting
                    cleaned_content = re.sub(r'^```json\s*|\s*```$', '', content.strip())
                    result = json.loads(cleaned_content)
                    return result
                except json.JSONDecodeError as e:
                    print(f"JSON parse error: {e}")
                    # If not valid JSON, extract the JSON part using regex
                    json_match = re.search(r'({[\s\S]*})', content)
                    if json_match:
                        try:
                            extracted_json = json_match.group(1)
                            print(f"Extracted JSON: {extracted_json[:200]}")
                            return json.loads(extracted_json)
                        except Exception as e:
                            print(f"Extraction failed: {e}")
                    
                    # If all parsing attempts fail, create a default response with the content
                    return {
                        "trending": ["No clear trends identified"],
                        "positive_aspects": ["Unable to extract positive aspects"],
                        "negative_aspects": ["Unable to extract negative aspects"],
                        "current_topics": ["Unable to extract current topics"],
                        "key_metrics": {},
                        "sentiment_score": 0,
                        "summary": f"Analysis failed to parse. Raw content: {content[:300]}"
                    }
            except Exception as e:
                print(f"Response processing error: {e}")
    except Exception as e:
        st.error(f"Analysis error: {e}")
        return None
    
def scrape_reviews(keyword, search_results, max_reviews=20):
    all_reviews = []
    
    # First, try to get review pages from the search results
    review_pages = [result for result in search_results 
                   if 'review' in result['title'].lower() or 'review' in result['body'].lower()]
    
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
            print(f"Error scraping reviews from {page['href']}: {e}")
    
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

# Function to visualize reviews
def create_review_visualizations(reviews):
    if not reviews:
        return None, None, None
    
    # Create a DataFrame from the reviews
    reviews_df = pd.DataFrame(reviews)
    
    # 1. Sentiment distribution chart
    sentiment_counts = reviews_df['category'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    sentiment_pie = px.pie(
        sentiment_counts, 
        values='Count', 
        names='Sentiment',
        title='Review Sentiment Distribution',
        color='Sentiment',
        color_discrete_map={
            'Positive': '#2ca02c',
            'Neutral': '#7f7f7f',
            'Negative': '#d62728'
        },
        hole=0.3
    )
    
    # 2. Sentiment scores bar chart
    sentiment_bar = px.bar(
        reviews_df,
        y='sentiment',
        title='Individual Review Sentiment Scores',
        color='sentiment',
        color_continuous_scale='RdBu',
        labels={'sentiment': 'Sentiment Score', 'index': 'Review #'}
    )
    
    # 3. Word frequency chart - extract key terms
    # Combine all review text
    all_text = ' '.join([review['text'] for review in reviews])
    
    # Basic cleaning and tokenization
    words = re.findall(r'\b\w+\b', all_text.lower())
    
    # Remove common stop words
    stop_words = set(['the', 'and', 'of', 'to', 'a', 'in', 'for', 'is', 'on', 'that', 'by', 
                      'this', 'with', 'i', 'you', 'it', 'not', 'or', 'be', 'are', 'from', 
                      'at', 'as', 'your', 'have', 'more', 'has', 'an', 'was', 'com', 'www'])
    
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    # Create word frequency chart
    word_freq_df = pd.DataFrame(word_counts.most_common(15), columns=['Word', 'Frequency'])
    
    word_freq_chart = px.bar(
        word_freq_df, 
        x='Word', 
        y='Frequency',
        title='Most Common Words in Reviews',
        color='Frequency',
        color_continuous_scale='Viridis'
    )
    
    return sentiment_pie, sentiment_bar, word_freq_chart    

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
    print('SENTIMENT ANALYSIS RESULTS:')
    print(df)
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
    print('COLLECTING IT AS PANDAS DATAFRAMES')
    print(pd.DataFrame(sentiments))
    return pd.DataFrame(sentiments)

def get_trend_data(keyword):
    try:
        # Initialize pytrends
        pytrends = TrendReq(hl='en-US', tz=360)
        
        # Build payload
        pytrends.build_payload([keyword], cat=0, timeframe='today 1-m')
        
        # Get interest over time
        interest_over_time_df = pytrends.interest_over_time()
        
        # If no data was returned, generate placeholder data
        if interest_over_time_df.empty:
            return create_placeholder_trend_data(keyword)
            
        # Process the dataframe for plotting
        # Remove partial data for the current day
        if not interest_over_time_df.empty:
            interest_over_time_df = interest_over_time_df.iloc[:-1]
        
        # Reset index to convert date to column
        trend_df = interest_over_time_df.reset_index()
        
        # Rename columns
        trend_df = trend_df.rename(columns={'date': 'date', keyword: 'interest'})
        
        # Convert date to string for easier plotting
        trend_df['date'] = trend_df['date'].dt.strftime('%Y-%m-%d')
        
        return trend_df
        
    except Exception as e:
        print(f"Error getting trend data: {e}")
        # Fall back to placeholder data if there's an error
        return create_placeholder_trend_data(keyword)

def create_placeholder_trend_data(keyword):
    # This function creates placeholder data when Google Trends fails
    # It's the same as your original function but clearly labeled as estimated
    # days = 30
    # today = datetime.now()
    # dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days, 0, -1)]
    
    # base = random.randint(50, 500)
    # trend = []
    # overall_direction = random.choice([-1, 1])
    
    # for i in range(days):
    #     day_of_week = (today - timedelta(days=days-i-1)).weekday()
    #     weekend_factor = 1.1 if day_of_week >= 5 else 1.0
        
    #     daily_change = random.uniform(-0.08, 0.12) 
    #     trend_influence = overall_direction * (i/days) * 0.15
        
    #     if i == 0:
    #         trend.append(base * weekend_factor)
    #     else:
    #         next_val = trend[-1] * (1 + daily_change + trend_influence) * weekend_factor
    #         trend.append(max(10, next_val))
    
    # df = pd.DataFrame({
    #     'date': dates,
    #     'interest': trend
    # })
    df = pd.DataFrame({
        'date': [],
        'interest': []
    })
    
    return df

def create_trend_chart(keyword):
    # Get actual trend data
    trend_df = get_trend_data(keyword)
    
    # Create the chart with Plotly
    fig = px.line(trend_df, x='date', y='interest', 
                 title=f'Interest in "{keyword}" over the last 30 days')
    
    fig.update_layout(
        xaxis_title='Date', 
        yaxis_title='Interest Level',
        hovermode='x unified'
    )
    
    # Add annotation if using placeholder data
    if 'is_placeholder' in trend_df.columns and trend_df['is_placeholder'].iloc[0]:
        fig.add_annotation(
            text="Note: Using estimated data as actual trends unavailable",
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            showarrow=False,
            font=dict(size=10, color="red")
        )
    
    return fig


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

# Function to create word cloud from search results
# def create_word_cloud(search_results):
#     if not search_results:
#         return None
    
#     # Combine all text from search results
#     text = ' '.join([f"{result['title']} {result['body']}" for result in search_results])
    
#     # Common stopwords to exclude
#     stopwords = set(['the', 'and', 'of', 'to', 'a', 'in', 'for', 'is', 'on', 'that', 'by', 'this', 'with', 'i', 'you', 'it', 'not', 'or', 'be', 'are', 'from', 'at', 'as', 'your', 'have', 'more', 'has', 'an', 'was', 'com', 'www'])
    
#     # Create and generate a word cloud image
#     wordcloud = WordCloud(
#         width=800, 
#         height=400, 
#         background_color='white', 
#         max_words=100, 
#         contour_width=3, 
#         contour_color='steelblue',
#         stopwords=stopwords,
#         collocations=True,  # Include bigrams
#         min_font_size=10,
#         max_font_size=100
#     ).generate(text)
    
#     # Create matplotlib figure
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.imshow(wordcloud, interpolation='bilinear')
#     ax.axis('off')
    
#     return fig

# Function to create metrics comparison chart
# def create_metrics_chart(metrics):
#     if not metrics:
#         return None
    
#     # Convert metrics to numeric values where possible
#     numeric_metrics = {}
#     for key, value in metrics.items():
#         try:
#             # Try to extract numeric value from string
#             if isinstance(value, str):
#                 # Extract numbers from strings like "$599" or "4.5 stars"
#                 match = re.search(r'[-+]?\d*\.\d+|\d+', value)
#                 if match:
#                     numeric_metrics[key] = float(match.group())
#             elif isinstance(value, (int, float)):
#                 numeric_metrics[key] = float(value)
#         except:
#             pass
    
#     if not numeric_metrics:
#         return None
    
#     # Create bar chart
#     fig = px.bar(
#         x=list(numeric_metrics.keys()),
#         y=list(numeric_metrics.values()),
#         title="Key Metrics",
#         labels={'x': 'Metric', 'y': 'Value'},
#         color=list(numeric_metrics.values()),
#         text=list(numeric_metrics.values())
#     )
    
#     # Improve formatting
#     fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
#     fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    
#     return fig

# Set up the cache for expensive operations
@st.cache_data(ttl=1800)  # Cache expires after 30 minutes
def cached_search(query):
    return search_web(query, max_results=10)

@st.cache_data(ttl=1800)
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
    st.markdown("""
    This dashboard analyzes any keyword to provide insights on:
    
    * 📈 Current trends and interest
    * 💬 Public sentiment and opinions
    * 📊 Key metrics and statistics
    * 🔤 Common associated terms
    
    Enter any product, brand, person, location, or topic to get started!
    """)
    
    # st.markdown("---")
    # st.markdown("#### Tips for Best Results")
    # st.markdown("""
    # * Be specific with your keyword
    # * Try both broad terms and specific ones
    # * Compare related keywords in separate searches
    # """)

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
        
        if not search_results:
            st.error("No search results found. Please try a different keyword.")
        else:
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            # Update progress
            progress_text.text("Collecting search data...")
            progress_bar.progress(25)
            
            # Perform local sentiment analysis
            sentiment_df = analyze_sentiment2(search_results)
            progress_bar.progress(50)
            
            # Perform AI analysis
            progress_text.text("Analyzing content with AI...")
            analysis = cached_analysis(keyword, search_results)
            print('ANALYSIS')
            print(analysis)
            progress_bar.progress(75)
            
            # Generate visualizations
            progress_text.text("Creating visualizations...")
            trend_chart = create_trend_chart(keyword)
            # word_cloud = create_word_cloud(search_results)
            # sentiment_chart = create_sentiment_chart(sentiment_df)
            # metrics_chart = None
            # if analysis and "key_metrics" in analysis and analysis["key_metrics"]:
            #     metrics_chart = create_metrics_chart(analysis["key_metrics"])
            # else:
            #     print('no key metrics present in analysis')
            
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
            else:
                st.info('no summary')
            
            # Create tabs for different aspects of the analysis
            tab1, tab2, tab3, tab4= st.tabs(["📈 Trends & Topics", "😊 Sentiment","Reviews","🔍 Scraped Data"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Trending topics
                    st.subheader("🔥 Trending Topics")
                    if analysis and "trending" in analysis:
                        for item in analysis["trending"]:
                            st.markdown(f"- {item}")
                    else:
                        st.markdown("no trending")
                    
                    # Current topics
                    st.subheader("📰 Current Discussions")
                    if analysis and "current_topics" in analysis:
                        for item in analysis["current_topics"]:
                            st.markdown(f"- {item}")
                    else:
                        st.markdown("no current topics")
                
                with col2:
                    # Trend chart
                    st.plotly_chart(trend_chart, use_container_width=True)
                    
                #     # Word cloud
                #     st.subheader("Word Cloud")
                #     if word_cloud:
                #         st.pyplot(word_cloud)
            
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
                
                with st.spinner("Scraping and analyzing reviews..."):
                    reviews = scrape_reviews(keyword, search_results)
                    
                    if reviews:
                        st.write(f"Found {len(reviews)} reviews related to '{keyword}'")
                        
                        # Generate visualizations
                        # sentiment_pie, sentiment_bar, word_freq_chart = create_review_visualizations(reviews)
                        
                        # Display visualizations
                        col1, col2 = st.columns(2)
                        
                        # with col1:
                        #     if sentiment_pie:
                        #         st.plotly_chart(sentiment_pie, use_container_width=True)
                        
                        # with col2:
                        #     if word_freq_chart:
                        #         st.plotly_chart(word_freq_chart, use_container_width=True)
                        
                        # if sentiment_bar:
                        #     st.plotly_chart(sentiment_bar, use_container_width=True)
                        
                        # Show actual reviews
                        st.subheader("Review Samples")
                        for i, review in enumerate(reviews[:10]):  # Limit to 10 displayed reviews
                            with st.expander(f"Review {i+1} ({review['category']})"):
                                st.write(review['text'])
                                st.caption(f"Source: {review['source']}")
                    else:
                        st.info("No reviews found for this keyword. Try a product or brand name for better results.")
                
            
            with tab4:
                # Raw search results
                st.subheader("Search Results")
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
