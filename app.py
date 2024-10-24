import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load processed data
try:
    data = pd.read_csv('processed_feedback.csv')
except FileNotFoundError:
    st.error("File 'processed_feedback.csv' not found.")
    st.stop()

# Convert 'Timestamp' column to datetime
try:
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
except ValueError:
    st.error("Error converting 'Timestamp' column to datetime.")
    st.stop()

# Streamlit dashboard code
st.set_page_config(page_title="Student Feedback Analysis Dashboard", layout="wide")
st.title('📊 Student Feedback Analysis Dashboard')

# Sidebar filters
st.sidebar.header("Filter Feedback")
sentiment_filter = st.sidebar.multiselect("Select Sentiments", options=data['sentiment'].unique(), default=data['sentiment'].unique())
topic_filter = st.sidebar.multiselect("Select Topics", options=data['topic'].unique(), default=data['topic'].unique())
filtered_data = data[(data['sentiment'].isin(sentiment_filter)) & (data['topic'].isin(topic_filter))]

# Sentiment distribution
st.header('Sentiment Distribution')
sentiment_counts = filtered_data['sentiment'].value_counts()
fig = px.bar(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.values,
             labels={'x': 'Sentiment', 'y': 'Count'}, title="Sentiment Distribution",
             color=sentiment_counts.index, color_discrete_sequence=px.colors.qualitative.Set3)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# Topic distribution
st.header('Topic Distribution')
topic_counts = filtered_data['topic'].value_counts()
fig = px.bar(topic_counts, x=topic_counts.index, y=topic_counts.values,
             labels={'x': 'Topic', 'y': 'Count'}, title="Topic Distribution",
             color=topic_counts.index, color_discrete_sequence=px.colors.qualitative.Pastel)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# Word Cloud for common themes
st.header('Word Cloud of Feedback')
all_feedback = ' '.join(filtered_data['cleaned_feedback'])
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(all_feedback)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

# Sentiment trends over time
st.header('Sentiment Trends Over Time')
try:
    sentiment_over_time = filtered_data.groupby(filtered_data['Timestamp'].dt.date)['sentiment'].value_counts().unstack().fillna(0)
    fig = px.line(sentiment_over_time, x=sentiment_over_time.index, y=sentiment_over_time.columns,
                  labels={'x': 'Date', 'value': 'Count'}, title="Sentiment Trends Over Time")
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"An error occurred: {e}")

# Heatmap for sentiment intensity
st.header('Sentiment Intensity Heatmap')
try:
    sentiment_intensity = filtered_data.pivot_table(index=filtered_data['Timestamp'].dt.date, columns='sentiment', aggfunc='size', fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(sentiment_intensity.T, cmap='coolwarm', linewidths=0.5, ax=ax)
    plt.title("Sentiment Intensity Heatmap")
    st.pyplot(fig)
except Exception as e:
    st.error(f"An error occurred: {e}")

# Sunburst chart for hierarchical data
st.header('Sunburst Chart of Sentiment and Topics')
filtered_data['feedback_count'] = 1
fig = px.sunburst(filtered_data, path=['sentiment', 'topic'], values='feedback_count',
                  title="Sunburst Chart of Sentiment and Topics",
                  color='sentiment', color_discrete_sequence=px.colors.qualitative.Dark2)
st.plotly_chart(fig, use_container_width=True)

# Display feedback comments
st.header('Feedback Comments')
with st.expander("See all feedback comments"):
    for index, row in filtered_data.iterrows():
        st.write(f"*Sentiment: {row['sentiment']} | **Topic: {row['topic']} | **Feedback*: {row['feedback']}")

# Summary statistics
st.header('Summary Statistics')
st.write(filtered_data.describe(include='all'))

# Engagement Analysis
st.header('Engagement Analysis')
try:
    engagement_over_time = filtered_data.groupby(filtered_data['Timestamp'].dt.date).size()
    fig = px.line(engagement_over_time, x=engagement_over_time.index, y=engagement_over_time.values,
                  labels={'x': 'Date', 'y': 'Number of Feedback Entries'}, title="Engagement Over Time")
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"An error occurred: {e}")

# Recommendations based on feedback
st.header('Actionable Insights and Recommendations')
st.write("""
- *Improve Course Content:* Based on negative sentiment towards specific topics.
- *Enhance Instructor Interaction:* Suggestions for more interactive sessions.
- *Technical Improvements:* Address recurring technical issues mentioned.
- *Additional Resources:* Provide supplementary materials based on common feedback.
""")

# Feedback comparison between different segments
st.header('Comparison of Feedback Between Segments')
comparison_group = st.selectbox('Select a group to compare', ['Course Name', 'Instructor Name'])
try:
    if comparison_group == 'Course Name':
        comparison_data = filtered_data.groupby('Course Name')['sentiment'].value_counts().unstack().fillna(0)
    else:
        comparison_data = filtered_data.groupby('Instructor Name')['sentiment'].value_counts().unstack().fillna(0)
    fig = px.bar(comparison_data, x=comparison_data.index, y=comparison_data.columns,
                 labels={'x': comparison_group, 'value': 'Count'}, title=f"Feedback Comparison by {comparison_group}")
    st.plotly_chart(fig, use_container_width=True)
except KeyError:
    st.error(f"{comparison_group} column not found in the dataset.")
except Exception as e:
    st.error(f"An error occurred: {e}")