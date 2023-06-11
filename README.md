# YouTube-Popularity-Prediction

Social Media has become an inseparable part of today’s society. Not only is social
media a form of recreational activity, but more importantly, it is a money-making tool for
some individuals and businesses. With more than 2 billion monthly active users,
YouTube has been and continues to be one of the biggest social media gurus. Thus, we
are interested in investigating how videos become “trendy” on the YouTube platform.
This project will aim to develop an algorithm that predicts the popularity of
user-uploaded videos on YouTube.

Our project utilizes the YouTube API to extract information from the top 25 channels on the platform. This includes variables such as total view count across all videos, comment count, video categories, and more. By compiling this data into a comprehensive data frame with over thirty thousand observations, we aim to analyze and clean the sample dataset. Our objective is to identify relationships between various pairs of variables that can help us develop an algorithm using a machine learning model.

To achieve this, we divide the view count into ten subcategories and employ different machine learning models to train an algorithm with high accuracy. The ultimate goal is to create a web application that assists users, particularly businesses and influencers, in predicting the potential view count and popularity of their videos. By inputting parameters such as video duration, whether it is intended for children, year of upload, time of day, and title length, users will receive a predicted view count for their videos.

The 4 main technical components of the project include web scraping YouTube API to extract data, data analysis with complex data visualization, training machine learning models, and building an interactive and dynamic webserver.

