# Sentiment-Analysis-On-YouTube-Comments

## Overview
This project is a **Sentiment Analysis Web App** that classifies YouTube comments as **positive, negative, or neutral** using **Natural Language Processing (NLP)**. It is built using **Python Flask** as the backend framework and integrates NLP techniques to analyze user comments effectively.

## Features
- Collects YouTube comments using Kaggle.
- Cleans and preprocesses text data (removes stop words, special characters, etc.).
- Uses **Machine Learning (ML) models** for sentiment classification.
- Provides a **web interface** built using Flask.
- Displays sentiment analysis results in an intuitive UI.

## Technologies Used
- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, JavaScript, Bootstrap
- **Machine Learning:** NLP, Scikit-learn

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Flask
- Required Python libraries (install using `requirements.txt`)

### Steps to Run the Project
1. Clone the repository:
   ```sh
   git clone https://github.com/BSKanoje/Sentiment-Analysis-On-YouTube-Comments.git
   cd Sentiment-Analysis-On-YouTube-Comments
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Run the Flask application:
   ```sh
   python app.py
   ```

5. Open the web app in your browser:
   ```sh
   http://127.0.0.1:5000/
   ```

## Screenshots
![image](https://github.com/user-attachments/assets/f99e4b20-78e8-4701-83a3-baca415e4275)
![image](https://github.com/user-attachments/assets/251ee777-8d4b-4bf9-8ad4-11297c6411f2)
![image](https://github.com/user-attachments/assets/00480aca-5219-4c68-8e00-15710dbe2305)


## Future Enhancements
- Improve ML model accuracy using advanced NLP techniques (e.g., Transformer-based models like BERT).
- Add multilingual sentiment analysis.
- Deploy on a cloud platform (AWS, Heroku, etc.).
