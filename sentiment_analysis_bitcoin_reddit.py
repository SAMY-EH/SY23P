
import praw # Pour Reddit
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 1. Connexion
reddit = praw.Reddit(client_id='...', client_secret='...', user_agent='MonProjet')
analyzer = SentimentIntensityAnalyzer()

# 2. Boucle en direct
for submission in reddit.subreddit("Bitcoin").stream.submissions():
    
    # 3. Analyse
    titre = submission.title
    score = analyzer.polarity_scores(titre)
    sentiment = score['compound'] # Score entre -1 (Négatif) et +1 (Positif)
    
    # 4. Résultat
    print(f"Post: {titre} | Sentiment: {sentiment}")
    
    # 5. Sauvegarde pour le graphique (ex: dans une liste ou base de données)
    update_dashboard(sentiment)