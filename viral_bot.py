import os
import praw
import pandas as pd
from datetime import datetime
import requests
import time
from collections import Counter
import nltk
from nltk.corpus import stopwords
import re
import logging

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Variabili d'ambiente (già configurate su Render)
REDDIT_CLIENT_ID = os.environ['REDDIT_CLIENT_ID']
REDDIT_CLIENT_SECRET = os.environ['REDDIT_CLIENT_SECRET']
TELEGRAM_BOT_TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
TELEGRAM_CHAT_ID = os.environ['TELEGRAM_CHAT_ID']

# Inizializza Reddit
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent="reddit-trend-bot-v1.0"
)

# Download stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class RedditTrendBot:
    def __init__(self):
        self.subreddits = ['all', 'popular', 'worldnews', 'technology', 'science', 'programming']
        
    def send_telegram_alert(self, message):
        """Invia alert su Telegram"""
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("✅ Alert inviato su Telegram")
                return True
            else:
                logger.error(f"❌ Errore Telegram: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Errore invio Telegram: {e}")
            return False

    def get_hot_posts(self, subreddit_name, limit=25):
        """Recupera i post più hot da un subreddit"""
        try:
            subreddit = reddit.subreddit(subreddit_name)
            posts = []
            
            for post in subreddit.hot(limit=limit):
                posts.append({
                    'subreddit': subreddit_name,
                    'title': post.title,
                    'score': post.score,
                    'comments': post.num_comments,
                    'engagement': post.score + post.num_comments,
                    'url': f"https://reddit.com{post.permalink}",
                    'created': datetime.fromtimestamp(post.created_utc)
                })
            
            return posts
        except Exception as e:
            logger.error(f"❌ Errore recupero post da r/{subreddit_name}: {e}")
            return []

    def analyze_trends(self, posts):
        """Analizza i trend dai post"""
        if not posts:
            return []
        
        # Filtra post con alto engagement
        hot_posts = [p for p in posts if p['engagement'] > 100]
        
        if not hot_posts:
            return []
        
        # Analizza parole chiave
        all_titles = ' '.join([p['title'].lower() for p in hot_posts])
        words = self.clean_text(all_titles)
        
        # Rimuovi stopwords
        stop_words = set(stopwords.words('english'))
        filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Trova trend
        word_freq = Counter(filtered_words)
        return word_freq.most_common(8)

    def clean_text(self, text):
        """Pulisce il testo"""
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.split()

    def generate_report(self):
        """Genera il report completo"""
        logger.info("🔄 Analisi trend Reddit in corso...")
        
        all_trends = {}
        
        for subreddit in self.subreddits:
            logger.info(f"📊 Scansionando r/{subreddit}...")
            
            posts = self.get_hot_posts(subreddit)
            trends = self.analyze_trends(posts)
            
            if trends:
                all_trends[subreddit] = {
                    'posts_analyzed': len(posts),
                    'hot_posts': len([p for p in posts if p['engagement'] > 100]),
                    'trends': trends[:5],
                    'top_post': max(posts, key=lambda x: x['engagement']) if posts else None
                }
            
            time.sleep(1)  # Rate limiting
        
        return all_trends

    def format_telegram_message(self, trends_data):
        """Formatta il messaggio per Telegram"""
        if not trends_data:
            return "🔍 Nessun trend significativo trovato"
        
        message = []
        message.append("🚀 <b>REDDIT TREND ALERT</b>")
        message.append(f"⏰ {datetime.now().strftime('%H:%M %d/%m/%Y')}")
        message.append("")
        
        for subreddit, data in trends_data.items():
            message.append(f"<b>📌 r/{subreddit}</b>")
            message.append(f"   📊 Post analizzati: {data['posts_analyzed']}")
            message.append(f"   🔥 Post hot: {data['hot_posts']}")
            
            if data['trends']:
                message.append("   🏷️ <b>Trend topics:</b>")
                for word, count in data['trends']:
                    message.append(f"      • {word} ({count}x)")
            
            if data['top_post']:
                post = data['top_post']
                title = post['title'][:80] + "..." if len(post['title']) > 80 else post['title']
                message.append("   🏆 <b>Top post:</b>")
                message.append(f"      {title}")
                message.append(f"      🔗 <a href='{post['url']}'>Vai al post</a>")
            
            message.append("")
        
        return "\n".join(message)

    def run_analysis(self):
        """Esegue l'analisi completa"""
        try:
            logger.info("🎯 Avvio analisi Reddit...")
            
            # Analizza i trend
            trends_data = self.generate_report()
            
            # Genera il messaggio
            message = self.format_telegram_message(trends_data)
            
            # Invia su Telegram
            success = self.send_telegram_alert(message)
            
            if success:
                logger.info("✅ Analisi completata e inviata!")
            else:
                logger.error("❌ Errore nell'invio del report")
            
            return success
            
        except Exception as e:
            error_msg = f"💥 Errore durante l'analisi: {str(e)}"
            logger.error(error_msg)
            self.send_telegram_alert(error_msg)
            return False

def main():
    """Funzione principale"""
    bot = RedditTrendBot()
    success = bot.run_analysis()
    
    if success:
        logger.info("🎉 Bot eseguito con successo!")
    else:
        logger.error("❌ Bot fallito!")
    
    # Esci per Render (no loop infinito)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()
