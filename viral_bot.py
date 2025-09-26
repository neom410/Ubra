import os
import praw
from datetime import datetime
import requests
import time
from collections import Counter
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

# Stopwords basic (senza NLTK)
BASIC_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
    'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'what', 'which', 'who', 'whom', 'where', 'when', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'just',
    'should', 'now', 'reddit', 'like', 'get', 'got', 'make', 'made', 'time', 'see',
    'its', 'his', 'her', 'our', 'your', 'their', 'has', 'have', 'had', 'does',
    'did', 'doing', 'would', 'could', 'should', 'might', 'must', 'shall', 'will'
}

class RedditTrendBot:
    def __init__(self):
        self.subreddits = ['all', 'popular', 'worldnews', 'technology', 'science', 'programming']
        
    def send_telegram_alert(self, message):
        """Invia alert su Telegram"""
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True
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

    def get_hot_posts(self, subreddit_name, limit=20):
        """Recupera i post più hot da un subreddit"""
        try:
            subreddit = reddit.subreddit(subreddit_name)
            posts = []
            
            for post in subreddit.hot(limit=limit):
                engagement = post.score + (post.num_comments * 2)
                
                posts.append({
                    'subreddit': subreddit_name,
                    'title': post.title,
                    'score': post.score,
                    'comments': post.num_comments,
                    'engagement': engagement,
                    'url': f"https://reddit.com{post.permalink}",
                    'created': datetime.fromtimestamp(post.created_utc),
                    'upvote_ratio': getattr(post, 'upvote_ratio', 0.95)
                })
            
            logger.info(f"✅ r/{subreddit_name}: trovati {len(posts)} post")
            return posts
        except Exception as e:
            logger.error(f"❌ Errore recupero post da r/{subreddit_name}: {e}")
            return []

    def analyze_trends(self, posts):
        """Analizza i trend dai post"""
        if not posts:
            return [], []
        
        # Filtra post con alto engagement
        hot_posts = [p for p in posts if p['engagement'] > 50]
        
        if not hot_posts:
            return [], []
        
        # Analizza parole chiave nei titoli
        all_titles = ' '.join([p['title'].lower() for p in hot_posts])
        words = self.clean_text(all_titles)
        
        # Filtra stopwords
        filtered_words = [w for w in words if w not in BASIC_STOPWORDS and len(w) > 2]
        
        # Trova trend
        word_freq = Counter(filtered_words)
        trends = word_freq.most_common(6)
        
        return trends, hot_posts

    def clean_text(self, text):
        """Pulisce il testo"""
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().split()

    def calculate_engagement_stats(self, posts):
        """Calcola statistiche di engagement"""
        if not posts:
            return 0, 0
        
        engagements = [p['engagement'] for p in posts]
        avg_engagement = sum(engagements) / len(engagements)
        hot_posts = len([p for p in posts if p['engagement'] > 50])
        
        return avg_engagement, hot_posts

    def generate_report(self):
        """Genera il report completo"""
        logger.info("🔄 Analisi trend Reddit in corso...")
        
        all_reports = {}
        
        for subreddit in self.subreddits:
            logger.info(f"📊 Analizzando r/{subreddit}...")
            
            posts = self.get_hot_posts(subreddit)
            trends, hot_posts = self.analyze_trends(posts)
            avg_engagement, hot_count = self.calculate_engagement_stats(posts)
            
            if trends:
                top_post = max(hot_posts, key=lambda x: x['engagement']) if hot_posts else None
                
                all_reports[subreddit] = {
                    'total_posts': len(posts),
                    'hot_posts': hot_count,
                    'avg_engagement': avg_engagement,
                    'trends': trends,
                    'top_post': top_post
                }
            
            time.sleep(1)  # Rate limiting
        
        return all_reports

    def format_telegram_message(self, reports):
        """Formatta il messaggio per Telegram"""
        if not reports:
            return "🔍 Nessun trend significativo trovato nelle ultime analisi."
        
        message = []
        message.append("🚀 <b>TREND REDDIT - LIVE ALERT</b>")
        message.append(f"⏰ {datetime.now().strftime('%H:%M %d/%m/%Y')}")
        message.append("")
        
        for subreddit, data in reports.items():
            message.append(f"<b>📌 r/{subreddit}</b>")
            message.append(f"   📊 Post totali: {data['total_posts']}")
            message.append(f"   🔥 Post hot: {data['hot_posts']}")
            message.append(f"   💪 Engagement medio: {data['avg_engagement']:.0f}")
            
            if data['trends']:
                message.append("   🏷️ <b>Trend topics:</b>")
                for word, count in data['trends'][:4]:
                    message.append(f"      • {word} ({count}x)")
            
            if data['top_post']:
                post = data['top_post']
                title = post['title'][:70] + "..." if len(post['title']) > 70 else post['title']
                message.append("   🏆 <b>Top post:</b>")
                message.append(f"      📝 {title}")
                message.append(f"      👍 {post['score']} ⬆️  💬 {post['comments']}")
                message.append(f"      🔗 <a href='{post['url']}'>Vai al post</a>")
            
            message.append("")
        
        # Aggiungi summary
        total_trends = sum(len(data['trends']) for data in reports.values())
        message.append(f"<b>📈 RIEPILOGO:</b> {total_trends} trend topics trovati in {len(reports)} subreddit")
        
        return "\n".join(message)

    def run_analysis(self):
        """Esegue l'analisi completa"""
        try:
            logger.info("🎯 Avvio analisi Reddit Trend...")
            
            # Analizza i trend
            reports = self.generate_report()
            
            # Genera il messaggio
            message = self.format_telegram_message(reports)
            
            # Log del report
            logger.info("📄 Report generato con successo")
            
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
            self.send_telegram_alert(f"❌ Errore bot: {str(e)}")
            return False

def main():
    """Funzione principale"""
    logger.info("🤖 Reddit Trend Bot started")
    
    # Verifica configurazione
    logger.info(f"🔧 Config: Reddit {'✅' if REDDIT_CLIENT_ID else '❌'}, Telegram {'✅' if TELEGRAM_BOT_TOKEN else '❌'}")
    
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
