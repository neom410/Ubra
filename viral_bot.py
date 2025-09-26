import os
import praw
import pandas as pd
from datetime import datetime, timedelta
import requests
import time
from collections import Counter
import nltk
from nltk.corpus import stopwords
import re
import asyncio

# Configurazione da variabili d'ambiente
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

# Download stopwords NLTK (solo alla prima esecuzione)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class RedditTrendAnalyzer:
    def __init__(self):
        self.subreddits_monitorati = [
            'all', 'popular', 'worldnews', 'technology', 
            'science', 'programming', 'investing', 'stocks'
        ]
        self.ultime_trend = {}
        
    def invia_messaggio_telegram(self, messaggio):
        """Invia messaggio tramite bot Telegram"""
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': messaggio,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True
        }
        try:
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Errore Telegram: {e}")
            return False

    def analizza_subreddit(self, subreddit_name, limit=30):
        """Analizza un subreddit specifico"""
        try:
            subreddit = reddit.subreddit(subreddit_name)
            posts_data = []
            
            for post in subreddit.hot(limit=limit):
                # Calcola engagement score (upvotes + commenti)
                engagement_score = post.score + (post.num_comments * 2)
                
                posts_data.append({
                    'subreddit': subreddit_name,
                    'title': post.title.lower(),
                    'score': post.score,
                    'comments': post.num_comments,
                    'engagement': engagement_score,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'url': f"https://reddit.com{post.permalink}",
                    'author': str(post.author),
                    'flair': post.link_flair_text,
                    'upvote_ratio': post.upvote_ratio
                })
            
            df = pd.DataFrame(posts_data)
            return self._estrai_trend_da_dataframe(df)
            
        except Exception as e:
            print(f"❌ Errore analisi r/{subreddit_name}: {e}")
            return {}

    def _estrai_trend_da_dataframe(self, df):
        """Estrai trend dal DataFrame dei post"""
        if df.empty:
            return {}
        
        # Filtra post con alto engagement
        df_high_engagement = df[df['engagement'] > 50].copy()
        
        if df_high_engagement.empty:
            return {}
        
        # Analizza parole chiave nei titoli
        all_titles = ' '.join(df_high_engagement['title'].tolist())
        words = self._pulizia_testo(all_titles)
        
        # Filtra stopwords
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Trova parole più frequenti
        word_freq = Counter(filtered_words)
        top_trends = word_freq.most_common(10)
        
        # Prepara risultati
        trends = {
            'timestamp': datetime.now(),
            'total_posts': len(df),
            'high_engagement_posts': len(df_high_engagement),
            'avg_engagement': df['engagement'].mean(),
            'top_trends': top_trends,
            'top_posts': df_high_engagement.nlargest(3, 'engagement')[['title', 'engagement', 'url']].to_dict('records')
        }
        
        return trends

    def _pulizia_testo(self, testo):
        """Pulisce il testo per l'analisi"""
        # Rimuovi URL e caratteri speciali
        testo = re.sub(r'http\S+', '', testo)
        testo = re.sub(r'[^a-zA-Z\s]', '', testo)
        return testo.split()

    def analizza_trend_globali(self):
        """Analizza trend across tutti i subreddit monitorati"""
        print(f"🔍 Analisi trend alle {datetime.now()}")
        
        tutti_trends = {}
        
        for subreddit in self.subreddits_monitorati:
            print(f"📊 Analizzando r/{subreddit}...")
            trends = self.analizza_subreddit(subreddit)
            
            if trends:
                tutti_trends[subreddit] = trends
                
            # Pausa per evitare rate limiting
            time.sleep(2)
        
        return tutti_trends

    def genera_report(self, trends_data):
        """Genera un report formattato per Telegram"""
        if not trends_data:
            return "📊 Nessun trend significativo trovato nell'ultima analisi."
        
        report = []
        report.append("🚀 <b>TREND REDdit - Aggiornamento Live</b>")
        report.append(f"⏰ <i>{datetime.now().strftime('%H:%M %d/%m')}</i>")
        report.append("")
        
        for subreddit, data in trends_data.items():
            report.append(f"<b>📌 r/{subreddit}</b>")
            report.append(f"   📊 Post analizzati: {data['total_posts']}")
            report.append(f"   🔥 Post hot: {data['high_engagement_posts']}")
            report.append(f"   💪 Engagement medio: {data['avg_engagement']:.0f}")
            
            if data['top_trends']:
                report.append("   🏷️ <b>Trend Topics:</b>")
                for word, freq in data['top_trends'][:5]:
                    report.append(f"      • {word} ({freq}x)")
            
            if data['top_posts']:
                report.append("   🏆 <b>Top Post:</b>")
                for post in data['top_posts'][:1]:
                    title = post['title'][:50] + "..." if len(post['title']) > 50 else post['title']
                    report.append(f"      📝 {title}")
                    report.append(f"      🔗 <a href='{post['url']}'>Vai al post</a>")
            
            report.append("")
        
        return "\n".join(report)

    def esegui_analisi_completa(self):
        """Esegue analisi completa e invia report"""
        print("🔄 Avvio analisi trend Reddit...")
        
        try:
            # Analizza trend
            trends_data = self.analizza_trend_globali()
            
            # Genera report
            report = self.genera_report(trends_data)
            
            # Invia su Telegram
            success = self.invia_messaggio_telegram(report)
            
            if success:
                print("✅ Report inviato con successo!")
            else:
                print("❌ Errore nell'invio del report")
                
            return success
            
        except Exception as e:
            error_msg = f"❌ Errore nell'analisi: {str(e)}"
            print(error_msg)
            self.invia_messaggio_telegram(error_msg)
            return False

def main():
    """Funzione principale per l'esecuzione schedulata"""
    analyzer = RedditTrendAnalyzer()
    analyzer.esegui_analisi_completa()

if __name__ == "__main__":
    main()
