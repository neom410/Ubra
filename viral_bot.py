import os
import requests
import praw
from datetime import datetime
import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedditTrendBot:
    def __init__(self):
        # Carica variabili d'ambiente
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'RedditTrendBot v1.0')
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        
        if not all([self.reddit_client_id, self.reddit_client_secret, self.telegram_bot_token]):
            raise ValueError("Variabili d'ambiente mancanti")
        
        # Inizializza Reddit
        self.reddit = praw.Reddit(
            client_id=self.reddit_client_id,
            client_secret=self.reddit_client_secret,
            user_agent=self.reddit_user_agent
        )
        
        # Cache per chat ID
        self.chat_id = None
        
    def get_telegram_chat_id(self):
        """Recupera automaticamente il chat ID dal bot"""
        if self.chat_id:
            return self.chat_id
            
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/getUpdates"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data['ok'] and data['result']:
                # Prendi l'ultimo messaggio ricevuto
                last_update = data['result'][-1]
                self.chat_id = last_update['message']['chat']['id']
                logger.info(f"Chat ID trovato: {self.chat_id}")
                return self.chat_id
            else:
                logger.warning("Nessun messaggio ricevuto dal bot. Invia un messaggio al bot prima.")
                return None
                
        except Exception as e:
            logger.error(f"Errore nel recupero chat ID: {e}")
            return None
    
    def get_reddit_trends(self):
        """Analizza le tendenze su Reddit"""
        trends = []
        subreddits = ['all', 'popular', 'worldnews', 'technology', 'programming']
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Analizza post hot degli ultimi 15 minuti
                for post in subreddit.hot(limit=50):
                    post_time = datetime.fromtimestamp(post.created_utc)
                    time_diff = (datetime.now() - post_time).total_seconds() / 60
                    
                    # Considera solo post degli ultimi 60 minuti (per avere più contenuti)
                    if time_diff <= 60:
                        trend_score = (post.score + post.num_comments * 2) / (time_diff + 1)
                        
                        trends.append({
                            'title': post.title[:200],  # Limita lunghezza titolo
                            'subreddit': post.subreddit.display_name,
                            'score': post.score,
                            'comments': post.num_comments,
                            'url': f"https://reddit.com{post.permalink}",
                            'created_utc': post.created_utc,
                            'trend_score': trend_score
                        })
                        
            except Exception as e:
                logger.error(f"Errore nell'analisi di r/{subreddit_name}: {e}")
                continue
        
        # Ordina per trend score
        trends.sort(key=lambda x: x['trend_score'], reverse=True)
        return trends[:10]  # Top 10 trends
    
    def format_alert_message(self, trends):
        """Formatta il messaggio per Telegram"""
        if not trends:
            return "🔍 Nessuna tendenza significativa negli ultimi 15 minuti"
        
        message = "🚨 **TREND REDDIT - Ultimi 15 minuti** 🚨\n\n"
        
        for i, trend in enumerate(trends[:5], 1):
            message += f"{i}. **{trend['title']}**\n"
            message += f"   📊 Score: {trend['score']} | 💬 Comments: {trend['comments']}\n"
            message += f"   📍 r/{trend['subreddit']}\n"
            message += f"   🔗 [Link]({trend['url']})\n\n"
        
        message += f"⏰ Aggiornato: {datetime.now().strftime('%H:%M:%S')}"
        return message
    
    def send_telegram_alert(self, message):
        """Invia alert su Telegram"""
        chat_id = self.get_telegram_chat_id()
        if not chat_id:
            logger.error("Chat ID non disponibile")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, json=payload, timeout=10)
            if response.json().get('ok', False):
                logger.info("✅ Alert inviato con successo")
                return True
            else:
                logger.error("❌ Errore nell'invio Telegram")
                return False
                
        except Exception as e:
            logger.error(f"❌ Errore nell'invio messaggio Telegram: {e}")
            return False
    
    def check_and_alert(self):
        """Esegue il controllo e invia l'alert"""
        logger.info("🔍 Avvio analisi tendenze Reddit...")
        
        try:
            trends = self.get_reddit_trends()
            message = self.format_alert_message(trends)
            
            if self.send_telegram_alert(message):
                logger.info("✅ Analisi completata con successo")
            else:
                logger.error("❌ Errore nell'invio dell'alert")
                
        except Exception as e:
            logger.error(f"❌ Errore durante il controllo: {e}")

def main():
    try:
        bot = RedditTrendBot()
        
        # Configura scheduler
        scheduler = BlockingScheduler()
        
        # Esegui immediatamente e poi ogni 15 minuti
        bot.check_and_alert()
        scheduler.add_job(
            bot.check_and_alert,
            trigger=IntervalTrigger(minutes=15),
            id='reddit_trend_check',
            name='Check Reddit trends every 15 minutes'
        )
        
        logger.info("🤖 Reddit Trend Bot avviato - Controllo ogni 15 minuti")
        scheduler.start()
        
    except Exception as e:
        logger.error(f"❌ Errore nell'avvio del bot: {e}")

if __name__ == "__main__":
    main()
