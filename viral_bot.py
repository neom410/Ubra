import asyncpraw
import asyncio
import time
import os
from datetime import datetime, timedelta
import logging
from collections import Counter
import aiohttp
import re
import nest_asyncio

# Apply nest-asyncio per compatibilità con Render
nest_asyncio.apply()

# Configurazione logging avanzata
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('viral_bot.log')
    ]
)
logger = logging.getLogger(__name__)

class ViralNewsHunter:
    def __init__(self):
        # Legge le credenziali dalle variabili d'ambiente
        self.reddit_client_id = os.environ.get('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
        self.reddit_username = os.environ.get('REDDIT_USERNAME')
        self.reddit_password = os.environ.get('REDDIT_PASSWORD')
        self.telegram_token = os.environ.get('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        
        # Verifica che tutte le variabili siano presenti
        required_vars = {
            'REDDIT_CLIENT_ID': self.reddit_client_id,
            'REDDIT_CLIENT_SECRET': self.reddit_client_secret,
            'TELEGRAM_BOT_TOKEN': self.telegram_token,
            'TELEGRAM_CHAT_ID': self.telegram_chat_id
        }
        
        missing_vars = [k for k, v in required_vars.items() if not v]
        if missing_vars:
            raise ValueError(f"❌ Variabili d'ambiente mancanti: {missing_vars}")
        
        # Variabili per il tracking
        self.sent_posts = set()
        self.reddit = None
        
        # SUBREDDIT per notizie virali
        self.viral_subreddits = [
            'news', 'worldnews', 'technology', 'science', 'todayilearned',
            'interestingasfuck', 'nextfuckinglevel', 'gadgets', 'Futurology',
            'cryptocurrency', 'stocks', 'business', 'economics', 'space',
            'movies', 'gaming', 'nottheonion', 'offbeat', 'bestof'
        ]
        
        # KEYWORDS virali
        self.viral_indicators = [
            'breaking', 'urgent', 'record', 'historic', 'unprecedented',
            'shocking', 'viral', 'trending', 'million', 'billion', 
            'elon musk', 'ai', 'chatgpt', 'tesla', 'bitcoin', 'crisis'
        ]

    async def initialize_reddit(self):
        """Inizializza la connessione Reddit"""
        try:
            self.reddit = asyncpraw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                username=self.reddit_username,
                password=self.reddit_password,
                user_agent='ViralNewsBot/1.0 (by /u/your_username)'
            )
            logger.info("✅ Connessione Reddit inizializzata")
            return True
        except Exception as e:
            logger.error(f"❌ Errore inizializzazione Reddit: {e}")
            return False

    def calculate_viral_score(self, post, minutes_ago):
        """Calcola il punteggio virale di un post"""
        if minutes_ago <= 0:
            return 0
            
        score = 0
        title_lower = post.title.lower()
        
        # Punteggio base da upvotes e velocità
        upvotes_per_minute = post.score / minutes_ago
        score += min(upvotes_per_minute * 5, 100)
        
        # Bonus per engagement
        if post.num_comments > 0:
            comment_ratio = post.num_comments / minutes_ago
            score += min(comment_ratio * 3, 50)
        
        # Bonus per keywords virali
        for keyword in self.viral_indicators:
            if keyword in title_lower:
                score += 20
        
        return int(score)

    async def get_viral_posts(self):
        """Cerca post virali across tutti i subreddit"""
        viral_posts = []
        current_time = datetime.now()
        
        for subreddit_name in self.viral_subreddits:
            try:
                subreddit = await self.reddit.subreddit(subreddit_name)
                async for post in subreddit.new(limit=20):
                    post_time = datetime.fromtimestamp(post.created_utc)
                    minutes_ago = (current_time - post_time).total_seconds() / 60
                    
                    # Considera solo post delle ultime 2 ore
                    if minutes_ago <= 120:
                        viral_score = self.calculate_viral_score(post, minutes_ago)
                        
                        if viral_score >= 40 and post.id not in self.sent_posts:
                            viral_posts.append({
                                'id': post.id,
                                'title': post.title,
                                'score': post.score,
                                'subreddit': subreddit_name,
                                'url': f"https://reddit.com{post.permalink}",
                                'comments': post.num_comments,
                                'viral_score': viral_score,
                                'minutes_ago': int(minutes_ago)
                            })
                            
            except Exception as e:
                logger.warning(f"Errore in subreddit {subreddit_name}: {e}")
                continue
        
        # Ordina per viral score
        viral_posts.sort(key=lambda x: x['viral_score'], reverse=True)
        return viral_posts[:5]  # Top 5

    async def send_telegram_message(self, message):
        """Invia messaggio a Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.info("✅ Messaggio inviato a Telegram")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Errore Telegram: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Errore invio Telegram: {e}")
            return False

    def format_message(self, posts):
        """Formatta il messaggio per Telegram"""
        if not posts:
            return "🔍 Nessuna notizia virale trovata nell'ultima ora."
        
        message = "🔥 <b>NOTIZIE VIRALI IN TEMPO REALE</b> 🔥\n\n"
        
        for i, post in enumerate(posts, 1):
            message += f"📈 <b>Notizia #{i}</b>\n"
            message += f"📝 {post['title']}\n"
            message += f"🚀 Viral Score: {post['viral_score']}\n"
            message += f"👍 Upvotes: {post['score']} | 💬 Commenti: {post['comments']}\n"
            message += f"📍 Subreddit: r/{post['subreddit']}\n"
            message += f"⏰ {post['minutes_ago']} minuti fa\n"
            message += f"🔗 {post['url']}\n\n"
        
        message += f"⏰ Ultimo aggiornamento: {datetime.now().strftime('%H:%M:%S')}"
        return message

    async def run_scan(self):
        """Esegue una singola scansione"""
        try:
            logger.info("🔍 Inizio scansione notizie virali...")
            
            posts = await self.get_viral_posts()
            if posts:
                message = self.format_message(posts)
                success = await self.send_telegram_message(message)
                
                if success:
                    # Aggiungi i post inviati alla blacklist
                    for post in posts:
                        self.sent_posts.add(post['id'])
                    
                    logger.info(f"✅ Inviati {len(posts)} post virali")
                else:
                    logger.warning("❌ Fallito invio messaggio Telegram")
            else:
                logger.info("⚠️ Nessun post virale trovato")
                
        except Exception as e:
            logger.error(f"❌ Errore durante la scansione: {e}")

    async def main_loop(self):
        """Loop principale del bot"""
        # Inizializza Reddit
        if not await self.initialize_reddit():
            return
        
        logger.info("🚀 Viral News Bot avviato correttamente!")
        logger.info(f"📊 Monitoraggio {len(self.viral_subreddits)} subreddit")
        logger.info("⏰ Scansione ogni 15 minuti...")
        
        # Loop infinito
        while True:
            try:
                await self.run_scan()
                logger.info("⏳ Prossima scansione tra 15 minuti...")
                await asyncio.sleep(900)  # 15 minuti
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Errore nel main loop: {e}")
                await asyncio.sleep(300)  # Aspetta 5 minuti prima di riprovare
        
        # Cleanup
        if self.reddit:
            await self.reddit.close()
        logger.info("🛑 Bot fermato")

async def run_bot():
    """Funzione di avvio per Render"""
    bot = ViralNewsHunter()
    await bot.main_loop()

# Avvio sicuro per Render
if __name__ == "__main__":
    logger.info("🎯 Avvio Viral News Bot...")
    
    try:
        # Loop principale con gestione errori
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.info("🛑 Bot fermato manualmente")
    except Exception as e:
        logger.error(f"💥 Errore critico: {e}")
    finally:
        logger.info("👋 Uscita dal bot")
