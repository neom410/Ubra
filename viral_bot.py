import asyncpraw
import asyncio
import time
import os
from datetime import datetime, timedelta
import logging
from collections import Counter
import aiohttp
import re
import threading
from aiohttp import web

# Configurazione logging per produzione
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)
logger = logging.getLogger(__name__)

class HealthServer:
    def __init__(self):
        self.app = web.Application()
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/', self.health_check)
        self.last_activity = time.time()
        self.bot_status = "starting"
        
    async def health_check(self, request):
        """Endpoint per health check"""
        status = {
            'status': 'healthy',
            'bot_status': self.bot_status,
            'timestamp': time.time(),
            'uptime': time.time() - self.last_activity,
            'service': 'viral-news-hunter'
        }
        return web.json_response(status)
    
    def update_status(self, status, activity=True):
        """Aggiorna lo status del bot"""
        self.bot_status = status
        if activity:
            self.last_activity = time.time()
    
    async def start_server(self):
        """Avvia il server di health check"""
        port = int(os.getenv('PORT', 8080))
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        logger.info(f"Health check server avviato sulla porta {port}")

# Health server globale
health_server = HealthServer()

class ViralNewsHunter:
    def __init__(self):
        # Leggi credenziali dalle variabili d'ambiente
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        
        # Verifica che le credenziali esistano
        if not all([self.reddit_client_id, self.reddit_client_secret, self.telegram_token]):
            raise ValueError("Variabili d'ambiente mancanti!")
        
        self.active_chats = set()
        self.reddit = None
        self.sent_posts = set()
        
        # SUBREDDIT per notizie virali
        self.viral_subreddits = [
            'news', 'worldnews', 'breakingnews', 'technology', 'gadgets',
            'Futurology', 'artificial', 'MachineLearning', 'cryptocurrency', 
            'bitcoin', 'business', 'stocks', 'wallstreetbets', 'todayilearned',
            'interestingasfuck', 'nextfuckinglevel', 'science', 'space'
        ]
        
        # KEYWORDS virali
        self.viral_indicators = [
            'breaking', 'urgent', 'developing', 'record', 'highest', 'lowest',
            'shocking', 'unbelievable', 'viral', 'trending', 'million', 'billion',
            'elon musk', 'ai', 'chatgpt', 'tesla', 'unprecedented', 'historic'
        ]
        
    async def initialize(self):
        """Inizializza le connessioni"""
        try:
            self.reddit = asyncpraw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent='ViralNewsHunter/2.0'
            )
            health_server.update_status("reddit_connected")
            logger.info("✅ Reddit connesso")
            return True
        except Exception as e:
            health_server.update_status("reddit_error")
            logger.error(f"❌ Errore Reddit: {e}")
            return False
    
    async def get_active_chats(self):
        """Rileva chat attive"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates"
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data['ok'] and data['result']:
                            for update in data['result']:
                                if 'message' in update:
                                    chat_id = update['message']['chat']['id']
                                    if chat_id not in self.active_chats:
                                        self.active_chats.add(chat_id)
                                        logger.info(f"Nuova chat: {chat_id}")
                            
                            if data['result']:
                                last_id = data['result'][-1]['update_id']
                                clear_url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates?offset={last_id + 1}"
                                await session.get(clear_url)
                        return True
                    return False
        except Exception as e:
            logger.error(f"Errore chat detection: {e}")
            return False
    
    def calculate_viral_score(self, post, subreddit, minutes_ago):
        """Calcola viral score"""
        score = 0
        title_lower = post.title.lower()
        
        # Velocità upvotes
        if minutes_ago > 0:
            upvotes_per_minute = post.score / minutes_ago
            score += min(upvotes_per_minute * 2, 100)
        
        # Upvotes assoluti
        if post.score > 1000:
            score += 50
        elif post.score > 500:
            score += 30
        elif post.score > 100:
            score += 15
        
        # Commenti
        if post.num_comments > 500:
            score += 40
        elif post.num_comments > 200:
            score += 25
        elif post.num_comments > 50:
            score += 10
        
        # Keywords virali
        for keyword in self.viral_indicators:
            if keyword in title_lower:
                score += 25
        
        # Numeri nel titolo
        numbers = re.findall(r'\d+[%$]?', title_lower)
        score += len(numbers) * 10
        
        return int(score)
    
    def categorize_viral_post(self, title, subreddit):
        """Categorizza post"""
        title_lower = title.lower()
        
        if 'elon' in title_lower or 'tesla' in title_lower:
            return '🚗 ELON/TESLA'
        elif any(word in title_lower for word in ['ai', 'chatgpt', 'robot']):
            return '🤖 AI/TECH'
        elif any(word in title_lower for word in ['bitcoin', 'crypto', 'stock']):
            return '💰 FINANZA'
        elif any(word in title_lower for word in ['breaking', 'urgent']):
            return '🚨 BREAKING'
        elif subreddit == 'todayilearned':
            return '📚 TIL'
        else:
            return '🔥 VIRALE'
    
    async def hunt_viral_news(self):
        """Caccia notizie virali"""
        try:
            viral_posts = []
            current_time = datetime.now()
            health_server.update_status("scanning")
            
            for subreddit_name in self.viral_subreddits:
                try:
                    subreddit = await self.reddit.subreddit(subreddit_name)
                    count = 0
                    async for post in subreddit.hot(limit=20):
                        count += 1
                        post_time = datetime.fromtimestamp(post.created_utc)
                        minutes_ago = (current_time - post_time).total_seconds() / 60
                        
                        if minutes_ago <= 360:  # 6 ore
                            viral_score = self.calculate_viral_score(post, subreddit_name, minutes_ago)
                            
                            if viral_score >= 60 and post.id not in self.sent_posts:
                                viral_posts.append({
                                    'id': post.id,
                                    'title': post.title,
                                    'score': post.score,
                                    'subreddit': subreddit_name,
                                    'url': f"https://reddit.com{post.permalink}",
                                    'comments': post.num_comments,
                                    'viral_score': viral_score,
                                    'minutes_ago': round(minutes_ago),
                                    'category': self.categorize_viral_post(post.title, subreddit_name),
                                    'upvotes_per_min': round(post.score / max(minutes_ago, 1), 1)
                                })
                        if count >= 20:
                            break
                except Exception as e:
                    logger.warning(f"Errore subreddit {subreddit_name}: {e}")
                    continue
            
            viral_posts.sort(key=lambda x: x['viral_score'], reverse=True)
            health_server.update_status("scan_complete")
            logger.info(f"Trovati {len(viral_posts)} post virali")
            
            return {
                'viral_posts': viral_posts[:6],
                'timestamp': current_time
            }
        except Exception as e:
            health_server.update_status("scan_error")
            logger.error(f"Errore caccia: {e}")
            return None
    
    def format_viral_message(self, data):
        """Formatta messaggio"""
        if not data or not data['viral_posts']:
            return "❌ Nessuna notizia virale al momento."
        
        timestamp = data['timestamp'].strftime("%H:%M - %d/%m/%Y")
        message = f"🔥 NOTIZIE VIRALI DELL'ULTIMA ORA 🔥\n⏰ {timestamp}\n\n"
        
        for i, post in enumerate(data['viral_posts'], 1):
            title = post['title'][:70] + "..." if len(post['title']) > 70 else post['title']
            title = title.replace('[', '').replace(']', '').replace('*', '')
            
            message += f"{post['category']} {i}. {title}\n"
            message += f"🚀 Score: {post['viral_score']} | 👍 {post['score']} ({post['upvotes_per_min']}/min)\n"
            message += f"💬 {post['comments']} | r/{post['subreddit']} | {post['minutes_ago']}min\n"
            message += f"🔗 {post['url']}\n\n"
        
        return message
    
    async def send_to_telegram(self, message):
        """Invia a Telegram"""
        if not self.active_chats:
            return False
        
        success_count = 0
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            for chat_id in self.active_chats.copy():
                try:
                    url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                    payload = {
                        'chat_id': chat_id,
                        'text': message,
                        'disable_web_page_preview': True
                    }
                    
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            success_count += 1
                        elif response.status in [400, 403, 404]:
                            self.active_chats.discard(chat_id)
                except Exception as e:
                    logger.error(f"Errore invio {chat_id}: {e}")
        
        return success_count > 0
    
    async def run_viral_hunter(self):
        """Main loop"""
        logger.info("🚀 Avvio Viral News Hunter...")
        
        if not await self.initialize():
            health_server.update_status("init_failed")
            return
        
        health_server.update_status("running")
        logger.info("✅ Bot avviato!")
        
        while True:
            try:
                await self.get_active_chats()
                
                viral_data = await self.hunt_viral_news()
                
                if viral_data and viral_data['viral_posts']:
                    new_viral = [p for p in viral_data['viral_posts'] if p['id'] not in self.sent_posts]
                    
                    if new_viral and self.active_chats:
                        for post in new_viral:
                            self.sent_posts.add(post['id'])
                        
                        viral_data['viral_posts'] = new_viral
                        message = self.format_viral_message(viral_data)
                        success = await self.send_to_telegram(message)
                        
                        if success:
                            health_server.update_status("message_sent")
                            logger.info(f"🔥 Inviate {len(new_viral)} notizie a {len(self.active_chats)} chat!")
                    elif not self.active_chats:
                        health_server.update_status("no_chats")
                        logger.info("⏳ Nessuna chat attiva")
                else:
                    health_server.update_status("no_viral_news")
                
                # Pulizia cache
                if len(self.sent_posts) > 1000:
                    self.sent_posts.clear()
                
                health_server.update_status("waiting")
                await asyncio.sleep(900)  # 15 minuti
                
            except Exception as e:
                health_server.update_status("error")
                logger.error(f"Errore main loop: {e}")
                await asyncio.sleep(180)
        
        if self.reddit:
            await self.reddit.close()

async def run_health_server():
    """Avvia health server"""
    await health_server.start_server()

async def main():
    """Main con health server"""
    # Avvia health server in background
    asyncio.create_task(run_health_server())
    
    try:
        bot = ViralNewsHunter()
        await bot.run_viral_hunter()
    except Exception as e:
        logger.error(f"Errore critico: {e}")
        health_server.update_status("critical_error")
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
