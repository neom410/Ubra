import asyncpraw
import asyncio
import os
import logging
import aiohttp
from datetime import datetime
from collections import Counter, defaultdict

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleRedditTrendBot:
    def __init__(self):
        # Credenziali Reddit
        self.client_id = os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        
        # Credenziali Telegram
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.client_id or not self.client_secret:
            raise ValueError("❌ Configura REDDIT_CLIENT_ID e REDDIT_CLIENT_SECRET")
        
        # Subreddit da monitorare
        self.subreddits = [
            'all', 'popular', 'news', 'worldnews', 'technology',
            'gaming', 'movies', 'science', 'askreddit'
        ]
        
        # Filtri base
        self.min_score = 100
        self.min_comments = 20
        self.max_age_hours = 24
        
        logger.info("🚀 Bot semplice inizializzato")

    async def initialize_reddit(self):
        """Inizializza connessione Reddit"""
        try:
            self.reddit = asyncpraw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent='SimpleTrendBot/1.0'
            )
            logger.info("✅ Connesso a Reddit")
            return True
        except Exception as e:
            logger.error(f"❌ Errore connessione Reddit: {e}")
            return False

    def extract_keywords(self, text):
        """Estrae parole chiave semplici dal testo"""
        words = text.lower().split()
        
        # Filtra parole comuni
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                     'for', 'of', 'with', 'is', 'are', 'was', 'were', 'this', 'that'}
        
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        return Counter(keywords).most_common(5)  # Top 5 parole

    async def get_trending_posts(self, subreddit_name, limit=20):
        """Recupera post popolari da un subreddit"""
        try:
            subreddit = await self.reddit.subreddit(subreddit_name)
            posts_data = []
            
            async for post in subreddit.hot(limit=limit):
                # Calcola età del post
                post_age = (datetime.now().timestamp() - post.created_utc) / 3600
                
                # Applica filtri
                if (post.score >= self.min_score and 
                    post.num_comments >= self.min_comments and 
                    post_age <= self.max_age_hours):
                    
                    posts_data.append({
                        'id': post.id,
                        'title': post.title,
                        'subreddit': subreddit_name,
                        'score': post.score,
                        'comments': post.num_comments,
                        'age_hours': post_age,
                        'engagement': (post.score + post.num_comments) / max(post_age, 0.1)
                    })
            
            return posts_data
            
        except Exception as e:
            logger.warning(f"⚠️ Errore in r/{subreddit_name}: {e}")
            return []

    async def find_trends(self):
        """Trova i trend principali"""
        logger.info("🔍 Analizzando trend...")
        
        all_posts = []
        
        # Raccolta post da tutti i subreddit
        for subreddit in self.subreddits:
            posts = await self.get_trending_posts(subreddit)
            all_posts.extend(posts)
            logger.info(f"📊 r/{subreddit}: {len(posts)} post validi")
            await asyncio.sleep(1)  # Rate limiting
        
        if not all_posts:
            logger.warning("❌ Nessun post trovato")
            return None
        
        # Analizza le parole chiave più frequenti
        keyword_counter = Counter()
        
        for post in all_posts:
            keywords = self.extract_keywords(post['title'])
            for keyword, count in keywords:
                keyword_counter[keyword] += post['score']  # Pesa per score
        
        # Prendi i top 3 trend
        top_trends = keyword_counter.most_common(3)
        
        if not top_trends:
            return None
        
        # Prepara risultati
        trends_report = []
        
        for trend_word, trend_score in top_trends:
            # Trova post correlati a questo trend
            related_posts = []
            total_score = 0
            total_comments = 0
            
            for post in all_posts:
                if trend_word in post['title'].lower():
                    related_posts.append(post)
                    total_score += post['score']
                    total_comments += post['comments']
            
            if related_posts:
                trends_report.append({
                    'topic': trend_word,
                    'score': trend_score,
                    'post_count': len(related_posts),
                    'total_score': total_score,
                    'total_comments': total_comments,
                    'subreddits': list(set(p['subreddit'] for p in related_posts)),
                    'top_posts': sorted(related_posts, key=lambda x: x['engagement'], reverse=True)[:3]
                })
        
        return trends_report

    def format_trend_report(self, trends):
        """Formatta un report leggibile dei trend"""
        if not trends:
            return "📊 Nessun trend significativo trovato"
        
        report = "🔥 **TREND REDDIT IN TEMPO REALE**\n\n"
        
        for i, trend in enumerate(trends, 1):
            report += f"🎯 **TREND #{i}: {trend['topic'].upper()}**\n"
            report += f"📈 Potenza: {trend['score']:.0f} | "
            report += f"📊 Post: {trend['post_count']} | "
            report += f"💬 Commenti: {trend['total_comments']}\n"
            report += f"🌐 Subreddit: {', '.join([f'r/{sr}' for sr in trend['subreddits'][:3]])}\n"
            
            # Top post
            if trend['top_posts']:
                report += "📌 Top post:\n"
                for j, post in enumerate(trend['top_posts'][:2], 1):
                    title_short = post['title'][:60] + ('...' if len(post['title']) > 60 else '')
                    report += f"   {j}. {title_short}\n"
                    report += f"      ⬆️ {post['score']} | 💬 {post['comments']} | r/{post['subreddit']}\n"
            
            report += "\n" + "─" * 50 + "\n\n"
        
        report += f"⏰ Aggiornato: {datetime.now().strftime('%H:%M %d/%m/%Y')}"
        return report

    async def send_telegram_message(self, message):
        """Invia messaggio a Telegram"""
        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("⚠️ Telegram non configurato - skip invio")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            
            # Telegram ha limite di 4096 caratteri, dividiamo se necessario
            if len(message) > 4000:
                parts = [message[i:i+4000] for i in range(0, len(message), 4000)]
                for part in parts:
                    payload = {
                        'chat_id': self.telegram_chat_id,
                        'text': part,
                        'parse_mode': 'Markdown',
                        'disable_web_page_preview': True
                    }
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, json=payload) as response:
                            if response.status != 200:
                                logger.error(f"❌ Errore Telegram: {await response.text()}")
                            await asyncio.sleep(1)  # Rate limiting
            else:
                payload = {
                    'chat_id': self.telegram_chat_id,
                    'text': message,
                    'parse_mode': 'Markdown',
                    'disable_web_page_preview': True
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            logger.info("✅ Messaggio Telegram inviato")
                            return True
                        else:
                            logger.error(f"❌ Errore Telegram: {await response.text()}")
                            return False
                            
        except Exception as e:
            logger.error(f"❌ Errore invio Telegram: {e}")
            return False

    async def run(self):
        """Esegue il bot in loop"""
        if not await self.initialize_reddit():
            return
        
        logger.info("✅ Bot avviato - Analisi ogni 15 minuti")
        
        # Notifica di avvio
        if self.telegram_token:
            startup_msg = "🚀 **Reddit Trend Bot AVVIATO**\nMonitoraggio trend ogni 15 minuti"
            await self.send_telegram_message(startup_msg)
        
        analysis_count = 0
        
        while True:
            try:
                analysis_count += 1
                logger.info(f"🔄 Analisi #{analysis_count}")
                
                # Trova trend
                trends = await self.find_trends()
                
                if trends:
                    # Formatta report
                    report = self.format_trend_report(trends)
                    
                    # Stampa a console
                    print("\n" + "="*60)
                    print(report)
                    print("="*60 + "\n")
                    
                    # Salva su file
                    with open('trends_report.txt', 'w', encoding='utf-8') as f:
                        f.write(report)
                    logger.info("💾 Report salvato in trends_report.txt")
                    
                    # Invia a Telegram
                    if self.telegram_token:
                        await self.send_telegram_message(report)
                else:
                    logger.info("ℹ️ Nessun trend trovato in questo ciclo")
                    if self.telegram_token and analysis_count % 4 == 0:  # Ogni ora
                        no_trends_msg = "📊 **Ultima analisi**: Nessun trend significativo trovato"
                        await self.send_telegram_message(no_trends_msg)
                
                # Attesa 15 minuti
                logger.info("⏸️ In attesa di 15 minuti...")
                await asyncio.sleep(900)  # 15 minuti
                
            except KeyboardInterrupt:
                logger.info("🛑 Arresto richiesto")
                break
            except Exception as e:
                logger.error(f"❌ Errore: {e}")
                await asyncio.sleep(300)  # Riprova dopo 5 minuti
        
        # Notifica di chiusura
        if self.telegram_token:
            shutdown_msg = "🔴 **Reddit Trend Bot FERMATO**"
            await self.send_telegram_message(shutdown_msg)
        
        # Cleanup
        await self.reddit.close()
        logger.info("👋 Bot fermato")

# Esecuzione principale
async def main():
    try:
        bot = SimpleRedditTrendBot()
        await bot.run()
    except Exception as e:
        logger.error(f"❌ Errore fatale: {e}")

if __name__ == "__main__":
    print("🚀 AVVIO BOT TREND REDDIT")
    print("📊 Monitoraggio ogni 15 minuti")
    print("📱 Invio notifiche Telegram attivo")
    print("⏹️  Premi Ctrl+C per fermare\n")
    
    asyncio.run(main())
