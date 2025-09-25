import asyncpraw
import asyncio
import aiohttp
import json
import os
import logging
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import re

# Configurazione logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('topic_bot.log')]
)
logger = logging.getLogger(__name__)

# ===== SIMPLE TOPIC FINDER =====
class SimpleTopicFinder:
    def __init__(self):
        self.topics_file = 'hot_topics.json'
        self.processed_posts = set()
        
        # Liste di parole comuni da ignorare
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'must', 'can', 'what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom'
        }
        
        # Categorie semplici per classificazione
        self.categories = {
            'technology': ['ai', 'tech', 'programming', 'software', 'computer', 'code', 'app', 'digital'],
            'gaming': ['game', 'gaming', 'playstation', 'xbox', 'nintendo', 'steam', 'console'],
            'entertainment': ['movie', 'film', 'tv', 'music', 'netflix', 'youtube', 'celebrity'],
            'life': ['life', 'relationship', 'advice', 'personal', 'story', 'experience'],
            'politics': ['politics', 'government', 'election', 'policy', 'law', 'vote'],
            'work': ['job', 'work', 'career', 'salary', 'interview', 'employment'],
            'science': ['science', 'research', 'study', 'discovery', 'space', 'climate']
        }

    def extract_main_topic(self, title):
        """Estrae l'argomento principale dal titolo"""
        # Pulisci il titolo e converti in minuscolo
        title_clean = re.sub(r'[^\w\s]', ' ', title).lower()
        words = title_clean.split()
        
        # Filtra parole significative (lunghe > 3 caratteri, non stop words)
        meaningful_words = [
            word for word in words 
            if len(word) > 3 and word not in self.stop_words
        ]
        
        if not meaningful_words:
            return "general"
        
        # Conta le parole più frequenti
        word_counts = Counter(meaningful_words)
        main_word = word_counts.most_common(1)[0][0]
        
        return main_word

    def categorize_topic(self, topic_word, title):
        """Classifica l'argomento in una categoria"""
        title_lower = title.lower()
        
        for category, keywords in self.categories.items():
            if topic_word in keywords:
                return category
            # Controlla anche nel titolo completo
            for keyword in keywords:
                if keyword in title_lower:
                    return category
        
        return "general"

    def calculate_popularity_score(self, post):
        """Calcola un punteggio di popolarità semplice"""
        score = post.score + (post.num_comments * 2)  # Commenti pesano il doppio
        return score

# ===== REDDIT TOPIC BOT =====
class RedditTopicBot:
    def __init__(self):
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not all([self.reddit_client_id, self.reddit_client_secret]):
            raise ValueError("Credenziali Reddit mancanti!")
        
        self.topic_finder = SimpleTopicFinder()
        self.hot_topics = defaultdict(list)
        
        # Subreddit popolari per analisi
        self.subreddits = [
            'all', 'popular', 'askreddit', 'technology', 'gaming', 'science',
            'worldnews', 'politics', 'personalfinance', 'relationships',
            'jobs', 'lifeadvice', 'explainlikeimfive', 'todayilearned'
        ]

    async def initialize_reddit(self):
        try:
            self.reddit = asyncpraw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent='RedditTopicBot/1.0'
            )
            logger.info("Reddit connesso con successo")
            return True
        except Exception as e:
            logger.error(f"Errore connessione Reddit: {e}")
            return False

    async def find_hottest_topic(self):
        """Trova l'argomento più discusso in questo momento"""
        try:
            topic_scores = Counter()
            topic_details = defaultdict(list)
            
            for subreddit_name in self.subreddits[:8]:
                try:
                    subreddit = await self.reddit.subreddit(subreddit_name)
                    
                    async for post in subreddit.hot(limit=15):
                        if post.id in self.topic_finder.processed_posts:
                            continue
                            
                        # Filtra post recenti e popolari
                        post_time = datetime.fromtimestamp(post.created_utc)
                        hours_ago = (datetime.now() - post_time).total_seconds() / 3600
                        
                        if hours_ago <= 24 and post.score >= 10:
                            # Estrai argomento principale
                            main_topic = self.topic_finder.extract_main_topic(post.title)
                            category = self.topic_finder.categorize_topic(main_topic, post.title)
                            popularity = self.topic_finder.calculate_popularity_score(post)
                            
                            # Aggiorna punteggi
                            topic_scores[main_topic] += popularity
                            topic_details[main_topic].append({
                                'title': post.title,
                                'subreddit': subreddit_name,
                                'score': post.score,
                                'comments': post.num_comments,
                                'category': category
                            })
                            
                            self.topic_finder.processed_posts.add(post.id)
                            
                except Exception as e:
                    logger.warning(f"Errore in r/{subreddit_name}: {e}")
                    continue
            
            if not topic_scores:
                return None
            
            # Trova l'argomento più popolare
            hottest_topic, total_score = topic_scores.most_common(1)[0]
            topic_data = topic_details[hottest_topic]
            
            # Calcola statistiche
            total_posts = len(topic_data)
            total_comments = sum(t['comments'] for t in topic_data)
            avg_score = sum(t['score'] for t in topic_data) / total_posts
            
            # Trova la categoria predominante
            categories = Counter(t['category'] for t in topic_data)
            main_category = categories.most_common(1)[0][0]
            
            result = {
                'topic': hottest_topic,
                'total_score': total_score,
                'total_posts': total_posts,
                'total_comments': total_comments,
                'avg_score': round(avg_score, 1),
                'main_category': main_category,
                'example_posts': [t['title'] for t in topic_data[:3]],  # Prime 3 discussioni
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🔥 Trovato argomento caldo: {hottest_topic} (score: {total_score})")
            return result
            
        except Exception as e:
            logger.error(f"Errore ricerca argomenti: {e}")
            return None

    def format_topic_alert(self, topic_data):
        """Formatta l'alert per l'argomento più discusso"""
        if not topic_data:
            return "Nessun argomento popolare trovato al momento."
        
        emoji_map = {
            'technology': '💻', 'gaming': '🎮', 'entertainment': '🎬',
            'life': '❤️', 'politics': '🏛️', 'work': '💼', 'science': '🔬',
            'general': '🔥'
        }
        
        emoji = emoji_map.get(topic_data['main_category'], '🔥')
        
        message = f"{emoji} **ARGOMENTO PIÙ DISCUSSO SU REDDIT** {emoji}\n\n"
        message += f"🎯 **Tema Principale:** {topic_data['topic'].upper()}\n"
        message += f"📊 **Categoria:** {topic_data['main_category']}\n\n"
        
        message += f"📈 **Statistiche:**\n"
        message += f"• Punteggio Popolarità: {topic_data['total_score']}\n"
        message += f"• Discussioni Trovate: {topic_data['total_posts']}\n"
        message += f"• Commenti Totali: {topic_data['total_comments']}\n"
        message += f"• Score Medio: {topic_data['avg_score']}\n\n"
        
        message += f"💬 **Esempi di Discussioni:**\n"
        for i, post_title in enumerate(topic_data['example_posts'], 1):
            shortened_title = post_title[:80] + "..." if len(post_title) > 80 else post_title
            message += f"{i}. {shortened_title}\n"
        
        message += f"\n⏰ Aggiornamento: {datetime.now().strftime('%H:%M - %d/%m/%Y')}"
        
        return message

    async def send_telegram_alert(self, message):
        """Invia l'alert via Telegram"""
        if not self.telegram_token or not self.telegram_chat_id:
            logger.info("Telegram non configurato - salto invio")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.info("✅ Alert Telegram inviato con successo!")
                        return True
                    else:
                        logger.error(f"❌ Errore Telegram: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Errore invio Telegram: {e}")
            return False

    async def run_bot(self):
        """Esegue il bot principale"""
        logger.info("Avvio Reddit Topic Bot")
        logger.info("Ricerca argomento più discusso ogni 15 minuti")
        
        if not await self.initialize_reddit():
            return
        
        # Messaggio di avvio
        if self.telegram_token:
            await self.send_telegram_alert("🤖 **Reddit Topic Bot avviato!**\nInizio monitoraggio argomenti più discussi...")
        
        logger.info("Bot operativo!")
        
        analysis_count = 0
        
        while True:
            try:
                analysis_count += 1
                logger.info(f"Analisi #{analysis_count} in corso...")
                
                # Trova l'argomento più discusso
                hottest_topic = await self.find_hottest_topic()
                
                if hottest_topic:
                    # Formatta e invia l'alert
                    alert_message = self.format_topic_alert(hottest_topic)
                    
                    if self.telegram_token:
                        await self.send_telegram_alert(alert_message)
                    
                    # Log dei risultati
                    logger.info(f"📊 Risultati analisi #{analysis_count}:")
                    logger.info(f"   Topic: {hottest_topic['topic']}")
                    logger.info(f"   Categoria: {hottest_topic['main_category']}")
                    logger.info(f"   Punteggio: {hottest_topic['total_score']}")
                    logger.info(f"   Discussioni: {hottest_topic['total_posts']}")
                
                # Pulizia periodica
                if analysis_count % 10 == 0:
                    self.topic_finder.processed_posts.clear()
                    logger.info("🧹 Pulizia post processati effettuata")
                
                # Attesa 15 minuti
                await asyncio.sleep(900)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Errore durante l'analisi: {e}")
                await asyncio.sleep(300)  # Riprova dopo 5 minuti
        
        if hasattr(self, 'reddit'):
            await self.reddit.close()

# Esegui il bot
async def main():
    try:
        bot = RedditTopicBot()
        await bot.run_bot()
    except Exception as e:
        logger.error(f"Errore critico: {e}")

if __name__ == "__main__":
    logger.info("Reddit Topic Bot v1.0")
    logger.info("Monitoraggio argomenti più discussi su Reddit")
    asyncio.run(main())
