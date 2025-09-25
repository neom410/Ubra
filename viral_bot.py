import asyncpraw
import asyncio
import aiohttp
import json
import os
import logging
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import statistics
import re

# Configurazione logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('reddit_trends.log')]
)
logger = logging.getLogger(__name__)

# ===== REDDIT TRENDING TOPICS ANALYZER =====
class RedditTrendingAnalyzer:
    def __init__(self):
        self.trends_file = 'reddit_trends.json'
        self.history_file = 'trends_history.json'
        
        # Parametri per il calcolo del trending score
        self.scoring_weights = {
            'upvotes': 1.0,
            'comments': 2.5,  # Commenti pesano più dei like (discussione)
            'recent_multiplier': 1.8,  # I post recenti hanno più peso
            'engagement_ratio': 3.0,   # Rapporto commenti/upvotes
            'subreddit_activity': 1.2  # Subreddit molto attivi
        }
        
        # Categorie di interesse per l'applicazione (COMPLETE)
        self.categories = {
            'technology': ['programming', 'webdev', 'android', 'ios', 'software', 'ai', 'machinelearning', 'coding', 'tech', 'developer', 'app', 'digital'],
            'gaming': ['gaming', 'games', 'pcgaming', 'ps5', 'xbox', 'nintendo', 'steam', 'videogames', 'esports', 'gamer'],
            'entertainment': ['movies', 'television', 'music', 'books', 'art', 'netflix', 'spotify', 'youtube', 'celebrity', 'film'],
            'lifestyle': ['fitness', 'food', 'travel', 'photography', 'diy', 'cooking', 'health', 'wellness', 'home', 'garden'],
            'science': ['science', 'space', 'physics', 'biology', 'technology', 'research', 'discovery', 'innovation', 'climate'],
            'business': ['business', 'startups', 'entrepreneur', 'marketing', 'finance', 'investing', 'economy', 'stock', 'crypto'],
            'politica': ['politics', 'government', 'election', 'vote', 'policy', 'law', 'congress', 'senate', 'democrat', 'republican', 'europe', 'parliament'],
            'social': ['social', 'society', 'community', 'culture', 'relationship', 'friendship', 'network', 'communication', 'media'],
            'life': ['life', 'personal', 'experience', 'story', 'advice', 'help', 'support', 'mentalhealth', 'selfimprovement', 'motivation'],
            'mercato lavoro': ['job', 'career', 'work', 'employment', 'hire', 'recruitment', 'salary', 'interview', 'resume', 'careerchange', 'unemployment', 'remote work']
        }
        
        self.trending_history = self.load_history()

    def load_history(self):
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Errore caricamento history: {e}")
        return {}

    def save_history(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.trending_history, f, indent=2)
        except Exception as e:
            logger.error(f"Errore salvataggio history: {e}")

    def calculate_trending_score(self, post, hours_ago):
        """Calcola lo score di trending basato su engagement e recentezza"""
        base_score = 0
        
        # Punteggio base da upvotes e commenti
        base_score += post.score * self.scoring_weights['upvotes']
        base_score += post.num_comments * self.scoring_weights['comments']
        
        # Moltiplicatore per recentezza (più recente = più peso)
        recent_multiplier = max(0.1, 1 - (hours_ago / 48)) * self.scoring_weights['recent_multiplier']
        base_score *= recent_multiplier
        
        # Engagement ratio (commenti vs upvotes - indica discussione)
        if post.score > 0:
            engagement_ratio = post.num_comments / post.score
            base_score *= (1 + engagement_ratio * self.scoring_weights['engagement_ratio'])
        
        # Bonus per subreddit molto attivi
        if post.num_comments > 100:
            base_score *= self.scoring_weights['subreddit_activity']
        
        return round(base_score, 2)

    def categorize_topic(self, title, subreddit):
        """Categorizza il topic basandosi su titolo e subreddit"""
        title_lower = title.lower()
        subreddit_lower = subreddit.lower()
        
        category_scores = defaultdict(int)
        
        # Controlla categoria dal subreddit
        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword in subreddit_lower:
                    category_scores[category] += 3
                if keyword in title_lower:
                    category_scores[category] += 2
        
        # Categoria dalle keyword nel titolo
        title_words = set(re.findall(r'\w+', title_lower))
        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword in title_words:
                    category_scores[category] += 1
        
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            return best_category[0] if best_category[1] > 0 else 'general'
        
        return 'general'

    def extract_key_topics(self, title, content):
        """Estrae i topic principali dal titolo e contenuto"""
        text = f"{title} {content}".lower()
        words = re.findall(r'\b[a-z]{3,15}\b', text)
        
        # Filtra parole comuni
        stop_words = {'the', 'and', 'for', 'with', 'this', 'that', 'have', 'has', 'was', 'were', 'are', 'you', 'your', 'about', 'from', 'their', 'they', 'been', 'will', 'would', 'should', 'could'}
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Conta frequenza parole
        word_freq = Counter(meaningful_words)
        return [word for word, count in word_freq.most_common(5)]

# ===== REDDIT TRENDS HUNTER =====
class RedditTrendsHunter:
    def __init__(self):
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        
        if not all([self.reddit_client_id, self.reddit_client_secret]):
            raise ValueError("Credenziali Reddit mancanti!")
        
        self.analyzer = RedditTrendingAnalyzer()
        self.active_chats = set()
        self.processed_posts = set()
        
        # Subreddit per trending topics (ampiamente vari) - AGGIUNTI SUBREDDIT PER NUOVE CATEGORIE
        self.trending_subreddits = [
            # Tecnologia e Gaming
            'all', 'popular', 'technology', 'programming', 'gaming', 'science',
            # Notizie e Attualità
            'worldnews', 'news', 'politics', 'europe', 'todayilearned',
            # Intrattenimento
            'askscience', 'explainlikeimfive', 'dataisbeautiful', 'lifehacks', 'youshouldknow',
            # Lifestyle e Social
            'internetisbeautiful', 'food', 'travel', 'photography', 'diy', 'relationships',
            # Scienza e Cultura
            'space', 'futurology', 'books', 'music', 'movies', 'television',
            # Sport e Salute
            'sports', 'fitness', 'getdisciplined', 'learnprogramming', 'personalfinance',
            # Nuove categorie: Politica, Social, Life, Lavoro
            'jobs', 'careerguidance', 'recruiting', 'work', 'antiwork', 'careeradvice',
            'socialskills', 'socialmedia', 'community', 'selfimprovement', 'mentalhealth',
            'lifeadvice', 'personalfinance', 'financialindependence', 'legaladvice'
        ]

    async def initialize_reddit(self):
        try:
            self.reddit = asyncpraw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent='RedditTrendingHunter/1.0'
            )
            logger.info("Reddit connesso per trending topics")
            return True
        except Exception as e:
            logger.error(f"Errore connessione Reddit: {e}")
            return False

    async def get_active_chats(self):
        """Recupera le chat Telegram attive"""
        if not self.telegram_token:
            return False
            
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
                                    self.active_chats.add(chat_id)
            
            return True
        except Exception as e:
            logger.error(f"Errore chat detection: {e}")
            return False

    async def analyze_trending_topics(self):
        """Analizza i trending topics da Reddit"""
        try:
            trending_data = []
            current_time = datetime.now()
            
            for subreddit_name in self.trending_subreddits[:15]:  # Limita per performance
                try:
                    subreddit = await self.reddit.subreddit(subreddit_name)
                    
                    async for post in subreddit.hot(limit=15):  # Più post per subreddit
                        if post.id in self.processed_posts:
                            continue
                            
                        post_time = datetime.fromtimestamp(post.created_utc)
                        hours_ago = (current_time - post_time).total_seconds() / 3600
                        
                        # Filtra per engagement e recentezza
                        if (post.score >= 50 and post.num_comments >= 10 and 
                            hours_ago <= 48 and not post.stickied):
                            
                            trending_score = self.analyzer.calculate_trending_score(post, hours_ago)
                            category = self.analyzer.categorize_topic(post.title, subreddit_name)
                            key_topics = self.analyzer.extract_key_topics(post.title, post.selftext)
                            
                            topic_data = {
                                'id': post.id,
                                'title': post.title,
                                'subreddit': subreddit_name,
                                'score': post.score,
                                'comments': post.num_comments,
                                'trending_score': trending_score,
                                'category': category,
                                'key_topics': key_topics,
                                'created_utc': post.created_utc,
                                'hours_ago': round(hours_ago, 1),
                                'url': f"https://reddit.com{post.permalink}",
                                'engagement_ratio': round(post.num_comments / max(1, post.score), 2)
                            }
                            
                            trending_data.append(topic_data)
                            self.processed_posts.add(post.id)
                            
                except Exception as e:
                    logger.warning(f"Errore analisi r/{subreddit_name}: {e}")
                    continue
            
            # Ordina per trending score
            trending_data.sort(key=lambda x: x['trending_score'], reverse=True)
            
            # Mantieni solo i top 10
            top_trends = trending_data[:10]
            
            # Aggrega per categoria
            category_trends = defaultdict(list)
            for trend in top_trends:
                category_trends[trend['category']].append(trend)
            
            logger.info(f"Trovati {len(top_trends)} trending topics")
            
            return {
                'timestamp': current_time.isoformat(),
                'top_trends': top_trends,
                'by_category': dict(category_trends),
                'total_analyzed': len(trending_data)
            }
            
        except Exception as e:
            logger.error(f"Errore analisi trending topics: {e}")
            return None

    def format_trending_report(self, data):
        """Formatta il report dei trending topics"""
        if not data or not data['top_trends']:
            return "Nessun trending topic significativo trovato."
        
        trends = data['top_trends'][:5]  # Top 5
        timestamp = datetime.fromisoformat(data['timestamp']).strftime("%H:%M - %d/%m/%Y")
        
        # Emoji per tutte le categorie (COMPLETE)
        category_emojis = {
            'technology': '💻', 'gaming': '🎮', 'entertainment': '🎬',
            'lifestyle': '🌟', 'science': '🔬', 'business': '💼',
            'politica': '🏛️', 'social': '👥', 'life': '❤️',
            'mercato lavoro': '💼', 'general': '📰'
        }
        
        message = f"🔥 TRENDING TOPICS REDDIT 🔥\n"
        message += f"⏰ Aggiornamento: {timestamp}\n"
        message += f"📊 Analizzati {data['total_analyzed']} post\n\n"
        
        for i, trend in enumerate(trends, 1):
            emoji = category_emojis.get(trend['category'], '📰')
            title = trend['title'][:70] + "..." if len(trend['title']) > 70 else trend['title']
            
            message += f"{emoji} {i}. {title}\n"
            message += f"   ⬆️ {trend['score']} | 💬 {trend['comments']} "
            message += f"| 🔥 {trend['trending_score']}\n"
            message += f"   📍 r/{trend['subreddit']} | 🏷️ {trend['category']}\n"
            message += f"   🎯 Topics: {', '.join(trend['key_topics'][:3])}\n"
            message += f"   🔗 {trend['url']}\n\n"
        
        # Aggiungi statistiche categorie
        if data['by_category']:
            message += "📈 TRENDING PER CATEGORIA:\n"
            category_counts = [(cat, len(trends)) for cat, trends in data['by_category'].items()]
            category_counts.sort(key=lambda x: x[1], reverse=True)
            
            for category, count in category_counts[:4]:  # Top 4 categorie
                emoji = category_emojis.get(category, '📰')
                message += f"   {emoji} {category}: {count} trends\n"
        
        message += "\n🔄 Prossimo aggiornamento: 15 minuti"
        
        return message

    async def send_to_telegram(self, message):
        """Invia il report via Telegram"""
        if not self.telegram_token or not self.active_chats:
            logger.info("Nessuna chat Telegram attiva")
            return False
        
        success_count = 0
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            for chat_id in self.active_chats.copy():
                try:
                    url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                    payload = {
                        'chat_id': chat_id,
                        'text': message,
                        'disable_web_page_preview': False  # Abilita anteprime per link
                    }
                    
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            success_count += 1
                        else:
                            logger.warning(f"Errore invio a {chat_id}: {response.status}")
                            
                except Exception as e:
                    logger.error(f"Errore invio Telegram {chat_id}: {e}")
        
        return success_count > 0

    async def run_trending_hunter(self):
        """Esegue il monitoraggio continuo"""
        logger.info("Avvio Reddit Trending Topics Hunter")
        logger.info("Analisi ogni 15 minuti")
        
        if not await self.initialize_reddit():
            return
        
        logger.info("Trending Hunter operativo!")
        
        while True:
            try:
                if self.telegram_token:
                    await self.get_active_chats()
                
                logger.info("Scansionando Reddit per trending topics...")
                data = await self.analyze_trending_topics()
                
                if data and data['top_trends']:
                    report = self.format_trending_report(data)
                    logger.info(f"Trovati {len(data['top_trends'])} trending topics")
                    
                    # Log delle categorie trovate
                    categories_found = list(data['by_category'].keys())
                    logger.info(f"Categorie trovate: {', '.join(categories_found)}")
                    
                    if self.active_chats and self.telegram_token:
                        await self.send_to_telegram(report)
                    
                    # Log dei top trends
                    for trend in data['top_trends'][:3]:
                        logger.info(f"TOP: {trend['title'][:50]}... (Score: {trend['trending_score']}, Cat: {trend['category']})")
                
                # Pulizia periodicamente la lista dei post processati
                if len(self.processed_posts) > 1000:
                    self.processed_posts.clear()
                
                await asyncio.sleep(900)  # 15 minuti
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Errore main loop: {e}")
                await asyncio.sleep(300)  # Riprova dopo 5 minuti in caso di errore
        
        if hasattr(self, 'reddit'):
            await self.reddit.close()

async def main():
    try:
        hunter = RedditTrendsHunter()
        await hunter.run_trending_hunter()
    except Exception as e:
        logger.error(f"Errore critico: {e}")

if __name__ == "__main__":
    logger.info("Reddit Trending Topics Hunter v1.0")
    logger.info("Monitoraggio topic in trend ogni 15 minuti")
    asyncio.run(main())
