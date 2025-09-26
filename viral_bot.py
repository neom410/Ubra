import asyncpraw
import asyncio
import aiohttp
import json
import os
import logging
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import re
import sqlite3
import hashlib
from pathlib import Path

# Configurazione logging professionale
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('reddit_trend_bot.log', encoding='utf-8'),
        logging.FileHandler('errors.log', level=logging.ERROR, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrendData:
    """Classe per rappresentare i dati di una tendenza"""
    topic: str
    category: str
    total_score: int
    velocity_score: float  # Velocità di crescita
    post_count: int
    comment_count: int
    avg_engagement: float
    top_posts: List[Dict]
    subreddits: List[str]
    timestamp: datetime
    confidence_level: float  # Livello di confidenza della previsione

@dataclass
class PostMetrics:
    """Metriche di un singolo post"""
    id: str
    title: str
    subreddit: str
    score: int
    comments: int
    created_utc: float
    upvote_ratio: float
    engagement_rate: float

class DatabaseManager:
    """Gestisce il database per lo storico delle tendenze"""
    
    def __init__(self, db_path: str = "reddit_trends.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Inizializza il database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS trends (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        topic TEXT NOT NULL,
                        category TEXT,
                        total_score INTEGER,
                        velocity_score REAL,
                        post_count INTEGER,
                        comment_count INTEGER,
                        confidence_level REAL,
                        timestamp DATETIME,
                        data_json TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS processed_posts (
                        post_id TEXT PRIMARY KEY,
                        processed_at DATETIME,
                        topic TEXT
                    )
                ''')
                
                # Indici per performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_trends_timestamp ON trends(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_trends_topic ON trends(topic)')
                
                logger.info("Database inizializzato correttamente")
        except Exception as e:
            logger.error(f"Errore inizializzazione database: {e}")
    
    def save_trend(self, trend: TrendData):
        """Salva una tendenza nel database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO trends 
                    (topic, category, total_score, velocity_score, post_count, 
                     comment_count, confidence_level, timestamp, data_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trend.topic, trend.category, trend.total_score, 
                    trend.velocity_score, trend.post_count, trend.comment_count,
                    trend.confidence_level, trend.timestamp, 
                    json.dumps(trend.top_posts, ensure_ascii=False)
                ))
        except Exception as e:
            logger.error(f"Errore salvataggio tendenza: {e}")
    
    def get_historical_trends(self, hours_back: int = 24) -> List[Dict]:
        """Recupera le tendenze storiche"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT * FROM trends 
                    WHERE timestamp > datetime('now', '-{} hours')
                    ORDER BY timestamp DESC
                '''.format(hours_back))
                
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Errore recupero storico: {e}")
            return []

class AdvancedTopicAnalyzer:
    """Analizzatore avanzato per l'identificazione dei topic"""
    
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
            'should', 'may', 'might', 'must', 'can', 'what', 'when', 'where', 'why', 'how', 
            'which', 'who', 'whom', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'just', 'now'
        }
        
        # Categorie più specifiche e dettagliate
        self.categories = {
            'technology': {
                'keywords': ['ai', 'artificial', 'intelligence', 'tech', 'programming', 'software', 
                           'computer', 'code', 'app', 'digital', 'cybersecurity', 'blockchain', 
                           'crypto', 'startup', 'innovation', 'algorithm', 'data', 'cloud'],
                'weight': 1.2
            },
            'gaming': {
                'keywords': ['game', 'gaming', 'playstation', 'xbox', 'nintendo', 'steam', 
                           'console', 'esports', 'streamer', 'twitch', 'discord', 'fps', 'mmo'],
                'weight': 1.1
            },
            'finance': {
                'keywords': ['stock', 'market', 'invest', 'crypto', 'bitcoin', 'trading', 
                           'economy', 'financial', 'money', 'bank', 'inflation', 'recession'],
                'weight': 1.3
            },
            'entertainment': {
                'keywords': ['movie', 'film', 'tv', 'series', 'music', 'netflix', 'youtube', 
                           'celebrity', 'hollywood', 'streaming', 'podcast', 'album', 'concert'],
                'weight': 1.0
            },
            'politics': {
                'keywords': ['politics', 'government', 'election', 'policy', 'law', 'vote', 
                           'president', 'congress', 'senate', 'democracy', 'republican', 'democrat'],
                'weight': 1.4
            },
            'science': {
                'keywords': ['science', 'research', 'study', 'discovery', 'space', 'climate', 
                           'medical', 'health', 'covid', 'vaccine', 'medicine', 'physics', 'biology'],
                'weight': 1.2
            },
            'social': {
                'keywords': ['life', 'relationship', 'advice', 'personal', 'story', 'experience',
                           'family', 'friend', 'dating', 'marriage', 'social', 'community'],
                'weight': 0.9
            },
            'business': {
                'keywords': ['job', 'work', 'career', 'salary', 'interview', 'employment', 
                           'business', 'company', 'corporate', 'startup', 'entrepreneur'],
                'weight': 1.1
            }
        }
        
        # Pattern per identificare trending topics
        self.trending_patterns = [
            r'\b(?:breaking|urgent|update|alert|news)\b',
            r'\b(?:just|now|today|happening)\b',
            r'\b(?:viral|trending|popular|hot)\b',
        ]
    
    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[Tuple[str, float]]:
        """Estrae parole chiave dal testo con punteggio di rilevanza"""
        text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text_clean.split()
        
        # Filtra e assegna punteggi
        keyword_scores = Counter()
        
        for word in words:
            if len(word) > 3 and word not in self.stop_words:
                # Punteggio base per lunghezza
                score = len(word) * 0.1
                
                # Bonus per parole chiave di categoria
                for category_data in self.categories.values():
                    if word in category_data['keywords']:
                        score *= category_data['weight']
                
                # Bonus per pattern trending
                for pattern in self.trending_patterns:
                    if re.search(pattern, text.lower()):
                        score *= 1.5
                
                keyword_scores[word] += score
        
        return keyword_scores.most_common(max_keywords)
    
    def categorize_content(self, title: str, keywords: List[str]) -> Tuple[str, float]:
        """Categorizza il contenuto con livello di confidenza"""
        text = f"{title} {' '.join(keywords)}".lower()
        
        category_scores = {}
        
        for category, data in self.categories.items():
            score = 0
            matches = 0
            
            for keyword in data['keywords']:
                if keyword in text:
                    score += data['weight']
                    matches += 1
            
            if matches > 0:
                # Normalizza il punteggio
                category_scores[category] = (score / len(data['keywords'])) * matches
        
        if not category_scores:
            return 'general', 0.3
        
        best_category = max(category_scores, key=category_scores.get)
        confidence = min(category_scores[best_category], 1.0)
        
        return best_category, confidence
    
    def calculate_velocity_score(self, posts: List[PostMetrics], time_window_hours: int = 6) -> float:
        """Calcola la velocità di crescita di un topic"""
        if not posts:
            return 0.0
        
        now = datetime.now().timestamp()
        recent_posts = [p for p in posts if (now - p.created_utc) / 3600 <= time_window_hours]
        
        if len(recent_posts) < 2:
            return len(recent_posts) * 10  # Punteggio base per post singoli
        
        # Calcola crescita temporale
        recent_posts.sort(key=lambda x: x.created_utc)
        time_span = recent_posts[-1].created_utc - recent_posts[0].created_utc
        
        if time_span == 0:
            return len(recent_posts) * 15
        
        # Score basato su frequenza e engagement
        post_frequency = len(recent_posts) / (time_span / 3600)  # Post per ora
        avg_engagement = sum(p.engagement_rate for p in recent_posts) / len(recent_posts)
        
        velocity = post_frequency * avg_engagement * 10
        return min(velocity, 100)  # Cap a 100

class ProfessionalRedditTrendBot:
    """Bot professionale per l'analisi delle tendenze Reddit"""
    
    def __init__(self):
        # Credenziali e configurazione
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not all([self.reddit_client_id, self.reddit_client_secret]):
            raise ValueError("❌ Credenziali Reddit mancanti nelle variabili d'ambiente!")
        
        # Componenti del sistema
        self.db = DatabaseManager()
        self.analyzer = AdvancedTopicAnalyzer()
        self.processed_posts: Set[str] = set()
        
        # Configurazione subreddit strategici
        self.tier1_subreddits = [  # Subreddit principali - alta priorità
            'all', 'popular', 'news', 'worldnews', 'technology', 'science'
        ]
        
        self.tier2_subreddits = [  # Subreddit specializzati - media priorità
            'gaming', 'movies', 'music', 'askreddit', 'todayilearned',
            'explainlikeimfive', 'lifeprotips', 'showerthoughts'
        ]
        
        self.tier3_subreddits = [  # Subreddit di nicchia - bassa priorità
            'cryptocurrency', 'stocks', 'programming', 'artificial',
            'futurology', 'space', 'dataisbeautiful'
        ]
        
        # Metriche per il filtraggio
        self.min_score_threshold = 50
        self.min_comments_threshold = 10
        self.max_post_age_hours = 12
        
        logger.info("🚀 Bot professionale inizializzato")
    
    async def initialize_reddit(self) -> bool:
        """Inizializza la connessione Reddit con gestione errori avanzata"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                self.reddit = asyncpraw.Reddit(
                    client_id=self.reddit_client_id,
                    client_secret=self.reddit_client_secret,
                    user_agent='ProfessionalRedditTrendBot/2.0 (by /u/YourUsername)',
                    timeout=60
                )
                
                # Test della connessione
                await self.reddit.user.me()
                logger.info("✅ Connessione Reddit stabilita con successo")
                return True
                
            except Exception as e:
                logger.warning(f"⚠️ Tentativo {attempt + 1}/{max_retries} fallito: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(5 * (attempt + 1))
        
        logger.error("❌ Impossibile connettersi a Reddit")
        return False
    
    def calculate_post_metrics(self, post) -> PostMetrics:
        """Calcola metriche avanzate per un post"""
        # Engagement rate = (score + comments) / età_in_ore
        post_age_hours = (datetime.now().timestamp() - post.created_utc) / 3600
        engagement_rate = (post.score + post.num_comments * 2) / max(post_age_hours, 0.1)
        
        return PostMetrics(
            id=post.id,
            title=post.title,
            subreddit=post.subreddit.display_name,
            score=post.score,
            comments=post.num_comments,
            created_utc=post.created_utc,
            upvote_ratio=post.upvote_ratio,
            engagement_rate=engagement_rate
        )
    
    async def analyze_subreddit_trends(self, subreddit_name: str, limit: int = 25) -> List[PostMetrics]:
        """Analizza le tendenze di un singolo subreddit"""
        try:
            subreddit = await self.reddit.subreddit(subreddit_name)
            posts_data = []
            
            # Combina hot e rising per catturare tendenze emergenti
            hot_posts = subreddit.hot(limit=limit // 2)
            rising_posts = subreddit.rising(limit=limit // 2)
            
            for post_stream in [hot_posts, rising_posts]:
                async for post in post_stream:
                    if post.id in self.processed_posts:
                        continue
                    
                    # Filtri di qualità
                    post_age_hours = (datetime.now().timestamp() - post.created_utc) / 3600
                    
                    if (post.score >= self.min_score_threshold and 
                        post.num_comments >= self.min_comments_threshold and 
                        post_age_hours <= self.max_post_age_hours and
                        post.upvote_ratio >= 0.6):
                        
                        posts_data.append(self.calculate_post_metrics(post))
                        self.processed_posts.add(post.id)
            
            return posts_data
            
        except Exception as e:
            logger.warning(f"⚠️ Errore analisi r/{subreddit_name}: {e}")
            return []
    
    async def find_trending_topics(self) -> Optional[TrendData]:
        """Trova il topic più in tendenza con analisi multi-livello"""
        logger.info("🔍 Avvio analisi tendenze avanzata...")
        
        all_posts: List[PostMetrics] = []
        subreddit_weights = {}
        
        # Analizza subreddit per tier con pesi diversi
        for tier, (subreddits, weight) in enumerate([
            (self.tier1_subreddits, 3.0),
            (self.tier2_subreddits, 2.0), 
            (self.tier3_subreddits, 1.0)
        ], 1):
            
            logger.info(f"📊 Analisi Tier {tier}: {len(subreddits)} subreddit")
            
            for subreddit_name in subreddits:
                posts = await self.analyze_subreddit_trends(subreddit_name)
                all_posts.extend(posts)
                subreddit_weights[subreddit_name] = weight
                
                await asyncio.sleep(0.5)  # Rate limiting
        
        if not all_posts:
            logger.warning("⚠️ Nessun post valido trovato")
            return None
        
        # Raggruppa per topic usando keywords
        topic_groups = defaultdict(list)
        
        for post in all_posts:
            keywords = self.analyzer.extract_keywords(post.title, 3)
            if keywords:
                main_keyword = keywords[0][0]  # Keyword principale
                topic_groups[main_keyword].append(post)
        
        # Analizza ogni topic e calcola metriche
        topic_scores = {}
        
        for topic, posts in topic_groups.items():
            if len(posts) < 2:  # Richiede almeno 2 post per essere considerato trending
                continue
            
            # Metriche aggregate
            total_score = sum(p.score for p in posts)
            total_comments = sum(p.comments for p in posts)
            avg_engagement = sum(p.engagement_rate for p in posts) / len(posts)
            
            # Peso per subreddit
            weighted_score = sum(
                p.score * subreddit_weights.get(p.subreddit, 1.0) 
                for p in posts
            )
            
            # Velocità di crescita
            velocity = self.analyzer.calculate_velocity_score(posts)
            
            # Diversità di subreddit (bonus per topic cross-subreddit)
            unique_subreddits = len(set(p.subreddit for p in posts))
            diversity_bonus = 1 + (unique_subreddits - 1) * 0.2
            
            # Score finale
            final_score = (weighted_score + velocity * 10) * diversity_bonus
            
            topic_scores[topic] = {
                'posts': posts,
                'total_score': total_score,
                'velocity': velocity,
                'engagement': avg_engagement,
                'final_score': final_score,
                'diversity': unique_subreddits
            }
        
        if not topic_scores:
            logger.warning("⚠️ Nessun topic trending identificato")
            return None
        
        # Trova il topic più in tendenza
        best_topic = max(topic_scores.keys(), key=lambda t: topic_scores[t]['final_score'])
        best_data = topic_scores[best_topic]
        
        # Categorizza e crea risultato finale
        all_keywords = []
        for post in best_data['posts'][:5]:
            keywords = self.analyzer.extract_keywords(post.title, 2)
            all_keywords.extend([k[0] for k in keywords])
        
        category, confidence = self.analyzer.categorize_content(best_topic, all_keywords)
        
        # Top posts per il report
        top_posts = sorted(best_data['posts'], key=lambda p: p.engagement_rate, reverse=True)[:5]
        top_posts_data = [
            {
                'title': p.title[:100] + ('...' if len(p.title) > 100 else ''),
                'subreddit': p.subreddit,
                'score': p.score,
                'comments': p.comments,
                'engagement': round(p.engagement_rate, 1)
            }
            for p in top_posts
        ]
        
        result = TrendData(
            topic=best_topic,
            category=category,
            total_score=best_data['total_score'],
            velocity_score=best_data['velocity'],
            post_count=len(best_data['posts']),
            comment_count=sum(p.comments for p in best_data['posts']),
            avg_engagement=best_data['engagement'],
            top_posts=top_posts_data,
            subreddits=list(set(p.subreddit for p in best_data['posts'])),
            timestamp=datetime.now(),
            confidence_level=confidence
        )
        
        # Salva nel database
        self.db.save_trend(result)
        
        logger.info(f"🔥 TRENDING: {best_topic} | Score: {best_data['final_score']:.1f} | Confidenza: {confidence:.2f}")
        return result
    
    def format_professional_alert(self, trend: TrendData) -> str:
        """Formatta un alert professionale per il trending topic"""
        
        # Emoji per categoria
        category_emojis = {
            'technology': '💻', 'gaming': '🎮', 'finance': '💰',
            'entertainment': '🎬', 'politics': '🏛️', 'science': '🔬',
            'social': '👥', 'business': '💼', 'general': '🔥'
        }
        
        emoji = category_emojis.get(trend.category, '🔥')
        confidence_indicator = "🟢" if trend.confidence_level > 0.7 else "🟡" if trend.confidence_level > 0.4 else "🔴"
        
        alert = f"""{emoji} **REDDIT TREND ALERT** {emoji}

🎯 **TOPIC IN TENDENZA**: {trend.topic.upper()}
📂 **Categoria**: {trend.category.title()}
{confidence_indicator} **Confidenza**: {trend.confidence_level:.1%}

📈 **METRICHE PERFORMANCE**
├ Velocità crescita: {trend.velocity_score:.1f}/100
├ Discussioni attive: {trend.post_count}
├ Commenti totali: {trend.comment_count:,}
├ Engagement medio: {trend.avg_engagement:.1f}
└ Score aggregato: {trend.total_score:,}

🌐 **Subreddit coinvolti**: {len(trend.subreddits)}
{', '.join(f'r/{sr}' for sr in trend.subreddits[:5])}{'...' if len(trend.subreddits) > 5 else ''}

💬 **TOP DISCUSSIONI**"""
        
        for i, post in enumerate(trend.top_posts[:3], 1):
            alert += f"\n{i}. {post['title']}"
            alert += f"\n   📊 {post['score']} upvotes • {post['comments']} commenti • r/{post['subreddit']}"
        
        alert += f"\n\n⏰ **Rilevazione**: {trend.timestamp.strftime('%H:%M - %d/%m/%Y')}"
        alert += f"\n🤖 **Bot**: Professional Reddit Trend Analyzer v2.0"
        
        return alert
    
    async def send_telegram_notification(self, message: str) -> bool:
        """Invia notifica Telegram con retry automatico"""
        if not self.telegram_token or not self.telegram_chat_id:
            logger.info("📱 Telegram non configurato - skip notifica")
            return False
        
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                payload = {
                    'chat_id': self.telegram_chat_id,
                    'text': message,
                    'parse_mode': 'Markdown',
                    'disable_web_page_preview': True
                }
                
                timeout = aiohttp.ClientTimeout(total=30)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            logger.info("✅ Notifica Telegram inviata con successo")
                            return True
                        else:
                            logger.warning(f"⚠️ Telegram error {response.status}: {await response.text()}")
            
            except Exception as e:
                logger.warning(f"⚠️ Tentativo {attempt + 1} Telegram fallito: {e}")
                
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Backoff esponenziale
        
        logger.error("❌ Invio Telegram fallito dopo tutti i tentativi")
        return False
    
    async def run_professional_analysis(self):
        """Esegue l'analisi professionale delle tendenze"""
        logger.info("🚀 AVVIO PROFESSIONAL REDDIT TREND BOT")
        logger.info("📊 Frequenza analisi: ogni 15 minuti")
        logger.info("🎯 Focus: identificazione trending topics in tempo reale")
        
        # Inizializzazione
        if not await self.initialize_reddit():
            logger.error("❌ Impossibile avviare il bot - connessione Reddit fallita")
            return
        
        # Notifica di avvio
        if self.telegram_token:
            startup_msg = ("🚀 **Professional Reddit Trend Bot ONLINE**\n"
                         "📊 Monitoraggio trending topics attivato\n"
                         "⚡ Analisi ogni 15 minuti")
            await self.send_telegram_notification(startup_msg)
        
        analysis_counter = 0
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        logger.info("✅ Bot operativo e in monitoraggio...")
        
        while True:
            try:
                analysis_counter += 1
                start_time = datetime.now()
                
                logger.info(f"🔄 Analisi #{analysis_counter} iniziata")
                
                # Trova trending topic
                trending_topic = await self.find_trending_topics()
                
                if trending_topic:
                    # Formatta e invia alert
                    alert_message = self.format_professional_alert(trending_topic)
                    
                    # Log dettagliato
                    logger.info(f"📊 RISULTATI ANALISI #{analysis_counter}")
                    logger.info(f"   🎯 Topic: {trending_topic.topic}")
                    logger.info(f"   📂 Categoria: {trending_topic.category}")
                    logger.info(f"   ⚡ Velocità: {trending_topic.velocity_score:.1f}")
                    logger.info(f"   📈 Engagement: {trending_topic.avg_engagement:.1f}")
                    logger.info(f"   🔗 Subreddit: {len(trending_topic.subreddits)}")
                    
                    # Invia notifica
                    if self.telegram_token:
                        await self.send_telegram_notification(alert_message)
                    
                    consecutive_errors = 0  # Reset contatore errori
                    
                else:
                    logger.info(f"ℹ️ Nessun trending topic significativo trovato nell'analisi #{analysis_counter}")
                
                # Pulizia periodica
                if analysis_counter % 20 == 0:
                    old_count = len(self.processed_posts)
                    # Mantieni solo gli ultimi 1000 post processati per evitare memory leak
                    if len(self.processed_posts) > 1000:
                        posts_list = list(self.processed_posts)
                        self.processed_posts = set(posts_list[-500:])
                    
                    logger.info(f"🧹 Pulizia cache: {old_count} → {len(self.processed_posts)} post")
                
                # Statistiche performance
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(f"⏱️ Analisi #{analysis_counter} completata in {elapsed:.1f}s")
                
                # Attesa 15 minuti
                logger.info("⏸️ Pausa di 15 minuti prima della prossima analisi...")
                await asyncio.sleep(900)  # 15 minuti
                
            except KeyboardInterrupt:
                logger.info("🛑 Interruzione manuale richiesta")
                break
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"❌ Errore durante analisi #{analysis_counter}: {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(f"🚨 TROPPI ERRORI CONSECUTIVI ({consecutive_errors})")
                    if self.telegram_token:
                        error_msg = (f"🚨 **ALERT SISTEMA**\n"
                                   f"Bot in errore dopo {consecutive_errors} tentativi\n"
                                   f"Ultimo errore: {str(e)[:100]}")
                        await self.send_telegram_notification(error_msg)
                    
                    # Pausa più lunga in caso di errori multipli
                    logger.info("⏸️ Pausa estesa di 30 minuti per recupero sistema...")
                    await asyncio.sleep(1800)  # 30 minuti
                    consecutive_errors = 0
                else:
                    # Pausa breve per errori singoli
                    logger.info(f"⏸️ Pausa di 5 minuti per recupero (errore {consecutive_errors}/{max_consecutive_errors})")
                    await asyncio.sleep(300)  # 5 minuti
        
        # Cleanup finale
        try:
            if hasattr(self, 'reddit'):
                await self.reddit.close()
                logger.info("🔌 Connessione Reddit chiusa")
        except Exception as e:
            logger.warning(f"⚠️ Errore chiusura Reddit: {e}")
        
        # Notifica shutdown
        if self.telegram_token:
            shutdown_msg = "🔴 **Bot Spento**\nProfessional Reddit Trend Bot disconnesso"
            await self.send_telegram_notification(shutdown_msg)
        
        logger.info("👋 Professional Reddit Trend Bot terminato")

    async def generate_daily_report(self) -> str:
        """Genera un report giornaliero delle tendenze"""
        try:
            trends = self.db.get_historical_trends(24)
            
            if not trends:
                return "📊 **REPORT GIORNALIERO**\nNessun dato disponibile per le ultime 24 ore."
            
            # Analisi dei dati
            topics_counter = Counter(trend['topic'] for trend in trends)
            categories_counter = Counter(trend['category'] for trend in trends)
            
            top_topics = topics_counter.most_common(5)
            top_categories = categories_counter.most_common(3)
            
            avg_velocity = sum(trend['velocity_score'] for trend in trends) / len(trends)
            total_posts = sum(trend['post_count'] for trend in trends)
            total_comments = sum(trend['comment_count'] for trend in trends)
            
            report = f"""📊 **REPORT GIORNALIERO TENDENZE REDDIT**
🗓️ Periodo: Ultime 24 ore
📈 Analisi effettuate: {len(trends)}

🔥 **TOP TRENDING TOPICS**"""
            
            for i, (topic, count) in enumerate(top_topics, 1):
                report += f"\n{i}. {topic.title()} ({count} rilevazioni)"
            
            report += f"\n\n📂 **CATEGORIE PIÙ ATTIVE**"
            for i, (category, count) in enumerate(top_categories, 1):
                report += f"\n{i}. {category.title()} ({count} trending)"
            
            report += f"""

📊 **STATISTICHE AGGREGATE**
├ Velocità media: {avg_velocity:.1f}/100
├ Post totali analizzati: {total_posts:,}
├ Commenti totali: {total_comments:,}
└ Engagement medio: {(total_posts + total_comments) / len(trends):.1f}

⚡ Prossimo report: tra 24 ore
🤖 Professional Reddit Trend Bot v2.0"""
            
            return report
            
        except Exception as e:
            logger.error(f"Errore generazione report: {e}")
            return f"❌ Errore nella generazione del report giornaliero: {str(e)[:100]}"

    async def send_daily_report(self):
        """Invia il report giornaliero"""
        if self.telegram_token:
            report = await self.generate_daily_report()
            await self.send_telegram_notification(report)
            logger.info("📊 Report giornaliero inviato")

# ===== UTILITÀ E CONFIGURAZIONE =====

def setup_environment():
    """Configura l'ambiente e verifica le dipendenze"""
    required_vars = ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET']
    optional_vars = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
    
    missing_required = [var for var in required_vars if not os.getenv(var)]
    
    if missing_required:
        logger.error(f"❌ Variabili d'ambiente mancanti: {', '.join(missing_required)}")
        logger.error("💡 Configura le credenziali Reddit:")
        logger.error("   export REDDIT_CLIENT_ID='your_client_id'")
        logger.error("   export REDDIT_CLIENT_SECRET='your_client_secret'")
        return False
    
    missing_optional = [var for var in optional_vars if not os.getenv(var)]
    if missing_optional:
        logger.warning(f"⚠️ Variabili opzionali mancanti (no Telegram): {', '.join(missing_optional)}")
    
    # Crea directory per logs se non esiste
    Path('logs').mkdir(exist_ok=True)
    
    return True

async def main():
    """Funzione principale con gestione errori robusta"""
    logger.info("=" * 60)
    logger.info("🚀 PROFESSIONAL REDDIT TREND BOT v2.0")
    logger.info("📊 Analisi avanzata tendenze Reddit in tempo reale")
    logger.info("=" * 60)
    
    # Verifica ambiente
    if not setup_environment():
        return
    
    # Inizializza e avvia bot
    try:
        bot = ProfessionalRedditTrendBot()
        
        # Crea task per analisi principale
        main_task = asyncio.create_task(bot.run_professional_analysis())
        
        # Task per report giornaliero (opzionale)
        async def daily_report_scheduler():
            while True:
                # Calcola secondi fino alla prossima mezzanotte
                now = datetime.now()
                tomorrow = now.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
                seconds_until_report = (tomorrow - now).total_seconds()
                
                await asyncio.sleep(seconds_until_report)
                await bot.send_daily_report()
                await asyncio.sleep(24 * 60 * 60)  # 24 ore
        
        if bot.telegram_token:
            report_task = asyncio.create_task(daily_report_scheduler())
            await asyncio.gather(main_task, report_task)
        else:
            await main_task
            
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown richiesto dall'utente")
    except Exception as e:
        logger.critical(f"🚨 ERRORE CRITICO: {e}")
        logger.critical("💡 Controlla le credenziali e la connessione internet")
    finally:
        logger.info("👋 Arrivederci!")

# ===== ENTRY POINT =====

if __name__ == "__main__":
    # Configurazione finale logging
    logger.info("🔧 Inizializzazione Professional Reddit Trend Bot...")
    
    # Avvia il bot
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Interruzione da tastiera")
    except Exception as e:
        logger.critical(f"🚨 Errore fatale: {e}")
        exit(1)
