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
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import time

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
    velocity_score: float
    post_count: int
    comment_count: int
    avg_engagement: float
    top_posts: List[Dict]
    subreddits: List[str]
    timestamp: datetime
    confidence_level: float
    sentiment_score: float
    discussion_quality: float
    controversy_score: float

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
    comment_velocity: float
    awards: int
    sentiment: float

@dataclass
class CommentAnalysis:
    """Analisi dei commenti per una discussione"""
    total_comments: int
    unique_authors: int
    avg_comment_score: float
    comment_velocity: float
    sentiment_distribution: Dict[str, float]
    controversial_comments: int
    depth_distribution: Dict[int, int]
    top_keywords: List[Tuple[str, int]]

class DiscussionAnalyzer:
    """Analizzatore avanzato delle discussioni e commenti"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.discussion_indicators = {
            'question_patterns': [
                r'\b(why|how|what|when|where|who|does|is|are|can)\b.*\?',
                r'opinion.*\?',
                r'thoughts.*\?',
                r'advice.*\?'
            ],
            'debate_indicators': [
                r'\b(agree|disagree|argument|debate|discuss|opinion)\b',
                r'change my mind',
                r'cmv\b',
                r'unpopular opinion'
            ],
            'emotional_indicators': [
                r'\b(wow|amazing|shocking|crazy|unbelievable|hate|love)\b',
                r'!\s*$',
                r'\?{2,}'
            ]
        }
    
    async def analyze_comments(self, post) -> CommentAnalysis:
        """Analizza i commenti di un post per valutare la qualità della discussione"""
        try:
            comments_data = {
                'total': 0,
                'authors': set(),
                'scores': [],
                'sentiments': [],
                'controversial': 0,
                'depths': defaultdict(int),
                'timestamps': []
            }
            
            comment_keywords = Counter()
            
            # Analizza i primi 100 commenti per performance
            post.comments.replace_more(limit=5)  # Limita i "load more comments"
            comments = post.comments.list()
            
            if not comments:
                return self._create_empty_analysis()
            
            for comment in comments[:200]:  # Limita per performance
                comments_data['total'] += 1
                comments_data['authors'].add(comment.author.name if comment.author else 'deleted')
                comments_data['scores'].append(comment.score)
                comments_data['timestamps'].append(comment.created_utc)
                
                # Analisi profondità
                depth = 0
                parent = comment.parent()
                while parent and hasattr(parent, 'parent'):
                    depth += 1
                    parent = parent.parent()
                comments_data['depths'][depth] += 1
                
                # Commenti controversi (score negativo o molto divisivo)
                if comment.score < 0 or comment.controversiality > 0:
                    comments_data['controversial'] += 1
                
                # Analisi sentiment e keywords
                if comment.body:
                    sentiment = self.sia.polarity_scores(comment.body)
                    comments_data['sentiments'].append(sentiment['compound'])
                    
                    # Estrai keywords dai commenti
                    keywords = self._extract_comment_keywords(comment.body)
                    comment_keywords.update(keywords)
            
            return self._compile_analysis(comments_data, comment_keywords)
            
        except Exception as e:
            logger.warning(f"Errore analisi commenti: {e}")
            return self._create_empty_analysis()
    
    def _extract_comment_keywords(self, text: str) -> List[str]:
        """Estrae keywords significative dai commenti"""
        # Rimuovi URL e caratteri speciali
        clean_text = re.sub(r'http\S+', '', text)
        clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
        
        words = clean_text.lower().split()
        # Filtra parole significative
        meaningful_words = [
            word for word in words 
            if len(word) > 3 and word not in self._get_stopwords()
        ]
        
        return meaningful_words[:5]  # Limita a 5 keywords per commento
    
    def _get_stopwords(self) -> Set[str]:
        """Lista di stopwords per l'analisi commenti"""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'this', 'that', 'these', 'those', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do'
        }
    
    def _create_empty_analysis(self) -> CommentAnalysis:
        """Crea un'analisi vuota per post senza commenti"""
        return CommentAnalysis(
            total_comments=0,
            unique_authors=0,
            avg_comment_score=0,
            comment_velocity=0,
            sentiment_distribution={'positive': 0, 'neutral': 0, 'negative': 0},
            controversial_comments=0,
            depth_distribution={0: 0},
            top_keywords=[]
        )
    
    def _compile_analysis(self, data: Dict, keywords: Counter) -> CommentAnalysis:
        """Compila i dati dell'analisi in un oggetto strutturato"""
        # Calcola metriche base
        unique_authors = len(data['authors'])
        avg_score = sum(data['scores']) / len(data['scores']) if data['scores'] else 0
        
        # Calcola velocità commenti (commenti per ora)
        if data['timestamps']:
            time_span = max(data['timestamps']) - min(data['timestamps'])
            comment_velocity = len(data['timestamps']) / (time_span / 3600) if time_span > 0 else 0
        else:
            comment_velocity = 0
        
        # Distribuzione sentiment
        sentiments = data['sentiments']
        sentiment_dist = {
            'positive': len([s for s in sentiments if s > 0.1]),
            'neutral': len([s for s in sentiments if -0.1 <= s <= 0.1]),
            'negative': len([s for s in sentiments if s < -0.1])
        }
        
        # Normalizza distribuzione sentiment
        total_sentiments = len(sentiments) if sentiments else 1
        for key in sentiment_dist:
            sentiment_dist[key] = sentiment_dist[key] / total_sentiments
        
        # Top keywords
        top_keywords = keywords.most_common(10)
        
        return CommentAnalysis(
            total_comments=data['total'],
            unique_authors=unique_authors,
            avg_comment_score=avg_score,
            comment_velocity=comment_velocity,
            sentiment_distribution=sentiment_dist,
            controversial_comments=data['controversial'],
            depth_distribution=dict(data['depths']),
            top_keywords=top_keywords
        )
    
    def calculate_discussion_quality(self, comment_analysis: CommentAnalysis) -> float:
        """Calcola un punteggio di qualità per la discussione (0-100)"""
        if comment_analysis.total_comments == 0:
            return 0
        
        quality_score = 0
        
        # Diversità autori (massimo 25 punti)
        author_diversity = min(comment_analysis.unique_authors / comment_analysis.total_comments, 1)
        quality_score += author_diversity * 25
        
        # Engagement commenti (massimo 25 punti)
        engagement_score = min(comment_analysis.avg_comment_score / 10, 1)
        quality_score += engagement_score * 25
        
        # Velocità discussione (massimo 20 punti)
        velocity_score = min(comment_analysis.comment_velocity / 50, 1)
        quality_score += velocity_score * 20
        
        # Profondità discussione (massimo 15 punti)
        deep_comments = sum(count for depth, count in comment_analysis.depth_distribution.items() if depth >= 2)
        depth_score = min(deep_comments / comment_analysis.total_comments, 1)
        quality_score += depth_score * 15
        
        # Qualità sentiment (massimo 15 punti)
        positive_ratio = comment_analysis.sentiment_distribution['positive']
        neutral_ratio = comment_analysis.sentiment_distribution['neutral']
        sentiment_score = (positive_ratio + neutral_ratio * 0.5)
        quality_score += sentiment_score * 15
        
        return min(quality_score, 100)
    
    def calculate_controversy_score(self, comment_analysis: CommentAnalysis, upvote_ratio: float) -> float:
        """Calcola un punteggio di controversia (0-100)"""
        if comment_analysis.total_comments == 0:
            return 0
        
        controversy_score = 0
        
        # Commenti controversi (massimo 40 punti)
        controversial_ratio = comment_analysis.controversial_comments / comment_analysis.total_comments
        controversy_score += controversial_ratio * 40
        
        # Upvote ratio basso (massimo 30 punti)
        ratio_score = (1 - upvote_ratio) * 30
        controversy_score += ratio_score
        
        # Distribuzione sentiment polarizzata (massimo 30 punti)
        sentiment_balance = abs(comment_analysis.sentiment_distribution['positive'] - 
                              comment_analysis.sentiment_distribution['negative'])
        controversy_score += sentiment_balance * 30
        
        return min(controversy_score, 100)

class AdvancedTopicAnalyzer:
    """Analizzatore avanzato per l'identificazione dei topic con focus discussioni"""
    
    def __init__(self):
        self.discussion_analyzer = DiscussionAnalyzer()
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
            'should', 'may', 'might', 'must', 'can', 'what', 'when', 'where', 'why', 'how'
        }
        
        # Pattern per identificare discussioni di qualità
        self.discussion_patterns = [
            r'.*\b(discuss|debate|opinion|thoughts|cmv|change my mind)\b.*',
            r'.*\?$',  # Domande
            r'.*\[serious\].*',  # Tag serious
            r'.*\b(why|how|what).*',  # Domande aperte
        ]

    async def analyze_post_discussion(self, post) -> Tuple[float, float, float]:
        """Analizza la qualità della discussione di un post"""
        try:
            # Analizza i commenti
            comment_analysis = await self.discussion_analyzer.analyze_comments(post)
            
            # Calcola metriche
            discussion_quality = self.discussion_analyzer.calculate_discussion_quality(comment_analysis)
            controversy_score = self.discussion_analyzer.calculate_controversy_score(comment_analysis, post.upvote_ratio)
            
            # Sentiment medio
            sentiments = comment_analysis.sentiment_distribution
            sentiment_score = (sentiments['positive'] - sentiments['negative']) * 100
            
            return discussion_quality, controversy_score, sentiment_score
            
        except Exception as e:
            logger.warning(f"Errore analisi discussione: {e}")
            return 0, 0, 0

class ProfessionalRedditTrendBot:
    """Bot professionale per l'analisi delle tendenze Reddit con focus discussioni"""
    
    def __init__(self):
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not all([self.reddit_client_id, self.reddit_client_secret]):
            raise ValueError("❌ Credenziali Reddit mancanti!")
        
        self.analyzer = AdvancedTopicAnalyzer()
        self.processed_posts: Set[str] = set()
        
        # Subreddit ottimizzati per discussioni di qualità
        self.discussion_subreddits = [
            'askreddit', 'changemyview', 'trueaskreddit', 'discussion', 
            'seriousconversation', 'casualconversation', 'unpopularopinion',
            'debate', 'explainlikeimfive', 'tooafraidtoask', 'nostupidquestions'
        ]
        
        # Metriche specifiche per discussioni
        self.min_comments_threshold = 20  # Minimo commenti per considerare una discussione
        self.min_discussion_quality = 30  # Qualità minima discussione
        
        logger.info("🚀 Bot discussioni Reddit inizializzato")

    async def initialize_reddit(self) -> bool:
        """Inizializza la connessione Reddit"""
        try:
            self.reddit = asyncpraw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent='DiscussionTrendBot/1.0',
                timeout=60
            )
            await self.reddit.user.me()
            logger.info("✅ Connessione Reddit stabilita")
            return True
        except Exception as e:
            logger.error(f"❌ Errore connessione Reddit: {e}")
            return False

    async def analyze_discussion_trends(self) -> Optional[TrendData]:
        """Analizza le tendenze delle discussioni ogni 15 minuti"""
        logger.info("🔍 Analisi discussioni trending in corso...")
        
        all_posts = []
        
        # Analizza i subreddit di discussione
        for subreddit_name in self.discussion_subreddits:
            try:
                subreddit = await self.reddit.subreddit(subreddit_name)
                posts_data = []
                
                # Combina hot e new per catturare discussioni emergenti
                async for post in subreddit.hot(limit=15):
                    if (post.num_comments >= self.min_comments_threshold and 
                        post.id not in self.processed_posts):
                        
                        # Analizza la discussione
                        discussion_quality, controversy, sentiment = await self.analyzer.analyze_post_discussion(post)
                        
                        if discussion_quality >= self.min_discussion_quality:
                            post_metrics = PostMetrics(
                                id=post.id,
                                title=post.title,
                                subreddit=post.subreddit.display_name,
                                score=post.score,
                                comments=post.num_comments,
                                created_utc=post.created_utc,
                                upvote_ratio=post.upvote_ratio,
                                engagement_rate=(post.score + post.num_comments) / max((time.time() - post.created_utc) / 3600, 0.1),
                                comment_velocity=post.num_comments / max((time.time() - post.created_utc) / 3600, 0.1),
                                awards=post.total_awards_received,
                                sentiment=sentiment
                            )
                            
                            posts_data.append((post_metrics, discussion_quality, controversy))
                            self.processed_posts.add(post.id)
                
                all_posts.extend(posts_data)
                logger.info(f"📊 r/{subreddit_name}: {len(posts_data)} discussioni di qualità")
                
                await asyncio.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"⚠️ Errore analisi r/{subreddit_name}: {e}")
        
        if not all_posts:
            logger.info("ℹ️ Nessuna discussione di qualità trovata")
            return None
        
        # Raggruppa per topic e trova le discussioni più trend
        trending_discussion = await self._identify_trending_discussion(all_posts)
        return trending_discussion

    async def _identify_trending_discussion(self, posts_data: List[Tuple]) -> Optional[TrendData]:
        """Identifica la discussione più trend tra i post analizzati"""
        if not posts_data:
            return None
        
        # Calcola score composito per ogni discussione
        discussion_scores = []
        
        for post_metrics, quality, controversy in posts_data:
            # Score basato su engagement, qualità e velocità
            base_score = post_metrics.engagement_rate * 0.4
            quality_score = quality * 0.3
            velocity_score = post_metrics.comment_velocity * 0.2
            diversity_score = min(post_metrics.comments / 100, 1) * 0.1
            
            total_score = (base_score + quality_score + velocity_score + diversity_score) * 100
            
            discussion_scores.append((post_metrics, quality, controversy, total_score))
        
        # Ordina per score e prendi la migliore
        discussion_scores.sort(key=lambda x: x[3], reverse=True)
        best_post, best_quality, best_controversy, best_score = discussion_scores[0]
        
        # Estrai keywords dal titolo per categorizzazione
        keywords = self._extract_discussion_keywords(best_post.title)
        category = self._categorize_discussion(best_post.title, keywords)
        
        # Crea TrendData
        trend = TrendData(
            topic=best_post.title[:50] + ('...' if len(best_post.title) > 50 else ''),
            category=category,
            total_score=int(best_post.score),
            velocity_score=best_post.comment_velocity,
            post_count=1,  # Una discussione principale
            comment_count=best_post.comments,
            avg_engagement=best_post.engagement_rate,
            top_posts=[{
                'title': best_post.title,
                'subreddit': best_post.subreddit,
                'score': best_post.score,
                'comments': best_post.comments,
                'engagement': round(best_post.engagement_rate, 1),
                'quality': round(best_quality, 1),
                'controversy': round(best_controversy, 1)
            }],
            subreddits=[best_post.subreddit],
            timestamp=datetime.now(),
            confidence_level=min(best_score / 100, 1.0),
            sentiment_score=best_post.sentiment,
            discussion_quality=best_quality,
            controversy_score=best_controversy
        )
        
        logger.info(f"🔥 DISCUSSIONE TREND: {trend.topic}")
        logger.info(f"   📊 Qualità: {best_quality:.1f} | Controversia: {best_controversy:.1f}")
        logger.info(f"   💬 Commenti: {best_post.comments} | Score: {best_score:.1f}")
        
        return trend

    def _extract_discussion_keywords(self, text: str) -> List[str]:
        """Estrae keywords significative dal titolo della discussione"""
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = [word for word in clean_text.split() 
                if len(word) > 3 and word not in self.analyzer.stop_words]
        return words[:5]

    def _categorize_discussion(self, title: str, keywords: List[str]) -> str:
        """Categorizza la discussione basandosi sul contenuto"""
        text = f"{title} {' '.join(keywords)}".lower()
        
        categories = {
            'social': ['relationship', 'friend', 'family', 'dating', 'marriage'],
            'philosophy': ['life', 'meaning', 'purpose', 'exist', 'think'],
            'politics': ['government', 'policy', 'vote', 'political', 'election'],
            'technology': ['tech', 'ai', 'internet', 'digital', 'phone'],
            'entertainment': ['movie', 'music', 'game', 'show', 'celebrity'],
            'education': ['learn', 'study', 'school', 'university', 'teach'],
            'health': ['health', 'medical', 'doctor', 'fitness', 'diet']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category
        
        return 'general'

    def format_discussion_alert(self, trend: TrendData) -> str:
        """Formatta un alert specifico per discussioni trend"""
        emoji = "💬" if trend.controversy_score < 50 else "⚡"
        quality_emoji = "🟢" if trend.discussion_quality > 70 else "🟡" if trend.discussion_quality > 40 else "🔴"
        
        alert = f"""{emoji} **DISCUSSIONE TREND SU REDDIT** {emoji}

🗣️ **Discussione**: {trend.topic}
📂 **Categoria**: {trend.category}
🏷️ **Subreddit**: r/{trend.subreddits[0]}

📊 **METRICHE DISCUSSIONE**
{quality_emoji} **Qualità**: {trend.discussion_quality:.1f}/100
⚖️ **Controversia**: {trend.controversy_score:.1f}/100
😊 **Sentiment**: {trend.sentiment_score:.1f}
⬆️ **Upvotes**: {trend.total_score:,}
💬 **Commenti**: {trend.comment_count:,}
🚀 **Velocità**: {trend.velocity_score:.1f} commenti/ora

🔍 **DETTAGLI**
• Engagement rate: {trend.avg_engagement:.1f}
• Confidenza trend: {trend.confidence_level:.1%}
• Timestamp: {trend.timestamp.strftime('%H:%M - %d/%m/%Y')}

💡 **Analisi**: {'Discussione di alta qualità' if trend.discussion_quality > 70 else 
                'Discussione attiva' if trend.discussion_quality > 40 else 
                'Discussione emergente'}"""

        return alert

    async def run_continuous_analysis(self):
        """Esegue l'analisi continua ogni 15 minuti"""
        logger.info("🚀 AVVIO ANALISI DISCUSSIONI TREND")
        logger.info("⏰ Frequenza: ogni 15 minuti")
        
        if not await self.initialize_reddit():
            return
        
        analysis_count = 0
        
        while True:
            try:
                analysis_count += 1
                start_time = datetime.now()
                
                logger.info(f"🔄 Analisi #{analysis_count} iniziata")
                
                # Trova discussioni trend
                trending_discussion = await self.analyze_discussion_trends()
                
                if trending_discussion:
                    # Formatta e invia alert
                    alert_message = self.format_discussion_alert(trending_discussion)
                    
                    if self.telegram_token:
                        await self.send_telegram_notification(alert_message)
                    
                    logger.info(f"✅ Analisi #{analysis_count} completata - Discussione trovata")
                else:
                    logger.info(f"ℹ️ Analisi #{analysis_count} completata - Nessuna discussione significativa")
                
                # Pulizia periodica della cache
                if analysis_count % 10 == 0:
                    if len(self.processed_posts) > 1000:
                        self.processed_posts = set(list(self.processed_posts)[-500:])
                    logger.info(f"🧹 Cache pulita: {len(self.processed_posts)} post")
                
                # Calcola tempo rimanente per i 15 minuti
                elapsed = (datetime.now() - start_time).total_seconds()
                wait_time = max(900 - elapsed, 60)  # Minimo 1 minuto di attesa
                
                logger.info(f"⏸️ Prossima analisi tra {wait_time/60:.1f} minuti")
                await asyncio.sleep(wait_time)
                
            except KeyboardInterrupt:
                logger.info("🛑 Interruzione manuale")
                break
            except Exception as e:
                logger.error(f"❌ Errore analisi #{analysis_count}: {e}")
                logger.info("⏸️ Ripresa in 5 minuti...")
                await asyncio.sleep(300)

    async def send_telegram_notification(self, message: str) -> bool:
        """Invia notifica Telegram"""
        if not self.telegram_token or not self.telegram_chat_id:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.info("✅ Notifica Telegram inviata")
                        return True
                    else:
                        logger.warning(f"⚠️ Errore Telegram: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"❌ Errore invio Telegram: {e}")
            return False

# Configurazione NLTK
def setup_nltk():
    """Configura NLTK per l'analisi del sentiment"""
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')

async def main():
    """Funzione principale"""
    logger.info("=" * 50)
    logger.info("💬 REDDIT DISCUSSION TREND BOT")
    logger.info("⏰ Rilevamento discussioni trend ogni 15 minuti")
    logger.info("=" * 50)
    
    # Configura NLTK
    setup_nltk()
    
    # Verifica credenziali
    if not os.getenv('REDDIT_CLIENT_ID') or not os.getenv('REDDIT_CLIENT_SECRET'):
        logger.error("❌ Configura REDDIT_CLIENT_ID e REDDIT_CLIENT_SECRET")
        return
    
    # Avvia il bot
    bot = ProfessionalRedditTrendBot()
    
    try:
        await bot.run_continuous_analysis()
    except KeyboardInterrupt:
        logger.info("👋 Bot fermato")
    except Exception as e:
        logger.error(f"🚨 Errore critico: {e}")

if __name__ == "__main__":
    asyncio.run(main())
