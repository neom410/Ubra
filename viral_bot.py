import asyncpraw
import asyncio
import aiohttp
import json
import os
import logging
import numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict, deque
import statistics
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import threading
from typing import Dict, List, Set, Any
import hashlib

# Configurazione logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('interest_analyzer.log')]
)
logger = logging.getLogger(__name__)

# ===== ADVANCED INTEREST ANALYZER =====
class AdvancedInterestAnalyzer:
    def __init__(self):
        self.interests_file = 'user_interests.json'
        self.trends_file = 'interest_trends.json'
        self.history_file = 'interest_history.json'
        
        # Database degli interessi
        self.interest_database = self.load_interests()
        self.conversation_buffer = deque(maxlen=1000)  # Ultime conversazioni
        self.interest_trends = defaultdict(lambda: {'count': 0, 'momentum': 0, 'history': []})
        
        # Parametri di analisi avanzata
        self.analysis_config = {
            'min_discussion_length': 50,  # Caratteri minimi per analisi
            'min_comments_threshold': 5,  # Commenti minimi per considerare "discussione"
            'engagement_weight': 2.0,     # Peso engagement nelle discussioni
            'time_decay_hours': 72,       # Decadimento interesse nel tempo
            'cluster_topics': 8,          # Numero di cluster per topic discovery
            'emerging_threshold': 0.15    # Soglia per trend emergenti
        }
        
        # Categorie di interesse estese
        self.interest_categories = {
            'technology': self.get_tech_keywords(),
            'gaming': self.get_gaming_keywords(),
            'entertainment': self.get_entertainment_keywords(),
            'lifestyle': self.get_lifestyle_keywords(),
            'science': self.get_science_keywords(),
            'business': self.get_business_keywords(),
            'politica': self.get_politics_keywords(),
            'social': self.get_social_keywords(),
            'life': self.get_life_keywords(),
            'mercato lavoro': self.get_job_keywords(),
            'education': self.get_education_keywords(),
            'health': self.get_health_keywords(),
            'environment': self.get_environment_keywords(),
            'finance': self.get_finance_keywords(),
            'relationships': self.get_relationships_keywords()
        }
        
        # ML Components
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        self.cluster_model = None
        self.is_training = False

    def get_tech_keywords(self):
        return ['ai', 'artificial intelligence', 'machine learning', 'programming', 'software', 
                'hardware', 'startup', 'innovation', 'digital', 'tech', 'coding', 'developer',
                'app', 'website', 'cloud', 'data science', 'cybersecurity', 'blockchain']

    def get_gaming_keywords(self):
        return ['game', 'gaming', 'playstation', 'xbox', 'nintendo', 'steam', 'esports',
                'video game', 'gamer', 'multiplayer', 'singleplayer', 'release', 'update']

    def get_entertainment_keywords(self):
        return ['movie', 'film', 'tv show', 'netflix', 'youtube', 'music', 'spotify',
                'celebrity', 'actor', 'director', 'album', 'song', 'streaming']

    def get_lifestyle_keywords(self):
        return ['fitness', 'workout', 'diet', 'nutrition', 'travel', 'food', 'cooking',
                'fashion', 'beauty', 'home', 'decor', 'gardening', 'pet', 'hobby']

    def get_science_keywords(self):
        return ['science', 'research', 'discovery', 'space', 'physics', 'biology',
                'chemistry', 'astronomy', 'innovation', 'study', 'scientist']

    def get_business_keywords(self):
        return ['business', 'startup', 'entrepreneur', 'marketing', 'sales', 'company',
                'industry', 'market', 'economy', 'investment', 'strategy']

    def get_politics_keywords(self):
        return ['politics', 'government', 'election', 'policy', 'law', 'democrat',
                'republican', 'parliament', 'senate', 'vote', 'political']

    def get_social_keywords(self):
        return ['social', 'society', 'community', 'culture', 'relationship', 'friendship',
                'network', 'communication', 'media', 'social media', 'connection']

    def get_life_keywords(self):
        return ['life', 'personal', 'experience', 'story', 'advice', 'help', 'support',
                'mental health', 'self improvement', 'motivation', 'happiness']

    def get_job_keywords(self):
        return ['job', 'career', 'work', 'employment', 'hire', 'recruitment', 'salary',
                'interview', 'resume', 'career change', 'remote work', 'promotion']

    def get_education_keywords(self):
        return ['education', 'school', 'university', 'student', 'learn', 'study',
                'course', 'teacher', 'academic', 'degree', 'online learning']

    def get_health_keywords(self):
        return ['health', 'medical', 'doctor', 'hospital', 'treatment', 'medicine',
                'wellness', 'fitness', 'nutrition', 'mental health', 'therapy']

    def get_environment_keywords(self):
        return ['environment', 'climate', 'sustainability', 'eco', 'green', 'pollution',
                'renewable', 'energy', 'conservation', 'nature', 'planet']

    def get_finance_keywords(self):
        return ['finance', 'money', 'investment', 'stock', 'crypto', 'bank',
                'saving', 'budget', 'financial', 'wealth', 'retirement']

    def get_relationships_keywords(self):
        return ['relationship', 'dating', 'marriage', 'friend', 'family', 'love',
                'partner', 'communication', 'breakup', 'friendship']

    def load_interests(self):
        try:
            if os.path.exists(self.interests_file):
                with open(self.interests_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Errore caricamento interessi: {e}")
        return {'conversations': [], 'interest_patterns': {}, 'user_profiles': {}}

    def save_interests(self):
        try:
            with open(self.interests_file, 'w') as f:
                json.dump(self.interest_database, f, indent=2)
        except Exception as e:
            logger.error(f"Errore salvataggio interessi: {e}")

    def analyze_conversation_depth(self, post):
        """Analizza la profondità e qualità della discussione"""
        if not hasattr(post, 'comments') or post.num_comments < self.analysis_config['min_comments_threshold']:
            return 0
        
        # Calcola engagement score basato su commenti/upvotes ratio
        engagement_score = min(post.num_comments / max(1, post.score), 10)
        
        # Considera la lunghezza media dei commenti (se disponibile)
        discussion_quality = engagement_score * self.analysis_config['engagement_weight']
        
        return discussion_quality

    def extract_interest_patterns(self, text, category, engagement_score):
        """Estrae pattern di interesse dal testo"""
        text_lower = text.lower()
        patterns = {}
        
        # Analizza frequenza parole chiave per categoria
        for keyword in self.interest_categories.get(category, []):
            if keyword in text_lower:
                frequency = text_lower.count(keyword)
                patterns[keyword] = {
                    'frequency': frequency,
                    'engagement': engagement_score,
                    'timestamp': datetime.now().isoformat()
                }
        
        return patterns

    def calculate_interest_momentum(self, category, new_engagement):
        """Calcola il momentum di interesse per una categoria"""
        current_trend = self.interest_trends[category]
        time_decay = self.analysis_config['time_decay_hours']
        
        # Applica decadimento temporale
        current_count = current_trend['count'] * 0.9  # Decadimento
        new_count = current_count + new_engagement
        
        # Calcola momentum (cambiamento recente)
        momentum = new_engagement - current_trend['count']
        
        self.interest_trends[category] = {
            'count': new_count,
            'momentum': momentum,
            'history': current_trend['history'][-99:] + [{'count': new_count, 'timestamp': datetime.now().isoformat()}]
        }
        
        return momentum

    def identify_emerging_interests(self):
        """Identifica interessi emergenti basati su momentum"""
        emerging = []
        
        for category, trend in self.interest_trends.items():
            if len(trend['history']) >= 3:
                recent_growth = trend['momentum']
                avg_growth = np.mean([h['count'] for h in trend['history'][-3:]])
                
                if recent_growth > avg_growth * self.analysis_config['emerging_threshold']:
                    emerging.append({
                        'category': category,
                        'momentum': recent_growth,
                        'strength': trend['count'],
                        'trend': 'emerging'
                    })
        
        return sorted(emerging, key=lambda x: x['momentum'], reverse=True)

    def cluster_similar_interests(self, conversations):
        """Clusterizza conversazioni simili usando ML"""
        if len(conversations) < 10:
            return []
        
        try:
            texts = [conv['text'] for conv in conversations if len(conv['text']) > 50]
            
            if len(texts) < 5:
                return []
            
            # Vectorizzazione TF-IDF
            X = self.vectorizer.fit_transform(texts)
            
            # Clustering K-means
            n_clusters = min(self.analysis_config['cluster_topics'], len(texts))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X)
            
            # Estrai topic principali per cluster
            feature_names = self.vectorizer.get_feature_names_out()
            cluster_topics = []
            
            for i in range(n_clusters):
                cluster_indices = np.where(clusters == i)[0]
                if len(cluster_indices) > 0:
                    # Trova parole più importanti per il cluster
                    cluster_center = kmeans.cluster_centers_[i]
                    top_indices = cluster_center.argsort()[-10:][::-1]
                    top_words = [feature_names[idx] for idx in top_indices]
                    
                    cluster_topics.append({
                        'cluster_id': i,
                        'size': len(cluster_indices),
                        'top_words': top_words,
                        'sample_titles': [conversations[idx]['title'] for idx in cluster_indices[:3]]
                    })
            
            return cluster_topics
            
        except Exception as e:
            logger.error(f"Errore clustering: {e}")
            return []

    def analyze_user_sentiment(self, text):
        """Analizza il sentiment del testo (semplificato)"""
        positive_words = ['love', 'great', 'amazing', 'awesome', 'good', 'excellent', 'happy', 'best']
        negative_words = ['hate', 'terrible', 'awful', 'bad', 'worst', 'angry', 'sad', 'disappointing']
        
        text_lower = text.lower()
        positive_score = sum(1 for word in positive_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)
        
        sentiment = positive_score - negative_score
        return 'positive' if sentiment > 0 else 'negative' if sentiment < 0 else 'neutral'

    def process_discussion(self, post, category):
        """Processa una discussione completa per estrarre interessi"""
        discussion_text = f"{post.title} {post.selftext}"
        
        if len(discussion_text) < self.analysis_config['min_discussion_length']:
            return None
        
        engagement_score = self.analyze_conversation_depth(post)
        sentiment = self.analyze_user_sentiment(discussion_text)
        
        # Estrai pattern di interesse
        interest_patterns = self.extract_interest_patterns(discussion_text, category, engagement_score)
        
        # Calcola momentum
        momentum = self.calculate_interest_momentum(category, engagement_score)
        
        discussion_data = {
            'id': post.id,
            'title': post.title,
            'category': category,
            'engagement_score': engagement_score,
            'sentiment': sentiment,
            'interest_patterns': interest_patterns,
            'momentum': momentum,
            'timestamp': datetime.now().isoformat(),
            'comment_count': getattr(post, 'num_comments', 0),
            'upvotes': getattr(post, 'score', 0)
        }
        
        # Aggiungi al buffer delle conversazioni
        self.conversation_buffer.append(discussion_data)
        
        # Aggiorna database
        self.interest_database['conversations'].append(discussion_data)
        
        return discussion_data

    def get_interest_insights(self):
        """Genera insights sugli interessi degli utenti"""
        emerging_interests = self.identify_emerging_interests()
        cluster_analysis = self.cluster_similar_interests(list(self.conversation_buffer))
        
        # Calcola interessi più popolari
        category_popularity = Counter()
        for conv in self.conversation_buffer:
            category_popularity[conv['category']] += conv['engagement_score']
        
        popular_interests = [{'category': cat, 'score': score} 
                           for cat, score in category_popularity.most_common(5)]
        
        return {
            'emerging_interests': emerging_interests[:5],
            'popular_interests': popular_interests,
            'discussion_clusters': cluster_analysis,
            'total_conversations_analyzed': len(self.conversation_buffer),
            'timestamp': datetime.now().isoformat()
        }

# ===== INTELLIGENT INTEREST HUNTER =====
class IntelligentInterestHunter:
    def __init__(self):
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        
        if not all([self.reddit_client_id, self.reddit_client_secret]):
            raise ValueError("Credenziali Reddit mancanti!")
        
        self.analyzer = AdvancedInterestAnalyzer()
        self.active_chats = set()
        self.processed_posts = set()
        
        # Subreddit diversificati per analisi interessi
        self.analysis_subreddits = [
            'all', 'popular', 'askreddit', 'questions', 'discussion',
            'technology', 'gaming', 'science', 'worldnews', 'politics',
            'personalfinance', 'relationships', 'careerguidance', 'lifeadvice',
            'selfimprovement', 'mentalhealth', 'learnprogramming', 'explainlikeimfive',
            'todayilearned', 'youshouldknow', 'lifehacks', 'socialskills',
            'jobs', 'business', 'entrepreneur', 'startups', 'investing',
            'health', 'fitness', 'travel', 'food', 'music', 'movies',
            'books', 'sports', 'space', 'environment', 'education'
        ]

    async def initialize_reddit(self):
        try:
            self.reddit = asyncpraw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent='IntelligentInterestHunter/1.0'
            )
            logger.info("Reddit connesso per analisi interessi avanzata")
            return True
        except Exception as e:
            logger.error(f"Errore connessione Reddit: {e}")
            return False

    async def deep_analyze_interests(self):
        """Analisi approfondita degli interessi dalle discussioni Reddit"""
        try:
            current_time = datetime.now()
            analyzed_count = 0
            
            for subreddit_name in self.analysis_subreddits[:12]:  # Limita per performance
                try:
                    subreddit = await self.reddit.subreddit(subreddit_name)
                    
                    async for post in subreddit.hot(limit=20):
                        if post.id in self.processed_posts:
                            continue
                            
                        post_time = datetime.fromtimestamp(post.created_utc)
                        hours_ago = (current_time - post_time).total_seconds() / 3600
                        
                        # Analizza solo discussioni recenti e con engagement
                        if (hours_ago <= 72 and post.num_comments >= 10 and 
                            post.score >= 20 and not post.stickied):
                            
                            # Categorizza il post
                            category = self.analyzer.categorize_topic(post.title, subreddit_name)
                            
                            # Analisi approfondita della discussione
                            discussion_analysis = self.analyzer.process_discussion(post, category)
                            
                            if discussion_analysis:
                                analyzed_count += 1
                                self.processed_posts.add(post.id)
                            
                        if analyzed_count >= 50:  # Limite per sessione
                            break
                            
                except Exception as e:
                    logger.warning(f"Errore analisi r/{subreddit_name}: {e}")
                    continue
                
                if analyzed_count >= 50:
                    break
            
            # Genera insights dopo l'analisi
            insights = self.analyzer.get_interest_insights()
            
            logger.info(f"Analizzate {analyzed_count} discussioni per interessi")
            logger.info(f"Trovati {len(insights['emerging_interests'])} interessi emergenti")
            
            return insights
            
        except Exception as e:
            logger.error(f"Errore analisi interessi: {e}")
            return None

    def format_interest_report(self, insights):
        """Formatta il report degli interessi"""
        if not insights:
            return "Nessun dato di interesse analizzato ancora."
        
        message = "🧠 **ANALISI INTERESSI UTENTI REDDIT** 🧠\n\n"
        message += f"📊 **Panorama Interessi**\n"
        message += f"• Discussioni analizzate: {insights['total_conversations_analyzed']}\n"
        message += f"• Timestamp: {datetime.fromisoformat(insights['timestamp']).strftime('%H:%M - %d/%m/%Y')}\n\n"
        
        # Interessi emergenti
        if insights['emerging_interests']:
            message += "🚀 **INTERESSI EMERGENTI**\n"
            for interest in insights['emerging_interests'][:3]:
                message += f"• {interest['category'].upper()}: Momentum +{interest['momentum']:.1f}\n"
            message += "\n"
        
        # Interessi popolari
        if insights['popular_interests']:
            message += "🔥 **INTERESSI POPOLARI**\n"
            for interest in insights['popular_interests'][:3]:
                message += f"• {interest['category']}: Score {interest['score']:.1f}\n"
            message += "\n"
        
        # Cluster di discussione
        if insights['discussion_clusters']:
            message += "🎯 **TOPIC PRINCIPALI DELLE DISCUSSIONI**\n"
            for cluster in insights['discussion_clusters'][:2]:
                message += f"• {' '.join(cluster['top_words'][:5])}\n"
                message += f"  Esempio: {cluster['sample_titles'][0][:60]}...\n"
            message += "\n"
        
        message += "📈 **COSA INTERESSA REALMENTE LE PERSONE:**\n"
        
        # Insights specifici basati sui dati
        if insights['popular_interests']:
            top_interest = insights['popular_interests'][0]
            message += f"• Le persone sono più interessate a **{top_interest['category']}**\n"
        
        if insights['emerging_interests']:
            emerging = insights['emerging_interests'][0]
            message += f"• **{emerging['category']}** sta crescendo rapidamente\n"
        
        message += f"\n🔄 Prossima analisi: 15 minuti"
        
        return message

    async def run_interest_analysis(self):
        """Esegue l'analisi continua degli interessi"""
        logger.info("Avvio Intelligent Interest Hunter")
        logger.info("Analisi interessi ogni 15 minuti")
        
        if not await self.initialize_reddit():
            return
        
        logger.info("Interest Hunter operativo!")
        
        analysis_count = 0
        
        while True:
            try:
                analysis_count += 1
                logger.info(f"Analisi interessi #{analysis_count} in corso...")
                
                insights = await self.deep_analyze_interests()
                
                if insights:
                    report = self.format_interest_report(insights)
                    logger.info("Analisi interessi completata")
                    
                    # Salva periodicamente
                    if analysis_count % 10 == 0:
                        self.analyzer.save_interests()
                    
                    # Invia report se ci sono chat attive
                    if self.active_chats and self.telegram_token:
                        await self.send_to_telegram(report)
                
                # Pulizia periodica
                if len(self.processed_posts) > 2000:
                    self.processed_posts.clear()
                    logger.info("Pulizia post processati effettuata")
                
                await asyncio.sleep(900)  # 15 minuti
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Errore main loop: {e}")
                await asyncio.sleep(300)
        
        # Salva prima di chiudere
        self.analyzer.save_interests()
        if hasattr(self, 'reddit'):
            await self.reddit.close()

# Esegui l'analisi
async def main():
    try:
        hunter = IntelligentInterestHunter()
        await hunter.run_interest_analysis()
    except Exception as e:
        logger.error(f"Errore critico: {e}")

if __name__ == "__main__":
    logger.info("Intelligent Interest Hunter v2.0")
    logger.info("Analisi avanzata interessi utenti da discussioni Reddit")
    asyncio.run(main())
