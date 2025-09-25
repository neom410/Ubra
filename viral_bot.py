import asyncpraw
import asyncio
import aiohttp
import json
import os
import logging
import math
from datetime import datetime, timedelta
from collections import Counter, defaultdict, deque
import re
import hashlib

# Configurazione logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('interest_analyzer.log')]
)
logger = logging.getLogger(__name__)

# ===== ADVANCED INTEREST ANALYZER (Senza sklearn) =====
class AdvancedInterestAnalyzer:
    def __init__(self):
        self.interests_file = 'user_interests.json'
        self.trends_file = 'interest_trends.json'
        self.history_file = 'interest_history.json'
        
        # Database degli interessi
        self.interest_database = self.load_interests()
        self.conversation_buffer = deque(maxlen=500)  # Ultime conversazioni
        self.interest_trends = defaultdict(lambda: {'count': 0, 'momentum': 0, 'history': []})
        
        # Parametri di analisi avanzata
        self.analysis_config = {
            'min_discussion_length': 30,  # Caratteri minimi per analisi
            'min_comments_threshold': 3,  # Commenti minimi per considerare "discussione"
            'engagement_weight': 2.0,     # Peso engagement nelle discussioni
            'time_decay_hours': 72,       # Decadimento interesse nel tempo
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
        
        # Stop words per pulizia testo
        self.stop_words = {
            'the', 'and', 'for', 'with', 'this', 'that', 'have', 'has', 'was', 'were', 
            'are', 'you', 'your', 'about', 'from', 'their', 'they', 'been', 'will', 
            'would', 'should', 'could', 'what', 'when', 'where', 'why', 'how', 'which',
            'who', 'whom', 'there', 'here', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'must', 'can', 'cannot', 'couldnt', 'wouldnt', 'shouldnt', 'mightnt'
        }

    def get_tech_keywords(self):
        return ['ai', 'artificial intelligence', 'machine learning', 'programming', 'software', 
                'hardware', 'startup', 'innovation', 'digital', 'tech', 'coding', 'developer',
                'app', 'website', 'cloud', 'data science', 'cybersecurity', 'blockchain', 'computer']

    def get_gaming_keywords(self):
        return ['game', 'gaming', 'playstation', 'xbox', 'nintendo', 'steam', 'esports',
                'video game', 'gamer', 'multiplayer', 'singleplayer', 'release', 'update', 'console']

    def get_entertainment_keywords(self):
        return ['movie', 'film', 'tv show', 'netflix', 'youtube', 'music', 'spotify',
                'celebrity', 'actor', 'director', 'album', 'song', 'streaming', 'entertainment']

    def get_lifestyle_keywords(self):
        return ['fitness', 'workout', 'diet', 'nutrition', 'travel', 'food', 'cooking',
                'fashion', 'beauty', 'home', 'decor', 'gardening', 'pet', 'hobby', 'lifestyle']

    def get_science_keywords(self):
        return ['science', 'research', 'discovery', 'space', 'physics', 'biology',
                'chemistry', 'astronomy', 'innovation', 'study', 'scientist', 'experiment']

    def get_business_keywords(self):
        return ['business', 'startup', 'entrepreneur', 'marketing', 'sales', 'company',
                'industry', 'market', 'economy', 'investment', 'strategy', 'profit']

    def get_politics_keywords(self):
        return ['politics', 'government', 'election', 'policy', 'law', 'democrat',
                'republican', 'parliament', 'senate', 'vote', 'political', 'policy']

    def get_social_keywords(self):
        return ['social', 'society', 'community', 'culture', 'relationship', 'friendship',
                'network', 'communication', 'media', 'social media', 'connection', 'social']

    def get_life_keywords(self):
        return ['life', 'personal', 'experience', 'story', 'advice', 'help', 'support',
                'mental health', 'self improvement', 'motivation', 'happiness', 'life']

    def get_job_keywords(self):
        return ['job', 'career', 'work', 'employment', 'hire', 'recruitment', 'salary',
                'interview', 'resume', 'career change', 'remote work', 'promotion', 'work']

    def get_education_keywords(self):
        return ['education', 'school', 'university', 'student', 'learn', 'study',
                'course', 'teacher', 'academic', 'degree', 'online learning', 'education']

    def get_health_keywords(self):
        return ['health', 'medical', 'doctor', 'hospital', 'treatment', 'medicine',
                'wellness', 'fitness', 'nutrition', 'mental health', 'therapy', 'health']

    def get_environment_keywords(self):
        return ['environment', 'climate', 'sustainability', 'eco', 'green', 'pollution',
                'renewable', 'energy', 'conservation', 'nature', 'planet', 'environment']

    def get_finance_keywords(self):
        return ['finance', 'money', 'investment', 'stock', 'crypto', 'bank',
                'saving', 'budget', 'financial', 'wealth', 'retirement', 'finance']

    def get_relationships_keywords(self):
        return ['relationship', 'dating', 'marriage', 'friend', 'family', 'love',
                'partner', 'communication', 'breakup', 'friendship', 'relationship']

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
        if not hasattr(post, 'num_comments') or post.num_comments < self.analysis_config['min_comments_threshold']:
            return 0
        
        # Calcola engagement score basato su commenti/upvotes ratio
        engagement_score = min(post.num_comments / max(1, post.score), 10)
        
        # Bonus per discussioni lunghe
        text_length = len(getattr(post, 'selftext', ''))
        if text_length > 500:
            engagement_score *= 1.5
        elif text_length > 200:
            engagement_score *= 1.2
        
        return engagement_score

    def categorize_topic(self, title, subreddit):
        """Categorizza il topic basandosi su titolo e subreddit"""
        text_lower = f"{title} {subreddit}".lower()
        category_scores = defaultdict(int)
        
        for category, keywords in self.interest_categories.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Punteggio più alto per match esatti
                    if f" {keyword} " in f" {text_lower} ":
                        category_scores[category] += 3
                    else:
                        category_scores[category] += 1
        
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            return best_category[0] if best_category[1] > 0 else 'general'
        
        return 'general'

    def extract_interest_patterns(self, text, category, engagement_score):
        """Estrae pattern di interesse dal testo"""
        text_lower = text.lower()
        patterns = {}
        
        # Analizza frequenza parole chiave per categoria
        for keyword in self.interest_categories.get(category, []):
            if keyword in text_lower:
                frequency = text_lower.count(keyword)
                importance = frequency * engagement_score
                patterns[keyword] = {
                    'frequency': frequency,
                    'importance': importance,
                    'engagement': engagement_score,
                    'timestamp': datetime.now().isoformat()
                }
        
        return patterns

    def calculate_interest_momentum(self, category, new_engagement):
        """Calcola il momentum di interesse per una categoria"""
        current_trend = self.interest_trends[category]
        
        # Applica decadimento temporale
        current_count = current_trend['count'] * 0.85  # Decadimento più aggressivo
        new_count = current_count + new_engagement
        
        # Calcola momentum (cambiamento recente)
        momentum = new_engagement - (current_trend['count'] * 0.9)
        
        self.interest_trends[category] = {
            'count': new_count,
            'momentum': momentum,
            'history': current_trend['history'][-49:] + [{'count': new_count, 'timestamp': datetime.now().isoformat()}]
        }
        
        return momentum

    def identify_emerging_interests(self):
        """Identifica interessi emergenti basati su momentum"""
        emerging = []
        
        for category, trend in self.interest_trends.items():
            if len(trend['history']) >= 2:
                recent_growth = trend['momentum']
                if len(trend['history']) >= 3:
                    avg_growth = sum(h['count'] for h in trend['history'][-3:]) / 3
                else:
                    avg_growth = trend['count']
                
                if recent_growth > 0 and recent_growth > avg_growth * self.analysis_config['emerging_threshold']:
                    emerging.append({
                        'category': category,
                        'momentum': recent_growth,
                        'strength': trend['count'],
                        'trend': 'emerging'
                    })
        
        return sorted(emerging, key=lambda x: x['momentum'], reverse=True)

    def simple_text_clustering(self, conversations):
        """Clustering semplificato senza ML"""
        if len(conversations) < 5:
            return []
        
        # Estrai parole chiave comuni
        all_text = ' '.join([conv['title'] for conv in conversations]).lower()
        words = re.findall(r'\b[a-z]{4,15}\b', all_text)
        
        # Filtra stop words
        meaningful_words = [w for w in words if w not in self.stop_words]
        
        # Trova parole più frequenti
        word_freq = Counter(meaningful_words)
        common_topics = [word for word, count in word_freq.most_common(10) if count > 1]
        
        if not common_topics:
            return []
        
        # Raggruppa conversazioni per topic simili
        clusters = []
        for topic in common_topics[:5]:
            related_conversations = [
                conv for conv in conversations 
                if topic in conv['title'].lower() or topic in conv.get('text', '').lower()
            ]
            
            if related_conversations:
                clusters.append({
                    'topic': topic,
                    'size': len(related_conversations),
                    'sample_titles': [conv['title'][:60] + '...' for conv in related_conversations[:2]]
                })
        
        return clusters

    def analyze_user_sentiment(self, text):
        """Analizza il sentiment del testo (semplificato)"""
        positive_words = ['love', 'great', 'amazing', 'awesome', 'good', 'excellent', 
                         'happy', 'best', 'fantastic', 'wonderful', 'perfect', 'brilliant']
        negative_words = ['hate', 'terrible', 'awful', 'bad', 'worst', 'angry', 
                         'sad', 'disappointing', 'horrible', 'stupid', 'ridiculous']
        
        text_lower = text.lower()
        positive_score = sum(1 for word in positive_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)
        
        total_score = positive_score - negative_score
        
        if total_score > 2:
            return 'very positive'
        elif total_score > 0:
            return 'positive'
        elif total_score < -2:
            return 'very negative'
        elif total_score < 0:
            return 'negative'
        else:
            return 'neutral'

    def process_discussion(self, post, category):
        """Processa una discussione completa per estrarre interessi"""
        discussion_text = f"{post.title} {getattr(post, 'selftext', '')}"
        
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
            'text': discussion_text[:500],  # Salva solo prime 500 caratteri
            'category': category,
            'engagement_score': engagement_score,
            'sentiment': sentiment,
            'interest_patterns': interest_patterns,
            'momentum': momentum,
            'timestamp': datetime.now().isoformat(),
            'comment_count': getattr(post, 'num_comments', 0),
            'upvotes': getattr(post, 'score', 0),
            'subreddit': getattr(post, 'subreddit', 'unknown')
        }
        
        # Aggiungi al buffer delle conversazioni
        self.conversation_buffer.append(discussion_data)
        
        # Aggiorna database (mantieni solo ultime 1000)
        self.interest_database['conversations'] = self.interest_database.get('conversations', [])[-999:] + [discussion_data]
        
        return discussion_data

    def get_interest_insights(self):
        """Genera insights sugli interessi degli utenti"""
        emerging_interests = self.identify_emerging_interests()
        cluster_analysis = self.simple_text_clustering(list(self.conversation_buffer))
        
        # Calcola interessi più popolari
        category_popularity = Counter()
        category_sentiment = defaultdict(list)
        
        for conv in self.conversation_buffer:
            category_popularity[conv['category']] += conv['engagement_score']
            category_sentiment[conv['category']].append(conv['sentiment'])
        
        # Calcola sentiment medio per categoria
        avg_sentiment = {}
        for category, sentiments in category_sentiment.items():
            sentiment_counts = Counter(sentiments)
            avg_sentiment[category] = dict(sentiment_counts)
        
        popular_interests = [{'category': cat, 'score': score, 'sentiment': avg_sentiment.get(cat, {})} 
                           for cat, score in category_popularity.most_common(8)]
        
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
                user_agent='IntelligentInterestHunter/2.0'
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
            
            for subreddit_name in self.analysis_subreddits[:10]:  # Limita per performance
                try:
                    subreddit = await self.reddit.subreddit(subreddit_name)
                    
                    async for post in subreddit.hot(limit=15):
                        if post.id in self.processed_posts:
                            continue
                            
                        post_time = datetime.fromtimestamp(post.created_utc)
                        hours_ago = (current_time - post_time).total_seconds() / 3600
                        
                        # Analizza solo discussioni recenti e con engagement
                        if (hours_ago <= 96 and post.num_comments >= 5 and 
                            post.score >= 10 and not post.stickied):
                            
                            # Categorizza il post
                            category = self.analyzer.categorize_topic(post.title, str(post.subreddit))
                            
                            # Analisi approfondita della discussione
                            discussion_analysis = self.analyzer.process_discussion(post, category)
                            
                            if discussion_analysis:
                                analyzed_count += 1
                                self.processed_posts.add(post.id)
                            
                        if analyzed_count >= 30:  # Limite per sessione
                            break
                            
                except Exception as e:
                    logger.warning(f"Errore analisi r/{subreddit_name}: {e}")
                    continue
                
                if analyzed_count >= 30:
                    break
            
            # Genera insights dopo l'analisi
            insights = self.analyzer.get_interest_insights()
            
            logger.info(f"Analizzate {analyzed_count} discussioni per interessi")
            if insights['emerging_interests']:
                logger.info(f"Trovati {len(insights['emerging_interests'])} interessi emergenti")
            
            return insights
            
        except Exception as e:
            logger.error(f"Errore analisi interessi: {e}")
            return None

    def format_interest_report(self, insights):
        """Formatta il report degli interessi"""
        if not insights:
            return "Nessun dato di interesse analizzato ancora. Raccolta dati in corso..."
        
        message = "🧠 **ANALISI INTERESSI UTENTI REDDIT** 🧠\n\n"
        message += f"📊 **Panorama Interessi**\n"
        message += f"• Discussioni analizzate: {insights['total_conversations_analyzed']}\n"
        message += f"• Timestamp: {datetime.fromisoformat(insights['timestamp']).strftime('%H:%M - %d/%m/%Y')}\n\n"
        
        # Interessi emergenti
        if insights['emerging_interests']:
            message += "🚀 **INTERESSI EMERGENTI**\n"
            for interest in insights['emerging_interests'][:3]:
                message += f"• {interest['category'].upper()}: +{interest['momentum']:.1f} momentum\n"
            message += "\n"
        
        # Interessi popolari
        if insights['popular_interests']:
            message += "🔥 **INTERESSI PIÙ DISCUSSI**\n"
            for interest in insights['popular_interests'][:4]:
                # Calcola sentiment predominante
                sentiment = interest.get('sentiment', {})
                if sentiment:
                    main_sentiment = max(sentiment.items(), key=lambda x: x[1])[0]
                    sentiment_emoji = '😊' if 'positive' in main_sentiment else '😐' if 'neutral' in main_sentiment else '😞'
                else:
                    sentiment_emoji = '😐'
                
                message += f"• {interest['category']}: {interest['score']:.1f} engagement {sentiment_emoji}\n"
            message += "\n"
        
        # Cluster di discussione
        if insights['discussion_clusters']:
            message += "🎯 **ARGOMENTI PRINCIPALI**\n"
            for cluster in insights['discussion_clusters'][:2]:
                message += f"• Topic: {cluster['topic']} ({cluster['size']} discussioni)\n"
            message += "\n"
        
        # Insight finale
        message += "📈 **COSA INTERESSA REALMENTE LE PERSONE:**\n"
        
        if insights['popular_interests']:
            top_interest = insights['popular_interests'][0]
            message += f"• **{top_interest['category']}** è l'argomento più discusso\n"
        
        if insights['emerging_interests']:
            emerging = insights['emerging_interests'][0]
            message += f"• **{emerging['category']}** sta crescendo rapidamente\n"
        
        message += f"\n🔄 Prossima analisi: 15 minuti"
        
        return message

    async def send_to_telegram(self, message):
        """Invia il report via Telegram"""
        if not self.telegram_token or not self.active_chats:
            return False
        
        success_count = 0
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            for chat_id in self.active_chats:
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
                except Exception as e:
                    logger.error(f"Errore invio Telegram: {e}")
        
        return success_count > 0

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
                
                if insights and insights['total_conversations_analyzed'] > 0:
                    report = self.format_interest_report(insights)
                    logger.info("Analisi interessi completata")
                    
                    # Salva periodicamente
                    if analysis_count % 5 == 0:
                        self.analyzer.save_interests()
                        logger.info("Dati interessi salvati")
                    
                    # Invia report se ci sono chat attive
                    if self.telegram_token:
                        await self.send_to_telegram(report)
                
                # Pulizia periodica
                if len(self.processed_posts) > 1000:
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
    logger.info("Nessuna dipendenza ML esterna - Solo Python nativo")
    asyncio.run(main())
