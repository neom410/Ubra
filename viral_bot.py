import asyncpraw
import asyncio
import aiohttp
import json
import os
import logging
from datetime import datetime, timedelta
from collections import Counter, defaultdict, deque
import re

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
        
        self.interest_database = self.load_interests()
        self.conversation_buffer = deque(maxlen=200)
        self.interest_trends = defaultdict(lambda: {'count': 0, 'momentum': 0, 'history': [], 'specific_topics': []})
        
        self.analysis_config = {
            'min_discussion_length': 30,
            'min_comments_threshold': 3,
            'engagement_weight': 2.0,
            'emerging_threshold': 0.1
        }
        
        # Categorie più specifiche con keyword dettagliate
        self.interest_categories = {
            'technology': {
                'keywords': ['ai', 'artificial intelligence', 'machine learning', 'programming', 'software', 
                           'hardware', 'startup', 'innovation', 'digital', 'tech', 'coding', 'developer',
                           'app', 'website', 'cloud', 'data science', 'cybersecurity', 'blockchain', 'computer'],
                'subtopics': ['AI', 'programmazione', 'cybersecurity', 'blockchain', 'cloud computing', 'mobile app']
            },
            'gaming': {
                'keywords': ['game', 'gaming', 'playstation', 'xbox', 'nintendo', 'steam', 'esports',
                           'video game', 'gamer', 'multiplayer', 'singleplayer', 'release', 'update', 'console'],
                'subtopics': ['PS5', 'Xbox Series X', 'Nintendo Switch', 'PC gaming', 'eSports', 'game release']
            },
            'entertainment': {
                'keywords': ['movie', 'film', 'tv show', 'netflix', 'youtube', 'music', 'spotify',
                           'celebrity', 'actor', 'director', 'album', 'song', 'streaming', 'entertainment'],
                'subtopics': ['Netflix series', 'YouTube creators', 'music album', 'film release', 'celebrity news']
            },
            'lifestyle': {
                'keywords': ['fitness', 'workout', 'diet', 'nutrition', 'travel', 'food', 'cooking',
                           'fashion', 'beauty', 'home', 'decor', 'gardening', 'pet', 'hobby', 'lifestyle'],
                'subtopics': ['fitness routine', 'travel destinations', 'recipes', 'fashion trends', 'home decor']
            },
            'science': {
                'keywords': ['science', 'research', 'discovery', 'space', 'physics', 'biology',
                           'chemistry', 'astronomy', 'innovation', 'study', 'scientist', 'experiment'],
                'subtopics': ['space exploration', 'scientific discovery', 'climate research', 'medical breakthrough']
            },
            'business': {
                'keywords': ['business', 'startup', 'entrepreneur', 'marketing', 'sales', 'company',
                           'industry', 'market', 'economy', 'investment', 'strategy', 'profit'],
                'subtopics': ['startup funding', 'market trends', 'business strategy', 'investment opportunities']
            },
            'politica': {
                'keywords': ['politics', 'government', 'election', 'policy', 'law', 'democrat',
                           'republican', 'parliament', 'senate', 'vote', 'political', 'policy'],
                'subtopics': ['elections', 'government policy', 'political scandal', 'international relations']
            },
            'social': {
                'keywords': ['social', 'society', 'community', 'culture', 'relationship', 'friendship',
                           'network', 'communication', 'media', 'social media', 'connection', 'social'],
                'subtopics': ['social media trends', 'community issues', 'relationship advice', 'cultural topics']
            },
            'life': {
                'keywords': ['life', 'personal', 'experience', 'story', 'advice', 'help', 'support',
                           'mental health', 'self improvement', 'motivation', 'happiness', 'life'],
                'subtopics': ['mental health', 'personal growth', 'life advice', 'self improvement tips']
            },
            'mercato lavoro': {
                'keywords': ['job', 'career', 'work', 'employment', 'hire', 'recruitment', 'salary',
                           'interview', 'resume', 'career change', 'remote work', 'promotion', 'work'],
                'subtopics': ['remote work', 'job market', 'career advice', 'salary negotiation', 'interview tips']
            }
        }

    def load_interests(self):
        try:
            if os.path.exists(self.interests_file):
                with open(self.interests_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Errore caricamento interessi: {e}")
        return {'conversations': [], 'interest_patterns': {}}

    def save_interests(self):
        try:
            with open(self.interests_file, 'w') as f:
                json.dump(self.interest_database, f, indent=2)
        except Exception as e:
            logger.error(f"Errore salvataggio interessi: {e}")

    def extract_specific_topics(self, title, content):
        """Estrae topic specifici dal titolo e contenuto"""
        text = f"{title} {content}".lower()
        
        # Cerca frasi chiave specifiche
        specific_phrases = []
        
        # Pattern per frasi importanti (domande, affermazioni forti)
        question_patterns = [
            r'how to (.+?)\?', r'what is (.+?)\?', r'why does (.+?)\?',
            r'best way to (.+?)', r'tips for (.+?)', r'guide to (.+?)'
        ]
        
        for pattern in question_patterns:
            matches = re.findall(pattern, text)
            specific_phrases.extend(matches)
        
        # Estrai nomi propri e termini tecnici (parole con maiuscole nel titolo originale)
        proper_nouns = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', title)
        specific_phrases.extend(proper_nouns)
        
        # Estrai hashtag e termini tra virgolette
        quoted_terms = re.findall(r'"([^"]*)"', text)
        specific_phrases.extend(quoted_terms)
        
        # Filtra e pulisci i risultati
        cleaned_phrases = []
        for phrase in specific_phrases:
            phrase = phrase.strip()
            if len(phrase) > 3 and len(phrase) < 50:  # Lunghezza ragionevole
                cleaned_phrases.append(phrase)
        
        return cleaned_phrases[:5]  # Massimo 5 topic specifici

    def categorize_topic_detailed(self, title, content, subreddit):
        """Categorizzazione dettagliata con topic specifici"""
        text_lower = f"{title} {content} {subreddit}".lower()
        category_scores = defaultdict(int)
        specific_topics = self.extract_specific_topics(title, content)
        
        # Punteggio per categoria principale
        for category, data in self.interest_categories.items():
            for keyword in data['keywords']:
                if keyword in text_lower:
                    # Punteggio più alto per match esatti
                    if f" {keyword} " in f" {text_lower} ":
                        category_scores[category] += 3
                    else:
                        category_scores[category] += 1
        
        if not category_scores:
            return 'general', specific_topics, []
        
        best_category = max(category_scores.items(), key=lambda x: x[1])
        best_category_name = best_category[0] if best_category[1] > 0 else 'general'
        
        # Trova subtopic specifici per la categoria
        category_subtopics = []
        if best_category_name in self.interest_categories:
            for subtopic in self.interest_categories[best_category_name]['subtopics']:
                if subtopic.lower() in text_lower:
                    category_subtopics.append(subtopic)
        
        return best_category_name, specific_topics, category_subtopics

    def analyze_conversation_depth(self, post):
        if not hasattr(post, 'num_comments') or post.num_comments < self.analysis_config['min_comments_threshold']:
            return 0
        
        engagement_score = min(post.num_comments / max(1, post.score), 10)
        
        # Bonus per discussioni lunghe
        text_length = len(getattr(post, 'selftext', ''))
        if text_length > 500:
            engagement_score *= 1.5
        elif text_length > 200:
            engagement_score *= 1.2
        
        return engagement_score

    def calculate_interest_momentum(self, category, new_engagement, specific_topics):
        current_trend = self.interest_trends[category]
        new_count = current_trend['count'] * 0.9 + new_engagement
        momentum = new_engagement - (current_trend['count'] * 0.8)
        
        # Aggiorna i topic specifici
        updated_topics = current_trend.get('specific_topics', [])
        for topic in specific_topics:
            found = False
            for existing_topic in updated_topics:
                if existing_topic['topic'] == topic:
                    existing_topic['count'] += 1
                    existing_topic['last_seen'] = datetime.now().isoformat()
                    found = True
                    break
            if not found:
                updated_topics.append({
                    'topic': topic,
                    'count': 1,
                    'first_seen': datetime.now().isoformat(),
                    'last_seen': datetime.now().isoformat()
                })
        
        # Mantieni solo i topic più recenti e frequenti
        updated_topics.sort(key=lambda x: x['count'], reverse=True)
        updated_topics = updated_topics[:10]  # Massimo 10 topic
        
        self.interest_trends[category] = {
            'count': new_count,
            'momentum': momentum,
            'history': current_trend['history'][-29:] + [{'count': new_count, 'timestamp': datetime.now().isoformat()}],
            'specific_topics': updated_topics
        }
        
        return momentum

    def identify_emerging_interests(self):
        emerging = []
        for category, trend in self.interest_trends.items():
            if len(trend['history']) >= 2 and trend['momentum'] > 1.0:  # Soglia più alta
                # Trova il topic specifico più popolare
                top_topic = None
                if trend['specific_topics']:
                    top_topic = max(trend['specific_topics'], key=lambda x: x['count'])
                
                emerging.append({
                    'category': category,
                    'momentum': trend['momentum'],
                    'strength': trend['count'],
                    'specific_topic': top_topic,
                    'all_topics': trend['specific_topics'][:3],  # Top 3 topic
                    'trend': 'emerging'
                })
        
        return sorted(emerging, key=lambda x: x['momentum'], reverse=True)

    def process_discussion(self, post, category, specific_topics, subtopics):
        discussion_text = f"{post.title} {getattr(post, 'selftext', '')}"
        
        if len(discussion_text) < self.analysis_config['min_discussion_length']:
            return None
        
        engagement_score = self.analyze_conversation_depth(post)
        momentum = self.calculate_interest_momentum(category, engagement_score, specific_topics + subtopics)
        
        discussion_data = {
            'id': post.id,
            'title': post.title,
            'category': category,
            'specific_topics': specific_topics,
            'subtopics': subtopics,
            'engagement_score': engagement_score,
            'momentum': momentum,
            'timestamp': datetime.now().isoformat(),
            'comment_count': getattr(post, 'num_comments', 0),
            'upvotes': getattr(post, 'score', 0),
            'subreddit': str(getattr(post, 'subreddit', 'unknown'))
        }
        
        self.conversation_buffer.append(discussion_data)
        self.interest_database['conversations'] = self.interest_database.get('conversations', [])[-199:] + [discussion_data]
        
        return discussion_data

    def get_interest_insights(self):
        emerging_interests = self.identify_emerging_interests()
        
        category_popularity = Counter()
        for conv in self.conversation_buffer:
            category_popularity[conv['category']] += conv['engagement_score']
        
        popular_interests = [{'category': cat, 'score': score} 
                           for cat, score in category_popularity.most_common(5)]
        
        return {
            'emerging_interests': emerging_interests,
            'popular_interests': popular_interests,
            'total_conversations_analyzed': len(self.conversation_buffer),
            'timestamp': datetime.now().isoformat()
        }

# ===== INTELLIGENT INTEREST HUNTER =====
class IntelligentInterestHunter:
    def __init__(self):
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not all([self.reddit_client_id, self.reddit_client_secret]):
            raise ValueError("Credenziali Reddit mancanti!")
        
        self.analyzer = AdvancedInterestAnalyzer()
        self.processed_posts = set()
        
        self.analysis_subreddits = [
            'all', 'popular', 'askreddit', 'technology', 'gaming', 'science',
            'worldnews', 'politics', 'personalfinance', 'relationships',
            'jobs', 'business', 'lifeadvice', 'selfimprovement', 'explainlikeimfive',
            'todayilearned', 'youshouldknow', 'lifehacks'
        ]
        
        self.stats = {
            'total_analyses': 0,
            'total_posts_analyzed': 0,
            'last_alert_sent': None
        }

    async def initialize_reddit(self):
        try:
            self.reddit = asyncpraw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent='IntelligentInterestHunter/3.0'
            )
            logger.info("Reddit connesso per analisi interessi avanzata")
            return True
        except Exception as e:
            logger.error(f"Errore connessione Reddit: {e}")
            return False

    async def deep_analyze_interests(self):
        try:
            current_time = datetime.now()
            analyzed_count = 0
            
            for subreddit_name in self.analysis_subreddits[:10]:
                try:
                    subreddit = await self.reddit.subreddit(subreddit_name)
                    
                    async for post in subreddit.hot(limit=15):
                        if post.id in self.processed_posts:
                            continue
                            
                        post_time = datetime.fromtimestamp(post.created_utc)
                        hours_ago = (current_time - post_time).total_seconds() / 3600
                        
                        if (hours_ago <= 96 and post.num_comments >= 3 and 
                            post.score >= 5 and not post.stickied):
                            
                            # Analisi DETTAGLIATA con topic specifici
                            category, specific_topics, subtopics = self.analyzer.categorize_topic_detailed(
                                post.title, 
                                getattr(post, 'selftext', ''), 
                                str(post.subreddit)
                            )
                            
                            discussion_analysis = self.analyzer.process_discussion(
                                post, category, specific_topics, subtopics
                            )
                            
                            if discussion_analysis:
                                analyzed_count += 1
                                self.processed_posts.add(post.id)
                                
                                # Log dei topic specifici trovati
                                if specific_topics:
                                    logger.info(f"Topic specifici trovati: {specific_topics[:2]}")
                            
                        if analyzed_count >= 25:
                            break
                            
                except Exception as e:
                    logger.warning(f"Errore analisi r/{subreddit_name}: {e}")
                    continue
                
                if analyzed_count >= 25:
                    break
            
            insights = self.analyzer.get_interest_insights()
            self.stats['total_analyses'] += 1
            self.stats['total_posts_analyzed'] += analyzed_count
            
            logger.info(f"Analisi #{self.stats['total_analyses']}: {analyzed_count} post analizzati")
            if insights['emerging_interests']:
                for trend in insights['emerging_interests']:
                    logger.info(f"Trend emergente: {trend['category']} - Topic: {trend.get('specific_topic', {}).get('topic', 'N/A')}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Errore analisi interessi: {e}")
            return None

    def format_alert_message(self, insights, analysis_number):
        if not insights or not insights['emerging_interests']:
            return None
        
        emerging = insights['emerging_interests'][0]
        
        # Costruisci messaggio DETTAGLIATO con topic specifici
        message = "🚨 **TREND EMERGENTE TROVATO!** 🚨\n\n"
        message += f"📈 **{emerging['category'].upper()}** sta esplodendo!\n\n"
        
        # Aggiungi topic specifico se disponibile
        if emerging.get('specific_topic'):
            specific_topic = emerging['specific_topic']
            message += f"🎯 **Topic Specifico:** {specific_topic['topic']}\n"
            message += f"📊 **Menzioni:** {specific_topic['count']} volte\n\n"
        
        # Aggiungi altri topic rilevanti
        if emerging.get('all_topics'):
            message += "🔥 **Altri Topic Correlati:**\n"
            for topic in emerging['all_topics'][:3]:
                message += f"• {topic['topic']} ({topic['count']} mentions)\n"
            message += "\n"
        
        message += f"📈 **Metriche Trend:**\n"
        message += f"• Momentum: +{emerging['momentum']:.1f}\n"
        message += f"• Forza Engagement: {emerging['strength']:.1f}\n"
        message += f"• Post analizzati: {insights['total_conversations_analyzed']}\n\n"
        
        # Aggiungi contesto con classificica interessi
        if insights['popular_interests']:
            message += "🏆 **CLASSIFICA INTERESSI ATUALI:**\n"
            for i, interest in enumerate(insights['popular_interests'][:3], 1):
                message += f"{i}. {interest['category']} ({interest['score']:.1f})\n"
        
        message += f"\n⏰ Analisi #{analysis_number} - {datetime.now().strftime('%H:%M %d/%m')}"
        
        return message

    async def send_telegram_message(self, message, is_alert=False):
        if not self.telegram_token:
            logger.warning("Token Telegram non configurato - skip invio")
            return False
        
        chat_id = self.telegram_chat_id
        if not chat_id:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        if is_alert:
                            self.stats['last_alert_sent'] = datetime.now().strftime('%H:%M %d/%m')
                        logger.info(f"✅ Messaggio Telegram inviato")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Errore Telegram {response.status}: {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Errore invio Telegram: {e}")
            return False

    async def run_interest_analysis(self):
        logger.info("Avvio Intelligent Interest Hunter v3.0")
        
        if not await self.initialize_reddit():
            return
        
        if self.telegram_token:
            await self.send_telegram_message("🤖 **Bot avviato!**\nInizio analisi dettagliata interessi Reddit...")
        
        logger.info("Interest Hunter operativo!")
        
        while True:
            try:
                insights = await self.deep_analyze_interests()
                
                if insights:
                    # INVIO ALERT DETTAGLIATO
                    if insights['emerging_interests']:
                        alert_message = self.format_alert_message(insights, self.stats['total_analyses'])
                        if alert_message and self.telegram_token:
                            success = await self.send_telegram_message(alert_message, is_alert=True)
                            if success:
                                logger.info("🚨 Alert trend emergente INVIAO con topic specifici!")
                    
                    # Salva dati periodicamente
                    if self.stats['total_analyses'] % 5 == 0:
                        self.analyzer.save_interests()
                
                # Pulizia periodica
                if len(self.processed_posts) > 500:
                    self.processed_posts.clear()
                
                await asyncio.sleep(900)  # 15 minuti
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Errore main loop: {e}")
                await asyncio.sleep(300)
        
        self.analyzer.save_interests()
        if hasattr(self, 'reddit'):
            await self.reddit.close()

async def main():
    try:
        hunter = IntelligentInterestHunter()
        await hunter.run_interest_analysis()
    except Exception as e:
        logger.error(f"Errore critico: {e}")

if __name__ == "__main__":
    logger.info("Intelligent Interest Hunter v3.0")
    logger.info("Analisi TOPIC SPECIFICI da discussioni Reddit")
    asyncio.run(main())
