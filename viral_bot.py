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
        self.interest_trends = defaultdict(lambda: {'count': 0, 'momentum': 0, 'history': []})
        
        self.analysis_config = {
            'min_discussion_length': 30,
            'min_comments_threshold': 3,
            'engagement_weight': 2.0,
            'emerging_threshold': 0.1
        }
        
        self.interest_categories = {
            'technology': ['ai', 'programming', 'software', 'tech', 'coding', 'developer', 'computer'],
            'gaming': ['game', 'gaming', 'playstation', 'xbox', 'nintendo', 'steam'],
            'entertainment': ['movie', 'film', 'tv', 'netflix', 'music', 'spotify'],
            'lifestyle': ['fitness', 'travel', 'food', 'cooking', 'fashion', 'home'],
            'science': ['science', 'space', 'research', 'discovery', 'physics'],
            'business': ['business', 'startup', 'entrepreneur', 'marketing', 'investment'],
            'politica': ['politics', 'government', 'election', 'policy', 'law'],
            'social': ['social', 'community', 'relationship', 'friendship', 'media'],
            'life': ['life', 'personal', 'mental health', 'advice', 'support'],
            'mercato lavoro': ['job', 'career', 'work', 'employment', 'salary']
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

    def analyze_conversation_depth(self, post):
        if not hasattr(post, 'num_comments') or post.num_comments < self.analysis_config['min_comments_threshold']:
            return 0
        
        engagement_score = min(post.num_comments / max(1, post.score), 10)
        return engagement_score

    def categorize_topic(self, title, subreddit):
        text_lower = f"{title} {subreddit}".lower()
        category_scores = defaultdict(int)
        
        for category, keywords in self.interest_categories.items():
            for keyword in keywords:
                if keyword in text_lower:
                    category_scores[category] += 3
        
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            return best_category[0] if best_category[1] > 0 else 'general'
        
        return 'general'

    def calculate_interest_momentum(self, category, new_engagement):
        current_trend = self.interest_trends[category]
        new_count = current_trend['count'] * 0.9 + new_engagement
        momentum = new_engagement - (current_trend['count'] * 0.8)
        
        self.interest_trends[category] = {
            'count': new_count,
            'momentum': momentum,
            'history': current_trend['history'][-29:] + [{'count': new_count, 'timestamp': datetime.now().isoformat()}]
        }
        
        return momentum

    def identify_emerging_interests(self):
        emerging = []
        for category, trend in self.interest_trends.items():
            if len(trend['history']) >= 2 and trend['momentum'] > 0:
                emerging.append({
                    'category': category,
                    'momentum': trend['momentum'],
                    'strength': trend['count'],
                    'trend': 'emerging'
                })
        
        return sorted(emerging, key=lambda x: x['momentum'], reverse=True)

    def process_discussion(self, post, category):
        discussion_text = f"{post.title} {getattr(post, 'selftext', '')}"
        
        if len(discussion_text) < self.analysis_config['min_discussion_length']:
            return None
        
        engagement_score = self.analyze_conversation_depth(post)
        momentum = self.calculate_interest_momentum(category, engagement_score)
        
        discussion_data = {
            'id': post.id,
            'title': post.title,
            'category': category,
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
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')  # Chat ID specifico
        
        if not all([self.reddit_client_id, self.reddit_client_secret]):
            raise ValueError("Credenziali Reddit mancanti!")
        
        self.analyzer = AdvancedInterestAnalyzer()
        self.processed_posts = set()
        
        # Subreddit per analisi
        self.analysis_subreddits = [
            'all', 'popular', 'askreddit', 'technology', 'gaming', 'science',
            'worldnews', 'politics', 'personalfinance', 'relationships',
            'jobs', 'business', 'lifeadvice', 'selfimprovement'
        ]
        
        # Statistiche per debug
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
                user_agent='IntelligentInterestHunter/2.0'
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
            
            for subreddit_name in self.analysis_subreddits[:8]:
                try:
                    subreddit = await self.reddit.subreddit(subreddit_name)
                    
                    async for post in subreddit.hot(limit=10):
                        if post.id in self.processed_posts:
                            continue
                            
                        post_time = datetime.fromtimestamp(post.created_utc)
                        hours_ago = (current_time - post_time).total_seconds() / 3600
                        
                        if (hours_ago <= 96 and post.num_comments >= 3 and 
                            post.score >= 5 and not post.stickied):
                            
                            category = self.analyzer.categorize_topic(post.title, str(post.subreddit))
                            discussion_analysis = self.analyzer.process_discussion(post, category)
                            
                            if discussion_analysis:
                                analyzed_count += 1
                                self.processed_posts.add(post.id)
                            
                        if analyzed_count >= 20:
                            break
                            
                except Exception as e:
                    logger.warning(f"Errore analisi r/{subreddit_name}: {e}")
                    continue
                
                if analyzed_count >= 20:
                    break
            
            insights = self.analyzer.get_interest_insights()
            self.stats['total_analyses'] += 1
            self.stats['total_posts_analyzed'] += analyzed_count
            
            logger.info(f"Analisi #{self.stats['total_analyses']}: {analyzed_count} post analizzati")
            
            return insights
            
        except Exception as e:
            logger.error(f"Errore analisi interessi: {e}")
            return None

    def format_alert_message(self, insights, analysis_number):
        if not insights or not insights['emerging_interests']:
            return None
        
        emerging = insights['emerging_interests'][0]  # Prendi il trend più forte
        
        message = "🚨 **TREND EMERGENTE TROVATO!** 🚨\n\n"
        message += f"📈 **{emerging['category'].upper()}** sta esplodendo!\n"
        message += f"• Momentum: +{emerging['momentum']:.1f}\n"
        message += f"• Forza: {emerging['strength']:.1f}\n"
        message += f"• Post analizzati: {insights['total_conversations_analyzed']}\n\n"
        
        # Aggiungi contesto
        if insights['popular_interests']:
            message += "🔥 **CLASSIFICA INTERESSI:**\n"
            for i, interest in enumerate(insights['popular_interests'][:3], 1):
                message += f"{i}. {interest['category']} ({interest['score']:.1f})\n"
        
        message += f"\n⏰ Analisi #{analysis_number} - {datetime.now().strftime('%H:%M %d/%m')}"
        
        return message

    def format_debug_message(self, insights, analysis_number):
        message = "🔧 **DEBUG BOT STATUS** 🔧\n\n"
        message += f"📊 **Statistiche Bot:**\n"
        message += f"• Analisi totali: {self.stats['total_analyses']}\n"
        message += f"• Post analizzati: {self.stats['total_posts_analyzed']}\n"
        message += f"• Ultimo alert: {self.stats['last_alert_sent'] or 'Mai'}\n\n"
        
        if insights:
            message += f"📈 **Ultima Analisi (#{analysis_number}):**\n"
            message += f"• Discussioni analizzate: {insights['total_conversations_analyzed']}\n"
            message += f"• Trend emergenti: {len(insights['emerging_interests'])}\n"
            message += f"• Interessi popolari: {len(insights['popular_interests'])}\n"
            
            if insights['emerging_interests']:
                message += f"\n🚀 **Top Trend Emergente:**\n"
                message += f"• {insights['emerging_interests'][0]['category']} (+{insights['emerging_interests'][0]['momentum']:.1f})\n"
        
        message += f"\n🔄 Prossima analisi tra 15 minuti"
        
        return message

    async def send_telegram_message(self, message, is_alert=False):
        """Invia messaggio a Telegram"""
        if not self.telegram_token:
            logger.warning("Token Telegram non configurato - skip invio")
            return False
        
        # Usa chat ID specifico o trova automaticamente
        chat_id = self.telegram_chat_id
        
        if not chat_id:
            logger.warning("Nessun chat ID configurato - skip invio")
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
                        logger.info(f"✅ Messaggio Telegram inviato a {chat_id}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Errore Telegram {response.status}: {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Errore invio Telegram: {e}")
            return False

    async def find_telegram_chat_id(self):
        """Trova automaticamente l'ID della chat"""
        if not self.telegram_token:
            return None
            
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data['ok'] and data['result']:
                            # Prendi l'ultimo messaggio
                            last_update = data['result'][-1]
                            if 'message' in last_update:
                                chat_id = last_update['message']['chat']['id']
                                logger.info(f"Trovato chat ID: {chat_id}")
                                return chat_id
            return None
        except Exception as e:
            logger.error(f"Errore ricerca chat ID: {e}")
            return None

    async def run_interest_analysis(self):
        """Esegue l'analisi continua degli interessi"""
        logger.info("Avvio Intelligent Interest Hunter")
        
        if not await self.initialize_reddit():
            return
        
        # Configura Telegram
        if self.telegram_token and not self.telegram_chat_id:
            self.telegram_chat_id = await self.find_telegram_chat_id()
            if self.telegram_chat_id:
                await self.send_telegram_message("🤖 **Bot avviato correttamente!**\nInizio analisi interessi Reddit...")
        
        logger.info("Interest Hunter operativo!")
        
        while True:
            try:
                insights = await self.deep_analyze_interests()
                
                if insights:
                    # INVIO ALERT SE TROVATO TREND EMERGENTE
                    if insights['emerging_interests']:
                        alert_message = self.format_alert_message(insights, self.stats['total_analyses'])
                        if alert_message and self.telegram_token:
                            await self.send_telegram_message(alert_message, is_alert=True)
                            logger.info("🚨 Alert trend emergente inviato!")
                    
                    # INVIO REPORT DEBUG OGNI 4 ANALISI
                    if self.stats['total_analyses'] % 4 == 0 and self.telegram_token:
                        debug_message = self.format_debug_message(insights, self.stats['total_analyses'])
                        await self.send_telegram_message(debug_message)
                        logger.info("📊 Report debug inviato")
                    
                    # Salva dati ogni 10 analisi
                    if self.stats['total_analyses'] % 10 == 0:
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
    logger.info("Sistema di alert trend emergenti Reddit")
    asyncio.run(main())
