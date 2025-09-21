import asyncpraw
import asyncio
import time
import os
from datetime import datetime, timedelta
import logging
from collections import Counter
import aiohttp
import re
import json
import math

# Configurazione logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('bot.log')]
)
logger = logging.getLogger(__name__)

# ===== AI VIRAL PREDICTOR ENGINE =====
class ViralPredictorAI:
    def __init__(self):
        # Pattern storici di viralità (Machine Learning simulato)
        self.historical_patterns = {
            'elon_musk': {'viral_probability': 0.85, 'peak_hours': 4},
            'ai_breakthrough': {'viral_probability': 0.78, 'peak_hours': 6},
            'crypto_crash': {'viral_probability': 0.92, 'peak_hours': 2},
            'tech_layoffs': {'viral_probability': 0.73, 'peak_hours': 8},
            'scandal_celebrity': {'viral_probability': 0.88, 'peak_hours': 3},
            'market_crash': {'viral_probability': 0.95, 'peak_hours': 1},
            'space_news': {'viral_probability': 0.65, 'peak_hours': 12},
            'gaming_drama': {'viral_probability': 0.70, 'peak_hours': 5},
        }
        
        # Sentiment keywords per analisi emotiva
        self.sentiment_keywords = {
            'high_emotion': ['shocking', 'unbelievable', 'insane', 'crazy', 'amazing', 'incredible', 'breakthrough'],
            'viral_multipliers': ['breaking', 'urgent', 'record', 'highest', 'lowest', 'million', 'billion', 'exclusive', 'leaked']
        }
    
    def analyze_sentiment(self, title):
        """🧠 Analizza sentiment"""
        text = title.lower()
        sentiment_score = 0
        
        # High emotion keywords
        for keyword in self.sentiment_keywords['high_emotion']:
            if keyword in text:
                sentiment_score += 15
        
        # Viral multipliers
        for keyword in self.sentiment_keywords['viral_multipliers']:
            if keyword in text:
                sentiment_score += 20
        
        return min(sentiment_score, 100)
    
    def identify_pattern_category(self, title, subreddit):
        """🎯 Identifica categoria"""
        title_lower = title.lower()
        
        if any(keyword in title_lower for keyword in ['elon', 'musk', 'tesla', 'spacex']):
            return 'elon_musk'
        elif any(keyword in title_lower for keyword in ['ai', 'artificial intelligence', 'chatgpt', 'robot']):
            return 'ai_breakthrough' 
        elif any(keyword in title_lower for keyword in ['bitcoin', 'crypto', 'ethereum', 'crash']):
            return 'crypto_crash'
        elif any(keyword in title_lower for keyword in ['layoffs', 'fired', 'job cuts']):
            return 'tech_layoffs'
        elif any(keyword in title_lower for keyword in ['scandal', 'controversy', 'exposed']):
            return 'scandal_celebrity'
        elif any(keyword in title_lower for keyword in ['market', 'stock', 'crash', 'plummet']):
            return 'market_crash'
        elif any(keyword in title_lower for keyword in ['space', 'mars', 'moon', 'nasa']):
            return 'space_news'
        elif 'gaming' in title_lower or subreddit in ['gaming', 'games']:
            return 'gaming_drama'
        else:
            return 'general'
    
    def predict_viral_trajectory(self, post, subreddit, minutes_ago):
        """🔮 CORE: Predice se diventerà virale"""
        
        # Analisi sentiment
        sentiment_score = self.analyze_sentiment(post.title)
        
        # Pattern matching
        pattern_category = self.identify_pattern_category(post.title, subreddit)
        historical_pattern = self.historical_patterns.get(pattern_category, {
            'viral_probability': 0.5, 'peak_hours': 8
        })
        
        # Velocità virale (upvotes per minuto)
        if minutes_ago > 0:
            velocity = post.score / minutes_ago
            if velocity >= 30:
                velocity_multiplier = 2.0  # Explosive
            elif velocity >= 15:
                velocity_multiplier = 1.5  # Fast
            elif velocity >= 8:
                velocity_multiplier = 1.2  # Steady
            else:
                velocity_multiplier = 1.0  # Slow
        else:
            velocity_multiplier = 1.0
        
        # Engagement ratio (controversia = viralità)
        engagement_ratio = post.num_comments / max(post.score, 1)
        controversy_factor = min(engagement_ratio * 1.5, 2.0)
        
        # ALGORITMO PREDITTIVO
        base_probability = historical_pattern['viral_probability']
        sentiment_multiplier = 1 + (sentiment_score / 100)
        
        # Calcola probabilità finale
        viral_probability = min(
            base_probability * sentiment_multiplier * velocity_multiplier * controversy_factor, 
            0.99
        )
        
        # Predici score finale
        if viral_probability > 0.8:
            predicted_final_score = post.score * 10
        elif viral_probability > 0.6:
            predicted_final_score = post.score * 5
        else:
            predicted_final_score = post.score * 2
        
        return {
            'viral_probability': round(viral_probability * 100, 1),
            'predicted_peak_hours': historical_pattern['peak_hours'],
            'predicted_final_score': int(predicted_final_score),
            'sentiment_score': sentiment_score,
            'pattern_match': pattern_category,
            'velocity_per_min': round(post.score / max(minutes_ago, 1), 1)
        }

# ===== ENHANCED VIRAL NEWS HUNTER =====
class ViralNewsHunter:
    def __init__(self):
        # Credenziali dalle variabili d'ambiente
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        
        if not all([self.reddit_client_id, self.reddit_client_secret, self.telegram_token]):
            raise ValueError("Variabili d'ambiente mancanti!")
        
        # 🧠 AI PREDICTOR
        self.predictor_ai = ViralPredictorAI()
        
        # State management
        self.active_chats = set()
        self.reddit = None
        self.sent_posts = set()
        
        # Subreddit per analisi (stesso di prima)
        self.viral_subreddits = [
            'news', 'worldnews', 'breakingnews', 'nottheonion', 'offbeat',
            'technology', 'gadgets', 'Futurology', 'singularity', 'artificial',
            'MachineLearning', 'cryptocurrency', 'bitcoin', 'ethereum',
            'business', 'economics', 'stocks', 'wallstreetbets', 'investing',
            'todayilearned', 'interestingasfuck', 'nextfuckinglevel', 'Damnthatsinteresting',
            'mildlyinteresting', 'showerthoughts', 'explainlikeimfive',
            'facepalm', 'publicfreakout', 'instant_regret', 'whatcouldgowrong',
            'therewasanattempt', 'crappydesign', 'assholedesign',
            'science', 'space', 'physics', 'biology', 'medicine', 'health',
            'movies', 'television', 'gaming', 'music', 'books',
            'bestof', 'announcements', 'blog'
        ]
        
    async def initialize(self):
        """Inizializza Reddit connection"""
        try:
            self.reddit = asyncpraw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent='ViralNewsHunter/2.0'
            )
            logger.info("✅ Connessione Reddit inizializzata")
            return True
        except Exception as e:
            logger.error(f"❌ Errore inizializzazione Reddit: {e}")
            return False
    
    async def get_active_chats(self):
        """Rileva automaticamente le chat attive"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data['ok'] and data['result']:
                            new_chats = 0
                            for update in data['result']:
                                if 'message' in update:
                                    chat_id = update['message']['chat']['id']
                                    if chat_id not in self.active_chats:
                                        self.active_chats.add(chat_id)
                                        new_chats += 1
                                        logger.info(f"Nuova chat rilevata: {chat_id}")
                            
                            # Marca aggiornamenti come letti
                            if data['result']:
                                last_update_id = data['result'][-1]['update_id']
                                clear_url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates?offset={last_update_id + 1}"
                                await session.get(clear_url)
                            
                            if new_chats > 0:
                                logger.info(f"Rilevate {new_chats} nuove chat. Totale: {len(self.active_chats)}")
                        
                        return True
                    else:
                        logger.error(f"Errore Telegram API: {response.status}")
                        return False
                
        except Exception as e:
            logger.error(f"Errore nel rilevamento chat: {e}")
            return False
    
    async def hunt_viral_news_with_ai(self):
        """🔮 Cerca notizie virali con AI predictions"""
        try:
            viral_posts = []
            predictions = []
            current_time = datetime.now()
            
            for subreddit_name in self.viral_subreddits:
                try:
                    subreddit = await self.reddit.subreddit(subreddit_name)
                    
                    # Analizza post HOT (come prima)
                    count = 0
                    async for post in subreddit.hot(limit=25):
                        count += 1
                        
                        post_time = datetime.fromtimestamp(post.created_utc)
                        minutes_ago = (current_time - post_time).total_seconds() / 60
                        
                        # Focus su post delle ultime 6 ore (come prima)
                        if minutes_ago <= 360:  
                            # 🧠 AGGIUNGI AI PREDICTION
                            ai_prediction = self.predictor_ai.predict_viral_trajectory(
                                post, subreddit_name, minutes_ago
                            )
                            
                            # VIRAL SCORE ORIGINALE (come prima)
                            viral_score = self.calculate_viral_score(post, subreddit_name, minutes_ago)
                            
                            # Combina score originale + AI
                            if viral_score >= 60 and post.id not in self.sent_posts:
                                viral_posts.append({
                                    'id': post.id,
                                    'title': post.title,
                                    'score': post.score,
                                    'subreddit': subreddit_name,
                                    'url': f"https://reddit.com{post.permalink}",
                                    'comments': post.num_comments,
                                    'created': post_time,
                                    'viral_score': viral_score,
                                    'minutes_ago': round(minutes_ago),
                                    'category': self.categorize_viral_post(post.title, subreddit_name),
                                    'upvotes_per_min': round(post.score / max(minutes_ago, 1), 1),
                                    # 🔮 AGGIUNGI AI DATA
                                    'ai_prediction': ai_prediction
                                })
                        
                        if count >= 25:
                            break
                            
                except Exception as e:
                    logger.warning(f"Errore nel subreddit {subreddit_name}: {e}")
                    continue
            
            # Ordina per viral_score (come prima)
            viral_posts.sort(key=lambda x: x['viral_score'], reverse=True)
            
            logger.info(f"Trovati {len(viral_posts)} post virali con AI")
            
            return {
                'viral_posts': viral_posts[:8],  # Top 8 come prima
                'timestamp': current_time
            }
            
        except Exception as e:
            logger.error(f"Errore nella caccia virale: {e}")
            return None
    
    def calculate_viral_score(self, post, subreddit, minutes_ago):
        """Calcola viral score ORIGINALE (come prima)"""
        score = 0
        title_lower = post.title.lower()
        
        # Base score da upvotes e velocità
        if minutes_ago > 0:
            upvotes_per_minute = post.score / minutes_ago
            score += min(upvotes_per_minute * 2, 100)
        
        # Bonus per numero assoluto di upvotes
        if post.score > 1000:
            score += 50
        elif post.score > 500:
            score += 30
        elif post.score > 100:
            score += 15
        
        # Bonus per commenti
        if post.num_comments > 500:
            score += 40
        elif post.num_comments > 200:
            score += 25
        elif post.num_comments > 50:
            score += 10
        
        # Viral keywords
        viral_indicators = [
            'breaking', 'urgent', 'developing', 'record', 'highest', 'lowest',
            'shocking', 'unbelievable', 'viral', 'trending', 'million', 'billion',
            'elon musk', 'ai', 'chatgpt', 'tesla', 'unprecedented', 'historic'
        ]
        
        for keyword in viral_indicators:
            if keyword in title_lower:
                score += 25
        
        # Penalità per post troppo vecchi
        if minutes_ago > 180:
            score *= 0.5
        
        return int(score)
    
    def categorize_viral_post(self, title, subreddit):
        """Categorizza post (come prima)"""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['elon', 'tesla']):
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
    
    def format_viral_message_with_ai(self, data):
        """📱 Formatta messaggio CON AI predictions"""
        if not data or not data['viral_posts']:
            return "❌ Nessuna notizia virale rilevata in questo momento."
        
        timestamp = data['timestamp'].strftime("%H:%M - %d/%m/%Y")
        
        message = f"🔥 NOTIZIE VIRALI DELL'ULTIMA ORA 🔥\n"
        message += f"⏰ Scansione: {timestamp}\n"
        message += f"🧠 Powered by AI Predictions\n\n"
        
        # Top notizie virali
        message += "📈 TOP NOTIZIE CHE STANNO DIVENTANDO VIRALI:\n"
        
        for i, post in enumerate(data['viral_posts'], 1):
            title = post['title'][:70] + "..." if len(post['title']) > 70 else post['title']
            title = title.replace('[', '').replace(']', '').replace('*', '')
            
            # 🔮 AGGIUNGI AI INFO
            ai = post['ai_prediction']
            
            # Emoji AI confidence
            if ai['viral_probability'] >= 80:
                ai_emoji = "🚀🔥"
            elif ai['viral_probability'] >= 60:
                ai_emoji = "⚡📈"
            else:
                ai_emoji = "📊"
            
            message += f"\n{post['category']} {i}. {title}\n"
            message += f"🔥 Viral Score: {post['viral_score']} | "
            message += f"👍 {post['score']} ({post['upvotes_per_min']}/min) | "
            message += f"💬 {post['comments']}\n"
            
            # 🧠 AGGIUNGI AI PREDICTIONS
            message += f"{ai_emoji} AI Predice: {ai['viral_probability']}% prob → "
            message += f"{ai['predicted_final_score']:,} upvotes in {ai['predicted_peak_hours']}h\n"
            
            message += f"📍 r/{post['subreddit']} | ⏱️ {post['minutes_ago']} min fa\n"
            message += f"🔗 {post['url']}\n"
        
        # Statistiche finali
        total_viral = len(data['viral_posts'])
        avg_ai_prob = sum(p['ai_prediction']['viral_probability'] for p in data['viral_posts']) / total_viral
        
        message += f"\n📊 {total_viral} notizie virali trovate\n"
        message += f"🧠 AI Confidence media: {avg_ai_prob:.1f}%\n"
        message += f"🎯 Scansionati {len(self.viral_subreddits)} subreddit"
        
        return message
    
    async def send_to_telegram(self, message):
        """Invia messaggio a tutte le chat attive"""
        if not self.active_chats:
            logger.warning("Nessuna chat attiva trovata")
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
                            logger.info(f"Messaggio inviato alla chat {chat_id}")
                            success_count += 1
                        else:
                            error_data = await response.text()
                            logger.error(f"Errore nell'invio alla chat {chat_id}: {response.status}")
                            if response.status in [400, 403, 404]:
                                self.active_chats.discard(chat_id)
                                logger.info(f"Chat {chat_id} rimossa dalle chat attive")
                                
                except Exception as e:
                    logger.error(f"Errore nell'invio alla chat {chat_id}: {e}")
        
        return success_count > 0
    
    async def run_viral_hunter(self):
        """🚀 MAIN LOOP (stessa struttura di prima ma con AI)"""
        logger.info("🚀 Avvio Viral News Hunter con AI...")
        
        if not await self.initialize():
            logger.error("❌ Impossibile inizializzare Reddit!")
            return
        
        logger.info("✅ Viral News Hunter con AI avviato!")
        logger.info("🧠 AI Predictions integrate nel flusso esistente")
        logger.info("⏰ Scansione ogni 15 minuti (come prima)")
        
        while True:
            try:
                # Rileva nuove chat
                await self.get_active_chats()
                
                # Cerca viral news CON AI
                logger.info("🔍 Iniziando caccia notizie virali con AI...")
                viral_data = await self.hunt_viral_news_with_ai()
                
                if viral_data and viral_data['viral_posts']:
                    # Filtra solo i post non ancora inviati
                    new_viral = [p for p in viral_data['viral_posts'] if p['id'] not in self.sent_posts]
                    
                    if new_viral and self.active_chats:
                        # Aggiorna la lista dei post inviati
                        for post in new_viral:
                            self.sent_posts.add(post['id'])
                        
                        # Aggiorna i dati con solo i nuovi post
                        viral_data['viral_posts'] = new_viral
                        
                        message = self.format_viral_message_with_ai(viral_data)
                        success = await self.send_to_telegram(message)
                        
                        if success:
                            logger.info(f"🔥 Inviate {len(new_viral)} notizie virali con AI!")
                            
                            # Log AI predictions
                            for post in new_viral:
                                ai = post['ai_prediction']
                                logger.info(f"  🧠 AI: {ai['viral_probability']}% prob → {ai['predicted_final_score']} score: {post['title'][:40]}...")
                        else:
                            logger.warning("⚠️ Errore nell'invio dei messaggi")
                    elif not self.active_chats:
                        logger.info("⏳ Nessuna chat attiva. Invia un messaggio al bot per iniziare.")
                    else:
                        logger.info("⚠️ Nessuna nuova notizia virale (già inviate)")
                else:
                    logger.info("⚠️ Nessuna notizia virale rilevata")
                
                # Pulisci la cache ogni tanto per evitare memory leak
                if len(self.sent_posts) > 1000:
                    self.sent_posts.clear()
                    logger.info("🧹 Cache post inviati pulita")
                
                # 🕐 ATTENDI 15 MINUTI (come prima)
                logger.info("⏱️ Prossima caccia virale tra 15 minuti...")
                await asyncio.sleep(900)  # 15 minuti
                
            except KeyboardInterrupt:
                logger.info("🛑 Bot fermato dall'utente")
                break
            except Exception as e:
                logger.error(f"Errore nel ciclo principale: {e}")
                logger.info("🔄 Riprovando tra 3 minuti...")
                await asyncio.sleep(180)  # Attendi 3 minuti prima di riprovare
        
        # Chiudi la connessione Reddit
        if self.reddit:
            await self.reddit.close()
            logger.info("🔌 Connessione Reddit chiusa")

async def main():
    """Funzione principale"""
    try:
        bot = ViralNewsHunter()
        await bot.run_viral_hunter()
    except Exception as e:
        logger.error(f"Errore critico: {e}")
        # Attendi prima di uscire per permettere restart automatico
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
