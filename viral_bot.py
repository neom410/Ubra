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

# ===== GRADIENT LEARNING AI ENGINE =====
class GradientLearningAI:
    def __init__(self):
        # File per salvare i pesi appresi
        self.weights_file = 'ai_weights.json'
        self.predictions_file = 'predictions_track.json'
        
        # Pesi iniziali (verranno aggiustati automaticamente)
        self.default_weights = {
            # Pattern specifici
            'elon_musk': 2.5,
            'ai_breakthrough': 1.8,
            'crypto_crash': 2.8,
            'tech_layoffs': 1.6,
            'scandal_celebrity': 2.2,
            'market_crash': 3.0,
            'space_news': 1.4,
            'gaming_drama': 1.7,
            'political_news': 2.1,
            'health_news': 1.5,
            'climate_news': 1.3,
            'sports_news': 1.4,
            'entertainment': 1.6,
            'science_discovery': 1.7,
            'tech_general': 1.5,
            'business_news': 1.4,
            'general': 1.0,
            
            # Sentiment multipliers
            'high_emotion_weight': 1.5,
            'urgency_weight': 2.0,
            'controversy_weight': 1.8,
            'numbers_weight': 1.3,
            'exclusivity_weight': 1.6,
            
            # Velocity thresholds
            'velocity_explosive_threshold': 50.0,
            'velocity_fast_threshold': 20.0,
            'velocity_steady_threshold': 10.0,
            
            # Engagement weights
            'engagement_weight': 1.2,
            'comment_ratio_weight': 1.4,
            'time_decay_weight': 0.8,
            
            # Platform-specific
            'reddit_hot_multiplier': 1.0,
            'reddit_rising_multiplier': 1.3,
            'subreddit_size_weight': 1.1,
            
            # Time-based learning
            'learning_rate': 0.05,
            'success_boost': 1.03,
            'failure_reduction': 0.97
        }
        
        # Carica pesi salvati o usa default
        self.weights = self.load_weights()
        
        # Tracking predizioni per feedback
        self.active_predictions = self.load_predictions()
        
        # Pattern di riconoscimento avanzato
        self.pattern_keywords = {
            'elon_musk': ['elon', 'musk', 'tesla', 'spacex', 'neuralink', 'boring company'],
            'ai_breakthrough': ['ai', 'artificial intelligence', 'chatgpt', 'gpt', 'openai', 'claude', 'robot', 'automation', 'machine learning', 'neural network'],
            'crypto_crash': ['bitcoin', 'crypto', 'ethereum', 'blockchain', 'defi', 'nft', 'crash', 'pump', 'dump', 'bull', 'bear'],
            'tech_layoffs': ['layoffs', 'fired', 'job cuts', 'downsizing', 'restructuring', 'redundant'],
            'scandal_celebrity': ['scandal', 'controversy', 'exposed', 'caught', 'arrest', 'lawsuit', 'divorce', 'affair'],
            'market_crash': ['market', 'stock', 'dow', 'nasdaq', 'sp500', 'crash', 'plummet', 'bear market', 'recession'],
            'space_news': ['space', 'mars', 'moon', 'rocket', 'nasa', 'spacex', 'iss', 'satellite', 'astronaut'],
            'gaming_drama': ['gaming', 'game', 'streamer', 'twitch', 'youtube', 'esports', 'nintendo', 'sony', 'xbox'],
            'political_news': ['trump', 'biden', 'election', 'congress', 'senate', 'president', 'politics', 'vote', 'policy'],
            'health_news': ['covid', 'vaccine', 'pandemic', 'health', 'medical', 'doctor', 'hospital', 'disease', 'cure'],
            'climate_news': ['climate', 'global warming', 'carbon', 'emission', 'green', 'renewable', 'pollution'],
            'sports_news': ['football', 'basketball', 'soccer', 'olympics', 'championship', 'world cup', 'nfl', 'nba'],
            'entertainment': ['movie', 'netflix', 'disney', 'actor', 'actress', 'film', 'tv show', 'series', 'music'],
            'science_discovery': ['study', 'research', 'scientists', 'discovery', 'breakthrough', 'experiment', 'published'],
            'tech_general': ['technology', 'tech', 'startup', 'innovation', 'app', 'software', 'hardware', 'gadget'],
            'business_news': ['business', 'company', 'ceo', 'merger', 'acquisition', 'ipo', 'earnings', 'revenue']
        }
        
        # Sentiment keywords avanzate
        self.sentiment_keywords = {
            'high_emotion': ['shocking', 'unbelievable', 'insane', 'crazy', 'amazing', 'incredible', 'breakthrough', 'revolutionary', 'game-changing', 'mind-blowing', 'devastating', 'horrific', 'tragic', 'miraculous'],
            'urgency': ['breaking', 'urgent', 'just in', 'developing', 'live', 'now', 'alert', 'immediate', 'emergency'],
            'controversy': ['banned', 'censored', 'forbidden', 'illegal', 'controversial', 'outrageous', 'scandal', 'exposed', 'leaked'],
            'numbers': ['million', 'billion', 'trillion', '%', '$', 'record', 'highest', 'lowest', 'first', 'largest', 'biggest'],
            'exclusivity': ['exclusive', 'only', 'never before', 'unprecedented', 'rare', 'secret', 'hidden', 'revealed', 'insider']
        }
    
    def load_weights(self):
        """Carica pesi salvati o usa default"""
        try:
            if os.path.exists(self.weights_file):
                with open(self.weights_file, 'r') as f:
                    saved_weights = json.load(f)
                    # Merge con default per nuove chiavi
                    weights = self.default_weights.copy()
                    weights.update(saved_weights)
                    logger.info(f"📚 Caricati pesi AI da {self.weights_file}")
                    return weights
        except Exception as e:
            logger.warning(f"Errore caricamento pesi: {e}")
        
        logger.info("🆕 Inizializzo pesi AI default")
        return self.default_weights.copy()
    
    def save_weights(self):
        """Salva pesi aggiornati"""
        try:
            with open(self.weights_file, 'w') as f:
                json.dump(self.weights, f, indent=2)
            logger.debug("💾 Pesi AI salvati")
        except Exception as e:
            logger.error(f"Errore salvataggio pesi: {e}")
    
    def load_predictions(self):
        """Carica predizioni per tracking"""
        try:
            if os.path.exists(self.predictions_file):
                with open(self.predictions_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Errore caricamento predizioni: {e}")
        return {}
    
    def save_predictions(self):
        """Salva predizioni attive"""
        try:
            with open(self.predictions_file, 'w') as f:
                json.dump(self.active_predictions, f, indent=2)
        except Exception as e:
            logger.error(f"Errore salvataggio predizioni: {e}")
    
    def identify_pattern_category(self, title, subreddit):
        """🎯 Identifica categoria con pattern avanzato"""
        title_lower = title.lower()
        subreddit_lower = subreddit.lower()
        
        # Controlla ogni pattern
        for pattern, keywords in self.pattern_keywords.items():
            for keyword in keywords:
                if keyword in title_lower or keyword in subreddit_lower:
                    return pattern
        
        return 'general'
    
    def analyze_sentiment(self, title):
        """😍 Analisi sentiment avanzata"""
        text = title.lower()
        sentiment_data = {
            'high_emotion': 0,
            'urgency': 0,
            'controversy': 0,
            'numbers': 0,
            'exclusivity': 0
        }
        
        # Conta keywords per categoria
        for category, keywords in self.sentiment_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text)
            sentiment_data[category] = count
        
        # Calcola score totale
        total_score = (
            sentiment_data['high_emotion'] * self.weights['high_emotion_weight'] +
            sentiment_data['urgency'] * self.weights['urgency_weight'] +
            sentiment_data['controversy'] * self.weights['controversy_weight'] +
            sentiment_data['numbers'] * self.weights['numbers_weight'] +
            sentiment_data['exclusivity'] * self.weights['exclusivity_weight']
        )
        
        return {
            'total_score': min(total_score, 100),
            'categories': sentiment_data,
            'intensity': 'high' if total_score > 40 else 'medium' if total_score > 15 else 'low'
        }
    
    def calculate_velocity_score(self, post, minutes_ago):
        """⚡ Calcola score velocità adattivo"""
        if minutes_ago <= 0:
            return 0
        
        velocity = post.score / minutes_ago
        
        # Usa soglie adattive
        explosive_threshold = self.weights['velocity_explosive_threshold']
        fast_threshold = self.weights['velocity_fast_threshold']
        steady_threshold = self.weights['velocity_steady_threshold']
        
        if velocity >= explosive_threshold:
            return 100  # Explosive
        elif velocity >= fast_threshold:
            return 70   # Fast growth
        elif velocity >= steady_threshold:
            return 40   # Steady
        else:
            return velocity * 2  # Proportional for slow
    
    def predict_viral_trajectory(self, post, subreddit, minutes_ago):
        """🧠 CORE: Predizione con Gradient Learning"""
        
        # 1. Identifica pattern
        pattern_category = self.identify_pattern_category(post.title, subreddit)
        pattern_multiplier = self.weights.get(pattern_category, 1.0)
        
        # 2. Analisi sentiment
        sentiment = self.analyze_sentiment(post.title)
        sentiment_multiplier = 1 + (sentiment['total_score'] / 100)
        
        # 3. Velocità adattiva
        velocity_score = self.calculate_velocity_score(post, minutes_ago)
        velocity_multiplier = 1 + (velocity_score / 100)
        
        # 4. Engagement analysis
        if post.score > 0:
            engagement_ratio = post.num_comments / post.score
            engagement_multiplier = 1 + (engagement_ratio * self.weights['engagement_weight'])
        else:
            engagement_multiplier = 1.0
        
        # 5. Time decay
        if minutes_ago > 180:  # >3 ore
            time_multiplier = self.weights['time_decay_weight']
        else:
            time_multiplier = 1.0
        
        # 6. CALCOLO PROBABILITÀ con pesi appresi
        base_probability = 0.3  # 30% base
        
        final_probability = (
            base_probability * 
            pattern_multiplier * 
            sentiment_multiplier * 
            velocity_multiplier * 
            engagement_multiplier * 
            time_multiplier
        )
        
        # Clamp tra 0.01 e 0.99
        final_probability = max(0.01, min(final_probability, 0.99))
        
        # Predici score finale
        growth_factor = final_probability * 20  # Max 20x growth
        predicted_final_score = int(post.score * (1 + growth_factor))
        
        # Predici peak time (basato su pattern)
        peak_hours_base = 6  # Default 6 ore
        if pattern_category in ['market_crash', 'crypto_crash']:
            peak_hours = 2
        elif pattern_category in ['elon_musk', 'scandal_celebrity']:
            peak_hours = 4
        elif pattern_category in ['space_news', 'science_discovery']:
            peak_hours = 12
        else:
            peak_hours = peak_hours_base
        
        return {
            'viral_probability': round(final_probability * 100, 1),
            'confidence': sentiment['intensity'],
            'predicted_peak_hours': peak_hours,
            'predicted_final_score': predicted_final_score,
            'pattern_match': pattern_category,
            'sentiment_analysis': sentiment,
            'velocity_score': velocity_score,
            'pattern_multiplier': round(pattern_multiplier, 2),
            'reasoning': self.generate_reasoning(pattern_category, sentiment, velocity_score, final_probability * 100)
        }
    
    def generate_reasoning(self, pattern, sentiment, velocity, probability):
        """🤔 Genera spiegazione AI"""
        reasons = []
        
        # Pattern reasoning
        pattern_explanations = {
            'elon_musk': "Pattern Elon Musk - viralità quasi garantita",
            'ai_breakthrough': "AI news - tech community molto interessata",
            'crypto_crash': "Crypto volatility - spread virale rapido",
            'scandal_celebrity': "Celebrity scandal - engagement esplosivo",
            'market_crash': "Market news - panic spreading veloce",
            'political_news': "Politics - sempre divisivo e virale",
            'health_news': "Health topic - interesse pubblico alto",
            'gaming_drama': "Gaming community - passionate engagement"
        }
        
        if pattern in pattern_explanations:
            reasons.append(pattern_explanations[pattern])
        
        # Sentiment reasoning
        if sentiment['intensity'] == 'high':
            reasons.append(f"Alto carico emotivo (score: {sentiment['total_score']})")
        
        # Velocity reasoning
        if velocity >= 70:
            reasons.append("Crescita explosive in corso")
        elif velocity >= 40:
            reasons.append("Velocità sostenuta")
        
        # Probability reasoning
        if probability > 75:
            reasons.append("Tutti indicatori convergono su viral explosion")
        elif probability > 50:
            reasons.append("Pattern simili storicamente virali")
        
        return " • ".join(reasons[:3])
    
    def track_prediction(self, post_id, prediction_data, post_score):
        """📊 Traccia predizione per future learning"""
        self.active_predictions[post_id] = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction_data,
            'original_score': post_score,
            'pattern': prediction_data['pattern_match'],
            'predicted_probability': prediction_data['viral_probability'],
            'predicted_final_score': prediction_data['predicted_final_score']
        }
        self.save_predictions()
    
    async def check_and_learn(self, reddit):
        """🧠 Controlla predizioni passate e impara"""
        if not self.active_predictions:
            return
        
        current_time = datetime.now()
        learned_count = 0
        
        for post_id, prediction_data in list(self.active_predictions.items()):
            try:
                # Controlla predizioni di almeno 6 ore fa
                prediction_time = datetime.fromisoformat(prediction_data['timestamp'])
                hours_passed = (current_time - prediction_time).total_seconds() / 3600
                
                if hours_passed >= 6:
                    # Qui dovremmo ricontrollare il post su Reddit
                    # Per ora usiamo logica semplificata basata sui dati
                    
                    original_score = prediction_data['original_score']
                    predicted_final = prediction_data['predicted_final_score']
                    predicted_prob = prediction_data['predicted_probability']
                    pattern = prediction_data['pattern']
                    
                    # Stima se è diventato virale (logica semplificata)
                    estimated_viral = self.estimate_if_went_viral(
                        original_score, predicted_prob, hours_passed
                    )
                    
                    # GRADIENT LEARNING: Aggiusta pesi
                    self.apply_gradient_learning(
                        pattern, predicted_prob, estimated_viral
                    )
                    
                    # Rimuovi da tracking
                    del self.active_predictions[post_id]
                    learned_count += 1
                    
            except Exception as e:
                logger.warning(f"Errore learning per {post_id}: {e}")
                # Rimuovi predizioni problematiche
                if post_id in self.active_predictions:
                    del self.active_predictions[post_id]
        
        if learned_count > 0:
            logger.info(f"🧠 Gradient Learning: Aggiornati pesi da {learned_count} predizioni")
            self.save_weights()
            self.save_predictions()
    
    def estimate_if_went_viral(self, original_score, predicted_prob, hours_passed):
        """📈 Stima se è diventato virale (logica semplificata)"""
        # Logica semplificata: alta probabilità + tempo = probabile successo
        if predicted_prob > 70 and hours_passed > 8:
            return True
        elif predicted_prob > 50 and original_score > 200:
            return True
        elif predicted_prob > 30 and original_score > 500:
            return True
        else:
            return False
    
    def apply_gradient_learning(self, pattern, predicted_prob, actual_viral):
        """🎯 Applica gradient learning ai pesi"""
        learning_rate = self.weights['learning_rate']
        success_boost = self.weights['success_boost']
        failure_reduction = self.weights['failure_reduction']
        
        was_correct = (predicted_prob > 50 and actual_viral) or (predicted_prob <= 50 and not actual_viral)
        
        if was_correct:
            # Predizione corretta - rinforza pattern
            if pattern in self.weights:
                self.weights[pattern] *= success_boost
                logger.debug(f"✅ Rinforzato pattern {pattern}: {self.weights[pattern]:.3f}")
        else:
            # Predizione sbagliata - riduci peso pattern
            if pattern in self.weights:
                self.weights[pattern] *= failure_reduction
                logger.debug(f"❌ Ridotto pattern {pattern}: {self.weights[pattern]:.3f}")
            
            # Aggiusta anche pesi sentiment se molto sbagliato
            if abs(predicted_prob - (100 if actual_viral else 0)) > 40:
                if predicted_prob > 80 and not actual_viral:
                    # False positive - riduci pesi sentiment
                    self.weights['high_emotion_weight'] *= failure_reduction
                    self.weights['urgency_weight'] *= failure_reduction
                elif predicted_prob < 30 and actual_viral:
                    # False negative - aumenta pesi sentiment
                    self.weights['high_emotion_weight'] *= success_boost
                    self.weights['velocity_explosive_threshold'] *= 0.95  # Soglia più bassa

# ===== ENHANCED VIRAL NEWS HUNTER =====
class ViralNewsHunter:
    def __init__(self):
        # Credenziali
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        
        if not all([self.reddit_client_id, self.reddit_client_secret, self.telegram_token]):
            raise ValueError("Variabili d'ambiente mancanti!")
        
        # 🧠 GRADIENT LEARNING AI
        self.gradient_ai = GradientLearningAI()
        
        # State management
        self.active_chats = set()
        self.reddit = None
        self.sent_posts = set()
        
        # Subreddit per analisi
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
            'bestof', 'announcements', 'blog', 'politics', 'worldpolitics'
        ]
        
    async def initialize(self):
        """Inizializza Reddit connection"""
        try:
            self.reddit = asyncpraw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent='ViralNewsHunter-GradientAI/2.0'
            )
            logger.info("✅ Reddit connesso con Gradient AI")
            return True
        except Exception as e:
            logger.error(f"❌ Errore Reddit: {e}")
            return False
    
    async def get_active_chats(self):
        """Rileva chat attive"""
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
                                        logger.info(f"Nuova chat: {chat_id}")
                            
                            if data['result']:
                                last_update_id = data['result'][-1]['update_id']
                                clear_url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates?offset={last_update_id + 1}"
                                await session.get(clear_url)
                            
                            if new_chats > 0:
                                logger.info(f"📱 {new_chats} nuove chat. Totale: {len(self.active_chats)}")
                        
                        return True
                    return False
                
        except Exception as e:
            logger.error(f"Errore chat: {e}")
            return False
    
    async def hunt_viral_news_with_gradient_ai(self):
        """🧠 Cerca notizie con Gradient Learning AI"""
        try:
            viral_posts = []
            current_time = datetime.now()
            
            for subreddit_name in self.viral_subreddits:
                try:
                    subreddit = await self.reddit.subreddit(subreddit_name)
                    
                    count = 0
                    async for post in subreddit.hot(limit=25):
                        count += 1
                        
                        post_time = datetime.fromtimestamp(post.created_utc)
                        minutes_ago = (current_time - post_time).total_seconds() / 60
                        
                        if minutes_ago <= 360 and post.score >= 10:  # 6 ore, >10 upvotes
                            
                            # 🧠 GRADIENT AI PREDICTION
                            ai_prediction = self.gradient_ai.predict_viral_trajectory(
                                post, subreddit_name, minutes_ago
                            )
                            
                            # Calcola viral score classico
                            viral_score = self.calculate_viral_score(post, subreddit_name, minutes_ago)
                            
                            # Combina AI + score classico
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
                                    'ai_prediction': ai_prediction
                                })
                                
                                # 📊 Traccia per future learning
                                self.gradient_ai.track_prediction(post.id, ai_prediction, post.score)
                        
                        if count >= 25:
                            break
                            
                except Exception as e:
                    logger.warning(f"Errore {subreddit_name}: {e}")
                    continue
            
            viral_posts.sort(key=lambda x: x['viral_score'], reverse=True)
            
            logger.info(f"🧠 Trovati {len(viral_posts)} post con Gradient AI")
            
            return {
                'viral_posts': viral_posts[:8],
                'timestamp': current_time
            }
            
        except Exception as e:
            logger.error(f"Errore Gradient AI hunt: {e}")
            return None
    
    def calculate_viral_score(self, post, subreddit, minutes_ago):
        """Calcola viral score classico (mantenuto)"""
        score = 0
        title_lower = post.title.lower()
        
        if minutes_ago > 0:
            upvotes_per_minute = post.score / minutes_ago
            score += min(upvotes_per_minute * 2, 100)
        
        if post.score > 1000:
            score += 50
        elif post.score > 500:
            score += 30
        elif post.score > 100:
            score += 15
        
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
        
        if minutes_ago > 180:
            score *= 0.5
        
        return int(score)
    
    def categorize_viral_post(self, title, subreddit):
        """Categorizza post"""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['elon', 'tesla']):
            return '🚗 ELON/TESLA'
        elif any(word in title_lower for word in ['ai', 'chatgpt', 'robot']):
            return '🤖 AI/TECH'
        elif any(word in title_lower for word in ['bitcoin', 'crypto', 'stock']):
            return '💰 FINANZA'
        elif any(word in title_lower for word in ['breaking', 'urgent']):
            return '🚨 BREAKING'
        elif any(word in title_lower for word in ['trump', 'biden', 'election']):
            return '🗳️ POLITICS'
        elif any(word in title_lower for word in ['covid', 'health', 'medical']):
            return '🏥 HEALTH'
        elif subreddit == 'todayilearned':
            return '📚 TIL'
        else:
            return '🔥 VIRALE'
    
    def format_viral_message_with_gradient_ai(self, data):
        """📱 Formatta messaggio con Gradient AI"""
        if not data or not data['viral_posts']:
            return "❌ Nessuna notizia virale rilevata."
        
        timestamp = data['timestamp'].strftime("%H:%M - %d/%m/%Y")
        
        message = f"🔥 NOTIZIE VIRALI DELL'ULTIMA ORA 🔥\n"
        message += f"⏰ Scansione: {timestamp}\n"
        message += f"🧠 Powered by Gradient Learning AI\n\n"
        
        message += "📈 TOP NOTIZIE CHE STANNO DIVENTANDO VIRALI:\n"
        
        for i, post in enumerate(data['viral_posts'], 1):
            title = post['title'][:65] + "..." if len(post['title']) > 65 else post['title']
            title = title.replace('[', '').replace(']', '').replace('*', '')
            
            # 🧠 AI DATA
            ai = post['ai_prediction']
            
            # Emoji confidence basato su probabilità AI
            if ai['viral_probability'] >= 75:
                ai_emoji = "🚀🔥"
                confidence_text = "ALTISSIMA"
            elif ai['viral_probability'] >= 60:
                ai_emoji = "⚡📈"
                confidence_text = "ALTA" 
            elif ai['viral_probability'] >= 40:
                ai_emoji = "📊🎯"
                confidence_text = "MEDIA"
            else:
                ai_emoji = "📱💭"
                confidence_text = "BASSA"
            
            message += f"\n{post['category']} {i}. {title}\n"
            message += f"🔥 Viral Score: {post['viral_score']} | "
            message += f"👍 {post['score']} ({post['upvotes_per_min']}/min) | "
            message += f"💬 {post['comments']}\n"
            
            # 🧠 AI PREDICTIONS con Gradient Learning
            message += f"{ai_emoji} AI Gradient ({confidence_text}): {ai['viral_probability']}%\n"
            message += f"📈 Predice → {ai['predicted_final_score']:,} upvotes in {ai['predicted_peak_hours']}h\n"
            message += f"🎯 Pattern: {ai['pattern_match']} (x{ai['pattern_multiplier']})\n"
            
            # Reasoning AI
            if ai.get('reasoning'):
                message += f"🧠 {ai['reasoning']}\n"
            
            message += f"📍 r/{post['subreddit']} | ⏱️ {post['minutes_ago']} min fa\n"
            message += f"🔗 {post['url']}\n"
        
        # Statistiche AI
        total_predictions = len(data['viral_posts'])
        avg_ai_prob = sum(p['ai_prediction']['viral_probability'] for p in data['viral_posts']) / total_predictions
        
        # Conteggio patterns riconosciuti
        patterns_found = {}
        for post in data['viral_posts']:
            pattern = post['ai_prediction']['pattern_match']
            patterns_found[pattern] = patterns_found.get(pattern, 0) + 1
        
        top_pattern = max(patterns_found.items(), key=lambda x: x[1])[0] if patterns_found else 'general'
        
        message += f"\n🧠 GRADIENT AI STATS:\n"
        message += f"📊 {total_predictions} predizioni | Confidence media: {avg_ai_prob:.1f}%\n"
        message += f"🎯 Pattern dominante: {top_pattern} ({patterns_found.get(top_pattern, 0)} notizie)\n"
        message += f"📚 Learning attivo: pesi si adattano automaticamente"
        
        return message
    
    async def send_to_telegram(self, message):
        """📤 Invia a Telegram"""
        if not self.active_chats:
            logger.warning("Nessuna chat attiva")
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
                            logger.info(f"Messaggio inviato: {chat_id}")
                            success_count += 1
                        else:
                            logger.error(f"Errore invio {chat_id}: {response.status}")
                            if response.status in [400, 403, 404]:
                                self.active_chats.discard(chat_id)
                                
                except Exception as e:
                    logger.error(f"Errore invio {chat_id}: {e}")
        
        return success_count > 0
    
    async def run_viral_hunter_with_gradient_ai(self):
        """🚀 MAIN LOOP con Gradient Learning"""
        logger.info("🧠 Avvio Viral News Hunter con Gradient Learning AI...")
        
        if not await self.initialize():
            logger.error("❌ Impossibile inizializzare Reddit!")
            return
        
        logger.info("✅ Gradient Learning AI Bot avviato!")
        logger.info("🧠 AI impara automaticamente dai risultati")
        logger.info("⏰ Scansione ogni 15 minuti + learning check ogni ora")
        
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                
                # Rileva nuove chat
                await self.get_active_chats()
                
                # 🧠 GRADIENT LEARNING CHECK (ogni 4 cicli = ogni ora)
                if cycle_count % 4 == 0:
                    logger.info("🧠 Controllo predizioni passate per learning...")
                    await self.gradient_ai.check_and_learn(self.reddit)
                
                # Cerca viral news con Gradient AI
                logger.info("🔍 Scansione viral con Gradient AI...")
                viral_data = await self.hunt_viral_news_with_gradient_ai()
                
                if viral_data and viral_data['viral_posts']:
                    new_viral = [p for p in viral_data['viral_posts'] if p['id'] not in self.sent_posts]
                    
                    if new_viral and self.active_chats:
                        # Aggiorna sent posts
                        for post in new_viral:
                            self.sent_posts.add(post['id'])
                        
                        viral_data['viral_posts'] = new_viral
                        message = self.format_viral_message_with_gradient_ai(viral_data)
                        success = await self.send_to_telegram(message)
                        
                        if success:
                            logger.info(f"🔥 Inviate {len(new_viral)} notizie con Gradient AI!")
                            
                            # Log AI predictions per monitoring
                            for post in new_viral:
                                ai = post['ai_prediction']
                                logger.info(
                                    f"  🧠 {ai['pattern_match']}: {ai['viral_probability']}% "
                                    f"→ {ai['predicted_final_score']} | {post['title'][:35]}..."
                                )
                        else:
                            logger.warning("⚠️ Errore invio messaggi")
                    
                    elif not self.active_chats:
                        logger.info("⏳ Nessuna chat attiva")
                    else:
                        logger.info("⚠️ Nessuna nuova notizia virale")
                else:
                    logger.info("⚠️ Nessuna notizia virale trovata")
                
                # Pulizia cache
                if len(self.sent_posts) > 1000:
                    self.sent_posts.clear()
                    logger.info("🧹 Cache pulita")
                
                # Log stato AI ogni 8 cicli (ogni 2 ore)
                if cycle_count % 8 == 0:
                    tracked_predictions = len(self.gradient_ai.active_predictions)
                    logger.info(f"🧠 AI Status: {tracked_predictions} predizioni in tracking")
                    
                    # Mostra alcuni pesi appresi
                    key_weights = {
                        'elon_musk': self.gradient_ai.weights.get('elon_musk', 0),
                        'ai_breakthrough': self.gradient_ai.weights.get('ai_breakthrough', 0),
                        'crypto_crash': self.gradient_ai.weights.get('crypto_crash', 0)
                    }
                    logger.info(f"📊 Key weights: {key_weights}")
                
                # 🕐 ATTENDI 15 MINUTI
                logger.info("⏱️ Prossima scansione Gradient AI tra 15 minuti...")
                await asyncio.sleep(900)  # 15 minuti
                
            except KeyboardInterrupt:
                logger.info("🛑 Bot fermato")
                break
            except Exception as e:
                logger.error(f"Errore main loop: {e}")
                logger.info("🔄 Riprovando tra 3 minuti...")
                await asyncio.sleep(180)
        
        # Cleanup
        if self.reddit:
            await self.reddit.close()
            logger.info("🔌 Reddit chiuso")

async def main():
    """Main function"""
    try:
        bot = ViralNewsHunter()
        await bot.run_viral_hunter_with_gradient_ai()
    except Exception as e:
        logger.error(f"Errore critico: {e}")
        await asyncio.sleep(60)

if __name__ == "__main__":
    logger.info("🚀 Launching Gradient Learning AI Viral News Hunter...")
    asyncio.run(main())

