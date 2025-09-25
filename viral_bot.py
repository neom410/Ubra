import asyncpraw
import asyncio
import time
import os
from datetime import datetime, timedelta
import logging
from collections import Counter, deque
import aiohttp
import re
import json
import math
import statistics

# Configurazione logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('market_trend_analyzer.log')]
)
logger = logging.getLogger(__name__)

# ===== ENHANCED MARKET TREND AI ENGINE =====
class EnhancedMarketTrendAI:
    def __init__(self):
        # File per salvare i pesi appresi
        self.weights_file = 'enhanced_market_weights.json'
        self.predictions_file = 'enhanced_market_predictions.json'
        self.performance_stats_file = 'enhanced_market_performance.json'
        
        # Performance tracking
        self.performance_stats = self.load_performance_stats()
        
        # Learning parameters
        self.learning_config = {
            'base_learning_rate': 0.04,
            'min_samples_for_pattern': 2,
        }
        
        # Pesi specifici per trend di mercato con inizializzazione più aggressiva
        self.default_weights = {
            # Categorie di trend più specifiche
            'consumer_tech_trends': {'weight': 2.8, 'samples': 0, 'success_rate': 0.5},
            'retail_shopping': {'weight': 2.5, 'samples': 0, 'success_rate': 0.5},
            'brand_controversy': {'weight': 3.2, 'samples': 0, 'success_rate': 0.5},
            'market_regulation': {'weight': 2.2, 'samples': 0, 'success_rate': 0.5},
            'product_launches': {'weight': 2.4, 'samples': 0, 'success_rate': 0.5},
            'pricing_changes': {'weight': 2.6, 'samples': 0, 'success_rate': 0.5},
            'consumer_sentiment': {'weight': 2.3, 'samples': 0, 'success_rate': 0.5},
            'industry_disruption': {'weight': 2.9, 'samples': 0, 'success_rate': 0.5},
            'corporate_strategy': {'weight': 2.1, 'samples': 0, 'success_rate': 0.5},
        }
        
        # Carica pesi salvati
        self.weights = self.load_weights()
        self.active_predictions = self.load_predictions()
        
        # Pattern keywords più specifici e mirati
        self.enhanced_market_patterns = {
            'consumer_tech_trends': {
                'primary': ['microsoft', 'apple', 'google', 'amazon', 'meta', 'ai', 'chatgpt', 'windows'],
                'secondary': ['software', 'update', 'technology', 'app', 'digital', 'cloud'],
                'score': {'primary': 4, 'secondary': 2},
                'subreddits': ['technology', 'tech', 'gadgets', 'android', 'apple']
            },
            'retail_shopping': {
                'primary': ['amazon', 'walmart', 'target', 'shop', 'buy', 'purchase', 'sale', 'deal'],
                'secondary': ['shopping', 'retail', 'store', 'price', 'discount', 'black friday'],
                'score': {'primary': 4, 'secondary': 2},
                'subreddits': ['shopping', 'deals', 'discounts', 'ecommerce']
            },
            'brand_controversy': {
                'primary': ['settlement', 'lawsuit', 'billion', 'fine', 'claim', 'dubiously', 'forced'],
                'secondary': ['legal', 'court', 'sue', 'accuse', 'allege', 'controversy'],
                'score': {'primary': 5, 'secondary': 3},
                'subreddits': ['news', 'worldnews', 'business']
            },
            'market_regulation': {
                'primary': ['regulate', 'block', 'ban', 'israel', 'government', 'policy', 'law'],
                'secondary': ['compliance', 'approval', 'restriction', 'authority'],
                'score': {'primary': 4, 'secondary': 2},
                'subreddits': ['politics', 'worldpolitics', 'economics']
            },
            'product_launches': {
                'primary': ['launch', 'release', 'new product', 'announce', 'unveil'],
                'secondary': ['version', 'update', 'feature', 'rollout'],
                'score': {'primary': 3, 'secondary': 2},
                'subreddits': ['gadgets', 'technology', 'productreviews']
            },
            'pricing_changes': {
                'primary': ['price', 'cost', '$', 'billion', 'million', 'fee', 'charge'],
                'secondary': ['expensive', 'cheap', 'affordable', 'value'],
                'score': {'primary': 3, 'secondary': 2},
                'subreddits': ['personalfinance', 'finance', 'shopping']
            },
            'consumer_sentiment': {
                'primary': ['survey', 'report', 'study shows', 'research', 'data indicates'],
                'secondary': ['opinion', 'feedback', 'review', 'rating'],
                'score': {'primary': 3, 'secondary': 2},
                'subreddits': ['dataisbeautiful', 'surveys', 'consumer']
            },
            'industry_disruption': {
                'primary': ['disrupt', 'innovation', 'breakthrough', 'revolutionize', 'transform'],
                'secondary': ['change', 'new era', 'paradigm shift', 'innovative'],
                'score': {'primary': 4, 'secondary': 2},
                'subreddits': ['Futurology', 'innovation', 'startups']
            },
            'corporate_strategy': {
                'primary': ['strategy', 'merger', 'acquisition', 'partnership', 'investment'],
                'secondary': ['business model', 'expansion', 'diversification', 'restructure'],
                'score': {'primary': 3, 'secondary': 2},
                'subreddits': ['business', 'entrepreneur', 'investing']
            }
        }
        
        # Sentiment analysis migliorata per mercato
        self.market_sentiment_indicators = {
            'financial_impact': ['billion', 'million', '$', 'profit', 'revenue', 'settlement', 'fine'],
            'consumer_impact': ['users', 'customers', 'people', 'consumers', 'shoppers'],
            'urgency_indicators': ['now', 'today', 'immediately', 'urgent', 'breaking'],
            'scale_indicators': ['massive', 'huge', 'major', 'significant', 'substantial']
        }
    
    def load_performance_stats(self):
        try:
            if os.path.exists(self.performance_stats_file):
                with open(self.performance_stats_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Errore caricamento stats: {e}")
        
        return {'total_predictions': 0, 'correct_predictions': 0, 'accuracy_trend': deque(maxlen=50)}
    
    def load_weights(self):
        try:
            if os.path.exists(self.weights_file):
                with open(self.weights_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Errore caricamento pesi: {e}")
        
        return self.default_weights.copy()
    
    def save_data(self):
        """Salva tutti i dati"""
        try:
            with open(self.weights_file, 'w') as f:
                json.dump(self.weights, f, indent=2)
            with open(self.predictions_file, 'w') as f:
                json.dump(self.active_predictions, f, indent=2)
            with open(self.performance_stats_file, 'w') as f:
                stats = self.performance_stats.copy()
                stats['accuracy_trend'] = list(stats['accuracy_trend'])
                json.dump(stats, f, indent=2)
        except Exception as e:
            logger.error(f"Errore salvataggio: {e}")
    
    def load_predictions(self):
        try:
            if os.path.exists(self.predictions_file):
                with open(self.predictions_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Errore caricamento predizioni: {e}")
        return {}
    
    def identify_specific_trend(self, title, subreddit):
        """Identificazione più specifica dei trend"""
        title_lower = title.lower()
        subreddit_lower = subreddit.lower()
        
        pattern_scores = {}
        
        for pattern, data in self.enhanced_market_patterns.items():
            score = 0
            
            # Bonus per subreddit matching
            if subreddit_lower in data.get('subreddits', []):
                score += 2
            
            # Primary keywords
            for keyword in data['primary']:
                if keyword.lower() in title_lower:
                    score += data['score']['primary']
            
            # Secondary keywords
            for keyword in data['secondary']:
                if keyword.lower() in title_lower:
                    score += data['score']['secondary']
            
            if score > 0:
                pattern_scores[pattern] = score
        
        if pattern_scores:
            best_pattern = max(pattern_scores.items(), key=lambda x: x[1])
            return best_pattern[0], best_pattern[1]
        
        return 'consumer_sentiment', 0
    
    def analyze_market_impact(self, title):
        """Analisi dell'impatto di mercato"""
        text = title.lower()
        impact_score = 0
        indicators_found = []
        
        # Analisi indicatori finanziari
        for indicator in self.market_sentiment_indicators['financial_impact']:
            if indicator.lower() in text:
                impact_score += 3
                indicators_found.append(indicator)
        
        # Analisi impatto consumatori
        for indicator in self.market_sentiment_indicators['consumer_impact']:
            if indicator.lower() in text:
                impact_score += 2
                indicators_found.append(indicator)
        
        # Indicatori di urgenza
        for indicator in self.market_sentiment_indicators['urgency_indicators']:
            if indicator.lower() in text:
                impact_score += 2
                indicators_found.append(indicator)
        
        # Scala dell'impatto
        for indicator in self.market_sentiment_indicators['scale_indicators']:
            if indicator.lower() in text:
                impact_score += 2
                indicators_found.append(indicator)
        
        if impact_score >= 8:
            intensity = 'high'
        elif impact_score >= 5:
            intensity = 'medium'
        else:
            intensity = 'low'
        
        return {
            'impact_score': impact_score,
            'intensity': intensity,
            'indicators': indicators_found,
            'financial_terms': len([i for i in indicators_found if i in self.market_sentiment_indicators['financial_impact']])
        }
    
    def predict_trend_growth_enhanced(self, post, subreddit, minutes_ago):
        """Predizione migliorata della crescita del trend"""
        
        # 1. Identificazione pattern specifica
        trend_category, pattern_score = self.identify_specific_trend(post.title, subreddit)
        pattern_weight = self.weights.get(trend_category, {'weight': 2.0})['weight']
        
        # Bonus per pattern score alto
        if pattern_score >= 8:
            pattern_multiplier = pattern_weight * 1.4
        elif pattern_score >= 5:
            pattern_multiplier = pattern_weight * 1.2
        else:
            pattern_multiplier = pattern_weight
        
        # 2. Analisi impatto di mercato
        impact_analysis = self.analyze_market_impact(post.title)
        impact_multiplier = 1 + (impact_analysis['impact_score'] / 10)
        
        # 3. Velocità di crescita
        velocity = post.score / max(minutes_ago, 1)
        
        if velocity >= 20:
            velocity_score = 100
            velocity_multiplier = 1.5
        elif velocity >= 10:
            velocity_score = 75
            velocity_multiplier = 1.3
        elif velocity >= 5:
            velocity_score = 50
            velocity_multiplier = 1.1
        else:
            velocity_score = min(velocity * 10, 40)
            velocity_multiplier = 1.0
        
        # 4. Engagement analysis
        comment_ratio = post.num_comments / max(post.score, 1)
        if comment_ratio > 0.1:  # Alto engagement di discussione
            engagement_multiplier = 1.3
        elif comment_ratio > 0.05:
            engagement_multiplier = 1.1
        else:
            engagement_multiplier = 1.0
        
        # 5. Time factor per trend di mercato
        if minutes_ago < 60:  # Molto recente
            time_multiplier = 1.4
        elif minutes_ago < 180:  # Recente
            time_multiplier = 1.2
        elif minutes_ago < 360:  // Medio
            time_multiplier = 1.0
        else:  // Vecchio
            time_multiplier = 0.8
        
        # 6. Calcolo probabilità finale
        base_probability = 0.3  // Base più alta per trend di mercato
        
        final_probability = (
            base_probability * 
            pattern_multiplier * 
            impact_multiplier * 
            velocity_multiplier * 
            engagement_multiplier * 
            time_multiplier
        )
        
        final_probability = max(0.05, min(final_probability, 0.95))
        
        // Calcola confidence basata su historical data
        pattern_data = self.weights.get(trend_category, {'samples': 0, 'success_rate': 0.5})
        confidence = min(pattern_data['samples'] * 10, 80)  // Max 80% con 8+ samples
        if pattern_data['samples'] == 0:
            confidence = 40  // Confidence base per nuovi pattern
        
        // Predizione engagement
        growth_factor = final_probability * 15
        predicted_engagement = int(post.score * (1 + growth_factor))
        
        return {
            'trend_probability': round(final_probability * 100, 1),
            'confidence': confidence,
            'confidence_level': 'high' if confidence > 70 else 'medium' if confidence > 50 else 'low',
            'predicted_engagement': predicted_engagement,
            'trend_category': trend_category,
            'pattern_score': pattern_score,
            'impact_analysis': impact_analysis,
            'velocity_score': velocity_score,
            'velocity_raw': round(velocity, 2),
            'current_engagement': post.score,
            'comment_ratio': round(comment_ratio, 3),
            'reasoning': self.generate_trend_reasoning_enhanced(
                trend_category, pattern_score, impact_analysis, velocity_score, final_probability
            )
        }
    
    def generate_trend_reasoning_enhanced(self, category, pattern_score, impact, velocity, probability):
        """Genera reasoning più dettagliato"""
        reasons = []
        
        reasons.append(f"{category.replace('_', ' ').title()} (score: {pattern_score})")
        
        if impact['intensity'] == 'high':
            reasons.append(f"Alto impatto mercato: {impact['financial_terms']} indicatori finanziari")
        
        if velocity >= 75:
            reasons.append("Crescita virale in corso")
        elif velocity >= 50:
            reasons.append("Crescita sostenuta")
        
        if probability > 0.7:
            reasons.append("Alta probabilità di trend")
        
        return " • ".join(reasons)
    
    def track_prediction(self, post_id, prediction_data, post_score, subreddit, title):
        """Traccia predizione"""
        self.active_predictions[post_id] = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction_data,
            'original_score': post_score,
            'subreddit': subreddit,
            'title': title,
            'trend_category': prediction_data['trend_category']
        }
        self.save_data()
    
    async def check_and_learn(self, reddit):
        """Learning migliorato"""
        if not self.active_predictions:
            return
        
        current_time = datetime.now()
        learned_count = 0
        
        for post_id, prediction_data in list(self.active_predictions.items()):
            try:
                prediction_time = datetime.fromisoformat(prediction_data['timestamp'])
                hours_passed = (current_time - prediction_time).total_seconds() / 3600
                
                if hours_passed >= 8:
                    // Stima outcome (semplificata per ora)
                    predicted_engagement = prediction_data['prediction']['predicted_engagement']
                    actual_engagement = prediction_data['original_score'] * 3  // Stima semplificata
                    
                    was_correct = actual_engagement >= predicted_engagement * 0.7
                    
                    // Aggiorna weights
                    category = prediction_data['trend_category']
                    if category in self.weights:
                        pattern_data = self.weights[category]
                        pattern_data['samples'] += 1
                        
                        if pattern_data['samples'] == 1:
                            pattern_data['success_rate'] = 1.0 if was_correct else 0.0
                        else:
                            new_success = 1.0 if was_correct else 0.0
                            pattern_data['success_rate'] = (pattern_data['success_rate'] * 0.8 + new_success * 0.2)
                        
                        // Aggiorna weight basato su success rate
                        if pattern_data['success_rate'] > 0.6:
                            pattern_data['weight'] *= 1.1
                        elif pattern_data['success_rate'] < 0.4:
                            pattern_data['weight'] *= 0.9
                    
                    // Aggiorna stats
                    self.performance_stats['total_predictions'] += 1
                    if was_correct:
                        self.performance_stats['correct_predictions'] += 1
                    self.performance_stats['accuracy_trend'].append(1 if was_correct else 0)
                    
                    del self.active_predictions[post_id]
                    learned_count += 1
                    
            except Exception as e:
                logger.warning(f"Errore learning {post_id}: {e}")
                if post_id in self.active_predictions:
                    del self.active_predictions[post_id]
        
        if learned_count > 0:
            logger.info(f"Enhanced Learning: appreso da {learned_count} trend")
            self.save_data()
    
    def get_insights(self):
        """Ottieni insights"""
        if self.performance_stats['total_predictions'] == 0:
            return {'overall_accuracy': 50.0, 'total_predictions': 0}
        
        accuracy = (self.performance_stats['correct_predictions'] / 
                   self.performance_stats['total_predictions']) * 100
        
        return {
            'overall_accuracy': round(accuracy, 1),
            'total_predictions': self.performance_stats['total_predictions'],
            'recent_accuracy': round(sum(list(self.performance_stats['accuracy_trend'])[-10:]) / 10 * 100, 1) 
            if len(self.performance_stats['accuracy_trend']) >= 10 else accuracy
        }

# ===== ENHANCED MARKET TREND ANALYZER =====
class EnhancedMarketTrendAnalyzer:
    def __init__(self):
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        
        if not all([self.reddit_client_id, self.reddit_client_secret, self.telegram_token]):
            raise ValueError("Variabili d'ambiente mancanti!")
        
        self.market_ai = EnhancedMarketTrendAI()
        self.active_chats = set()
        self.reddit = None
        self.analyzed_posts = set()
        
        // Subreddit ottimizzati e verificati
        self.optimized_subreddits = [
            'business', 'economics', 'investing', 'stocks',
            'technology', 'tech', 'gadgets',
            'news', 'worldnews',
            'personalfinance', 'finance',
            'marketing', 'entrepreneur',
            'dataisbeautiful', 'consumer'
        ]
    
    async def initialize(self):
        try:
            self.reddit = asyncpraw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent='EnhancedMarketAnalyzer/3.0 (by u/YourUsername)'
            )
            logger.info("✅ Reddit connesso con Enhanced Market AI")
            return True
        except Exception as e:
            logger.error(f"❌ Errore Reddit: {e}")
            return False
    
    async def get_active_chats(self):
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
            logger.error(f"Errore chat: {e}")
            return False
    
    async def analyze_market_trends_enhanced(self):
        try:
            trend_posts = []
            current_time = datetime.now()
            
            for subreddit_name in self.optimized_subreddits:
                try:
                    subreddit = await self.reddit.subreddit(subreddit_name)
                    count = 0
                    
                    async for post in subreddit.hot(limit=12):  // Ridotto per stability
                        count += 1
                        if count > 12:
                            break
                            
                        post_time = datetime.fromtimestamp(post.created_utc)
                        minutes_ago = (current_time - post_time).total_seconds() / 60
                        
                        // Filtri più selettivi
                        if (minutes_ago <= 360 and post.score >= 20 and 
                            post.id not in self.analyzed_posts):
                            
                            prediction = self.market_ai.predict_trend_growth_enhanced(
                                post, subreddit_name, minutes_ago
                            )
                            
                            // Filtro più conservativo per qualità
                            if (prediction['trend_probability'] >= 45 and 
                                prediction['confidence'] >= 40 and
                                post.id not in self.analyzed_posts):
                                
                                trend_posts.append({
                                    'id': post.id,
                                    'title': post.title,
                                    'score': post.score,
                                    'subreddit': subreddit_name,
                                    'url': f"https://reddit.com{post.permalink}",
                                    'comments': post.num_comments,
                                    'created': post_time,
                                    'minutes_ago': round(minutes_ago),
                                    'upvotes_per_min': round(post.score / max(minutes_ago, 1), 1),
                                    'trend_prediction': prediction
                                })
                                
                                self.market_ai.track_prediction(
                                    post.id, prediction, post.score, subreddit_name, post.title
                                )
                        
                        await asyncio.sleep(0.5)  // Rate limiting
                    
                    await asyncio.sleep(1)  // Pausa tra subreddit
                    
                except Exception as e:
                    if "403" in str(e):
                        logger.warning(f"⏸️  Saltato {subreddit_name} (403)")
                    else:
                        logger.warning(f"Errore {subreddit_name}: {e}")
                    continue
            
            trend_posts.sort(key=lambda x: x['trend_prediction']['trend_probability'], reverse=True)
            logger.info(f"📈 Enhanced Market AI: {len(trend_posts)} trend di qualità rilevati")
            
            return {
                'trend_posts': trend_posts[:4],  // Massimo 4 trend di alta qualità
                'timestamp': current_time,
                'insights': self.market_ai.get_insights()
            }
            
        except Exception as e:
            logger.error(f"Errore analisi: {e}")
            return None
    
    def format_enhanced_alert(self, data):
        if not data or not data['trend_posts']:
            return "📊 Nessun trend di alta qualità rilevato."
        
        timestamp = data['timestamp'].strftime("%H:%M - %d/%m/%Y")
        posts = data['trend_posts']
        insights = data.get('insights', {})
        
        message = f"🎯 ENHANCED MARKET TREND ANALYZER 🎯\n"
        message += f"⏰ {timestamp} | 🧠 AI Precisione: {insights.get('overall_accuracy', 50)}%\n\n"
        message += f"🔥 {len(posts)} TREND AD ALTA PROBABILITÀ:\n"
        
        for i, post in enumerate(posts, 1):
            trend = post['trend_prediction']
            title = post['title'][:70] + "..." if len(post['title']) > 70 else post['title']
            
            // Emoji basate su categoria e probabilità
            category_emojis = {
                'consumer_tech_trends': '💻',
                'retail_shopping': '🛍️',
                'brand_controversy': '⚖️',
                'market_regulation': '🏛️',
                'product_launches': '🚀',
                'pricing_changes': '💰',
                'consumer_sentiment': '😊',
                'industry_disruption': '⚡',
                'corporate_strategy': '🏢'
            }
            
            emoji = category_emojis.get(trend['trend_category'], '📈')
            
            if trend['trend_probability'] >= 70:
                level = "TREND ESPLOSIVO"
                level_emoji = "🚀"
            elif trend['trend_probability'] >= 55:
                level = "TREND FORTE"
                level_emoji = "⚡"
            else:
                level = "TREND EMERGENTE"
                level_emoji = "📊"
            
            message += f"\n{emoji}{level_emoji} {i}. {title}\n"
            message += f"📈 Probabilità: {trend['trend_probability']}% | Affidabilità: {trend['confidence']}%\n"
            message += f"🎯 Categoria: {trend['trend_category'].replace('_', ' ').title()}\n"
            message += f"📊 Impatto: {trend['impact_analysis']['intensity'].upper()} "
            message += f"({trend['impact_analysis']['financial_terms']} indicatori finanziari)\n"
            message += f"⚡ Crescita: {trend['velocity_raw']}/min | Engagement: {post['score']} ↑ | 💬 {post['comments']}\n"
            message += f"📍 r/{post['subreddit']} | ⏱️ {post['minutes_ago']} min fa\n"
            message += f"🔗 {post['url']}\n"
        
        message += f"\n📚 Trend analizzati: {insights.get('total_predictions', 0)}"
        message += f" | Precisione recente: {insights.get('recent_accuracy', 50)}%\n"
        message += f"⚡ Enhanced Market Trend Analyzer v3.0 | Categorizzazione Avanzata"
        
        return message
    
    async def send_alert(self, message):
        if not self.active_chats:
            return False
        
        success_count = 0
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            for chat_id in self.active_chats:
                try:
                    url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                    payload = {'chat_id': chat_id, 'text': message, 'disable_web_page_preview': True}
                    
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            success_count += 1
                except Exception as e:
                    logger.error(f"Errore invio {chat_id}: {e}")
        
        return success_count > 0
    
    async def run_enhanced_analyzer(self):
        logger.info("🎯 Avvio Enhanced Market Trend Analyzer...")
        logger.info("🧠 AI con categorizzazione avanzata e filtri di qualità")
        logger.info("⏰ Scansione ogni 15 minuti")
        
        if not await self.initialize():
            return
        
        logger.info("✅ Enhanced Analyzer operativo!")
        
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                
                await self.get_active_chats()
                
                // Learning ogni 6 cicli (1.5 ore)
                if cycle_count % 6 == 0:
                    logger.info("🧠 Enhanced Learning in corso...")
                    await self.market_ai.check_and_learn(self.reddit)
                
                // Analisi trend
                logger.info("🔍 Scansione trend di mercato...")
                trend_data = await self.analyze_market_trends_enhanced()
                
                if trend_data and trend_data['trend_posts']:
                    new_trends = [p for p in trend_data['trend_posts'] if p['id'] not in self.analyzed_posts]
                    
                    if new_trends and self.active_chats:
                        for post in new_trends:
                            self.analyzed_posts.add(post['id'])
                        
                        alert_message = self.format_enhanced_alert(trend_data)
                        success = await self.send_alert(alert_message)
                        
                        if success:
                            logger.info(f"🎯 {len(new_trends)} trend alerts inviati!")
                            for post in new_trends:
                                trend = post['trend_prediction']
                                logger.info(
                                    f"  {trend['trend_category']}: {trend['trend_probability']}% "
                                    f"(conf: {trend['confidence']}%) | {post['title'][:40]}..."
                                )
                
                // Pulizia cache
                if len(self.analyzed_posts) > 800:
                    self.analyzed_posts.clear()
                
                // Log stats
                if cycle_count % 8 == 0:
                    insights = self.market_ai.get_insights()
                    logger.info(f"📊 Enhanced AI: {insights['total_predictions']} analisi, "
                              f"{insights['overall_accuracy']}% accuracy")
                
                await asyncio.sleep(900)  // 15 minuti
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Errore main loop: {e}")
                await asyncio.sleep(300)
        
        if self.reddit:
            await self.reddit.close()

async def main():
    try:
        analyzer = EnhancedMarketTrendAnalyzer()
        await analyzer.run_enhanced_analyzer()
    except Exception as e:
        logger.error(f"Errore critico: {e}")

if __name__ == "__main__":
    logger.info("🎯 Enhanced Market Trend Analyzer v3.0 - Advanced Categorization")
    asyncio.run(main())
