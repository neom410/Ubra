import asyncpraw
import asyncio
import aiohttp
import json
import os
import logging
from datetime import datetime, timedelta
from collections import Counter, deque
import statistics
import math
import re

# Configurazione logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('market_predictor.log')]
)
logger = logging.getLogger(__name__)

# ===== MARKET TRENDS PREDICTOR AI =====
class MarketTrendsPredictor:
    def __init__(self):
        # File per salvare i dati appresi
        self.weights_file = 'market_weights.json'
        self.predictions_file = 'market_predictions.json'
        self.performance_stats_file = 'market_performance.json'
        
        # Performance tracking
        self.performance_stats = self.load_performance_stats()
        
        # Learning parameters
        self.learning_config = {
            'base_learning_rate': 0.02,
            'momentum_factor': 0.15,
            'confidence_threshold': 0.65,
            'min_samples_for_trend': 3
        }
        
        # Market trend weights
        self.default_weights = {
            'technology_trend': {'weight': 2.2, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'finance_trend': {'weight': 2.8, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'consumer_goods': {'weight': 1.9, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'healthcare_trend': {'weight': 1.8, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'energy_trend': {'weight': 2.1, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'economic_indicators': {'weight': 3.1, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'social_sentiment': {'weight': 2.0, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'volatility_spike': {'weight': 2.5, 'momentum': 0, 'samples': 0, 'success_rate': 0.5}
        }
        
        self.weights = self.load_weights()
        self.active_predictions = self.load_predictions()
        
        # Pattern keywords per identificazione trend
        self.market_keywords = {
            'technology_trend': {
                'primary': ['ai', 'tech', 'innovation', 'digital', 'software'],
                'secondary': ['startup', 'automation', 'cloud', 'data'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'finance_trend': {
                'primary': ['bank', 'finance', 'credit', 'investment', 'money'],
                'secondary': ['fund', 'capital', 'loan', 'debt'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'consumer_goods': {
                'primary': ['retail', 'consumer', 'brand', 'shopping', 'product'],
                'secondary': ['fashion', 'food', 'luxury', 'discount'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'healthcare_trend': {
                'primary': ['health', 'medical', 'pharma', 'biotech'],
                'secondary': ['drug', 'treatment', 'therapy', 'wellness'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'energy_trend': {
                'primary': ['energy', 'oil', 'gas', 'renewable', 'solar'],
                'secondary': ['nuclear', 'wind', 'electric', 'battery'],
                'score': {'primary': 3, 'secondary': 2}
            }
        }

    def load_performance_stats(self):
        try:
            if os.path.exists(self.performance_stats_file):
                with open(self.performance_stats_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Errore caricamento performance stats: {e}")
        
        return {
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy_trend': []
        }

    def load_weights(self):
        try:
            if os.path.exists(self.weights_file):
                with open(self.weights_file, 'r') as f:
                    saved_weights = json.load(f)
                    weights = {}
                    for key, default_data in self.default_weights.items():
                        if key in saved_weights:
                            if isinstance(saved_weights[key], dict):
                                weights[key] = saved_weights[key]
                            else:
                                weights[key] = {
                                    'weight': saved_weights[key],
                                    'momentum': 0,
                                    'samples': 0,
                                    'success_rate': 0.5
                                }
                        else:
                            weights[key] = default_data.copy()
                    
                    logger.info(f"Caricati pesi Market AI da {self.weights_file}")
                    return weights
        except Exception as e:
            logger.warning(f"Errore caricamento pesi: {e}")
        
        return {k: v.copy() for k, v in self.default_weights.items()}

    def save_weights(self):
        try:
            with open(self.weights_file, 'w') as f:
                json.dump(self.weights, f, indent=2)
        except Exception as e:
            logger.error(f"Errore salvataggio pesi: {e}")

    def load_predictions(self):
        try:
            if os.path.exists(self.predictions_file):
                with open(self.predictions_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Errore caricamento predizioni: {e}")
        return {}

    def save_predictions(self):
        try:
            with open(self.predictions_file, 'w') as f:
                json.dump(self.active_predictions, f, indent=2)
        except Exception as e:
            logger.error(f"Errore salvataggio predizioni: {e}")

    def identify_market_pattern(self, text):
        text_lower = text.lower()
        pattern_scores = {}
        
        for pattern, keywords in self.market_keywords.items():
            score = 0
            
            for keyword in keywords['primary']:
                if keyword in text_lower:
                    score += keywords['score']['primary']
            
            for keyword in keywords['secondary']:
                if keyword in text_lower:
                    score += keywords['score']['secondary']
            
            if score > 0:
                pattern_scores[pattern] = score
        
        if pattern_scores:
            best_pattern = max(pattern_scores.items(), key=lambda x: x[1])
            return best_pattern[0], best_pattern[1]
        
        return 'general_market', 0

    def get_weight(self, key):
        if key in self.weights:
            return self.weights[key]['weight']
        return 1.0

    def predict_market_trend(self, data_point, context=""):
        combined_text = f"{data_point.get('title', '')} {context}"
        pattern_category, pattern_score = self.identify_market_pattern(combined_text)
        pattern_multiplier = self.get_weight(pattern_category)
        
        # Analisi sentiment semplificata
        positive_words = ['growth', 'increase', 'up', 'bullish', 'strong']
        negative_words = ['decline', 'down', 'bearish', 'weak', 'fall']
        
        sentiment_score = 0
        text_lower = combined_text.lower()
        for word in positive_words:
            if word in text_lower:
                sentiment_score += 1
        for word in negative_words:
            if word in text_lower:
                sentiment_score -= 1
        
        sentiment_multiplier = 1 + (sentiment_score * 0.1)
        
        # Calcolo probabilità trend
        base_probability = 0.3
        volatility_factor = data_point.get('volatility', 1.0)
        
        trend_probability = (
            base_probability * 
            pattern_multiplier * 
            sentiment_multiplier * 
            volatility_factor
        )
        
        trend_probability = max(0.01, min(trend_probability, 0.99))
        
        # Determina direzione
        if sentiment_score > 0:
            trend_direction = 'upward'
        elif sentiment_score < 0:
            trend_direction = 'downward'
        else:
            trend_direction = 'sideways'
        
        # Calcola confidence
        pattern_data = self.weights.get(pattern_category, {'samples': 0, 'success_rate': 0.5})
        confidence = min(pattern_data['samples'] / 10, 1.0) * pattern_data['success_rate'] * 100
        
        return {
            'trend_probability': round(trend_probability * 100, 1),
            'trend_direction': trend_direction,
            'confidence': round(confidence, 1),
            'pattern_match': pattern_category,
            'pattern_score': pattern_score,
            'sentiment_score': sentiment_score
        }

    def get_learning_insights(self):
        if self.performance_stats['total_predictions'] == 0:
            return {
                'overall_accuracy': 50.0,
                'total_predictions': 0,
                'trend': 'stable'
            }
        
        accuracy = (self.performance_stats['correct_predictions'] / 
                   self.performance_stats['total_predictions']) * 100
        
        return {
            'overall_accuracy': accuracy,
            'total_predictions': self.performance_stats['total_predictions'],
            'trend': 'stable'
        }


# ===== MARKET TRENDS HUNTER =====
class MarketTrendsHunter:
    def __init__(self):
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        
        if not all([self.reddit_client_id, self.reddit_client_secret, self.telegram_token]):
            raise ValueError("Variabili d'ambiente mancanti!")
        
        self.market_ai = MarketTrendsPredictor()
        self.active_chats = set()
        self.sent_predictions = set()
        self.reddit = None
        
        self.market_subreddits = [
            'investing', 'stocks', 'personalfinance', 'cryptocurrency',
            'technology', 'startups', 'business', 'economics',
            'wallstreetbets', 'gadgets', 'entrepreneur', 'smallbusiness'
        ]

    async def initialize_reddit(self):
        try:
            self.reddit = asyncpraw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent='MarketTrendsPredictor/1.0'
            )
            logger.info("Reddit connesso per analisi tendenze mercato")
            return True
        except Exception as e:
            logger.error(f"Errore connessione Reddit: {e}")
            return False

    async def get_active_chats(self):
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
                            
                            if data['result']:
                                last_update_id = data['result'][-1]['update_id']
                                clear_url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates?offset={last_update_id + 1}"
                                await session.get(clear_url)
                            
                            if new_chats > 0:
                                logger.info(f"Nuove chat: {new_chats}. Totale: {len(self.active_chats)}")
                        
                        return True
                    return False
                
        except Exception as e:
            logger.error(f"Errore chat detection: {e}")
            return False

    async def analyze_reddit_trends(self):
        try:
            market_trends = []
            current_time = datetime.now()
            
            for subreddit_name in self.market_subreddits:
                try:
                    subreddit = await self.reddit.subreddit(subreddit_name)
                    
                    post_count = 0
                    async for post in subreddit.hot(limit=10):
                        post_count += 1
                        
                        post_time = datetime.fromtimestamp(post.created_utc)
                        hours_ago = (current_time - post_time).total_seconds() / 3600
                        
                        if hours_ago <= 24 and post.score >= 20:
                            prediction = self.market_ai.predict_market_trend(
                                {'title': post.title, 'volatility': 1.0},
                                context=f"r/{subreddit_name}"
                            )
                            
                            if prediction['trend_probability'] >= 60:
                                market_trends.append({
                                    'post_id': post.id,
                                    'title': post.title,
                                    'subreddit': subreddit_name,
                                    'score': post.score,
                                    'comments': post.num_comments,
                                    'hours_ago': round(hours_ago, 1),
                                    'url': f"https://reddit.com{post.permalink}",
                                    'prediction': prediction
                                })
                        
                        if post_count >= 10:
                            break
                            
                except Exception as e:
                    logger.warning(f"Errore analisi r/{subreddit_name}: {e}")
                    continue
            
            market_trends.sort(key=lambda x: x['prediction']['trend_probability'], reverse=True)
            
            logger.info(f"Identificati {len(market_trends)} trend di mercato")
            
            return {
                'trends': market_trends[:5],
                'timestamp': current_time,
                'insights': self.market_ai.get_learning_insights()
            }
            
        except Exception as e:
            logger.error(f"Errore analisi Reddit trends: {e}")
            return None

    def format_market_report(self, data):
        if not data or not data['trends']:
            return "Nessun trend di mercato rilevato."
        
        timestamp = data['timestamp'].strftime("%H:%M - %d/%m/%Y")
        trends = data['trends']
        insights = data['insights']
        
        message = f"Market Trends Predictor\n"
        message += f"{timestamp} | Reddit Analysis\n"
        
        if insights['total_predictions'] > 0:
            accuracy = insights['overall_accuracy']
            message += f"AI Performance: {accuracy:.1f}% accuracy\n"
        
        message += f"{len(trends)} MARKET TRENDS:\n"
        
        for i, trend in enumerate(trends, 1):
            title = trend['title'][:60] + "..." if len(trend['title']) > 60 else trend['title']
            pred = trend['prediction']
            
            if pred['trend_direction'] == 'upward':
                emoji = "📈"
            elif pred['trend_direction'] == 'downward':
                emoji = "📉"
            else:
                emoji = "➡️"
            
            message += f"\n{emoji} {i}. {title}\n"
            message += f"r/{trend['subreddit']} | {trend['score']} upvotes | {trend['comments']} comments\n"
            message += f"Trend: {pred['trend_direction'].upper()} | Probability: {pred['trend_probability']}%\n"
            message += f"Pattern: {pred['pattern_match']} | Confidence: {pred['confidence']}%\n"
            message += f"{trend['url']}\n"
        
        message += f"\nTotal Predictions: {insights['total_predictions']}\n"
        message += f"Market Trends AI v1.0"
        
        return message

    async def send_to_telegram(self, message):
        if not self.active_chats:
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
                            success_count += 1
                        else:
                            if response.status in [400, 403, 404]:
                                self.active_chats.discard(chat_id)
                                
                except Exception as e:
                    logger.error(f"Errore invio {chat_id}: {e}")
        
        return success_count > 0

    async def run_market_hunter(self):
        logger.info("Avvio Reddit Market Trends Hunter")
        logger.info("Analisi ogni 30 minuti")
        
        if not await self.initialize_reddit():
            return
        
        logger.info("Market Hunter operativo!")
        
        while True:
            try:
                await self.get_active_chats()
                
                logger.info("Analizzando Reddit per trend di mercato...")
                data = await self.analyze_reddit_trends()
                
                if data and data['trends'] and self.active_chats:
                    new_trends = [t for t in data['trends'] 
                                if t['post_id'] not in self.sent_predictions]
                    
                    if new_trends:
                        for trend in new_trends:
                            self.sent_predictions.add(trend['post_id'])
                        
                        data['trends'] = new_trends
                        message = self.format_market_report(data)
                        success = await self.send_to_telegram(message)
                        
                        if success:
                            logger.info(f"Inviati {len(new_trends)} trend di mercato!")
                
                if len(self.sent_predictions) > 500:
                    self.sent_predictions.clear()
                
                await asyncio.sleep(1800)  # 30 minuti
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Errore main loop: {e}")
                await asyncio.sleep(300)
        
        if self.reddit:
            await self.reddit.close()


async def main():
    try:
        hunter = MarketTrendsHunter()
        await hunter.run_market_hunter()
    except Exception as e:
        logger.error(f"Errore critico: {e}")


if __name__ == "__main__":
    logger.info("Market Trends Predictor AI v1.0")
    logger.info("Analisi tendenze mercato da Reddit")
    asyncio.run(main())
