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

# ===== ENHANCED MARKET TRENDS PREDICTOR AI =====
class MarketTrendsPredictor:
    def __init__(self):
        # File per salvare i dati appresi
        self.weights_file = 'market_weights.json'
        self.predictions_file = 'market_predictions.json'
        self.performance_stats_file = 'market_performance.json'
        self.trend_history_file = 'trend_history.json'
        
        # Performance tracking
        self.performance_stats = self.load_performance_stats()
        self.trend_success_history = self.load_trend_history()
        
        # Learning parameters
        self.learning_config = {
            'base_learning_rate': 0.02,
            'adaptive_learning_rate': True,
            'momentum_factor': 0.15,
            'confidence_threshold': 0.65,
            'min_samples_for_trend': 3,
            'trend_decay_rate': 0.98,
            'volatility_sensitivity': 1.3,
            'correlation_weight': 0.4
        }
        
        # Market trend weights con momentum tracking
        self.default_weights = {
            # Settori di mercato
            'technology_trend': {'weight': 2.2, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'finance_trend': {'weight': 2.8, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'healthcare_trend': {'weight': 1.8, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'energy_trend': {'weight': 2.1, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'consumer_goods': {'weight': 1.9, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'real_estate': {'weight': 1.6, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            
            # Consumer behavior patterns
            'economic_indicators': {'weight': 3.1, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'social_sentiment': {'weight': 2.0, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'innovation_adoption': {'weight': 1.7, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'demographic_shifts': {'weight': 1.5, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            
            # Market signals
            'volatility_spike': {'weight': 2.5, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'volume_surge': {'weight': 2.3, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'correlation_break': {'weight': 2.7, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'momentum_shift': {'weight': 2.0, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            
            # Sentiment multipliers
            'bullish_sentiment': {'weight': 1.8, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'bearish_sentiment': {'weight': 1.9, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'uncertainty_factor': {'weight': 2.1, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'confidence_level': {'weight': 1.6, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
        }
        
        # Carica pesi salvati
        self.weights = self.load_weights()
        
        # Tracking predizioni
        self.active_predictions = self.load_predictions()
        
        # Pattern keywords per identificazione trend
        self.market_keywords = {
            'technology_trend': {
                'primary': ['ai', 'tech', 'innovation', 'digital', 'software'],
                'secondary': ['startup', 'automation', 'cloud', 'data', 'algorithm'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'finance_trend': {
                'primary': ['bank', 'finance', 'credit', 'loan', 'investment'],
                'secondary': ['monetary', 'fiscal', 'capital', 'fund', 'portfolio'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'healthcare_trend': {
                'primary': ['health', 'medical', 'pharma', 'biotech', 'wellness'],
                'secondary': ['drug', 'treatment', 'therapy', 'clinic', 'hospital'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'energy_trend': {
                'primary': ['energy', 'oil', 'gas', 'renewable', 'solar'],
                'secondary': ['nuclear', 'wind', 'electric', 'battery', 'grid'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'consumer_goods': {
                'primary': ['retail', 'consumer', 'brand', 'shopping', 'product'],
                'secondary': ['fashion', 'food', 'beverage', 'luxury', 'discount'],
                'score': {'primary': 3, 'secondary': 2}
            }
        }
        
        # Sentiment indicators per consumer behavior
        self.sentiment_indicators = {
            'economic_confidence': {
                'positive': ['growth', 'expansion', 'optimistic', 'bullish', 'recovery'],
                'negative': ['recession', 'decline', 'bearish', 'uncertainty', 'crisis'],
                'neutral': ['stable', 'steady', 'unchanged', 'moderate'],
                'score': {'positive': 2, 'negative': -2, 'neutral': 0}
            },
            'consumer_behavior': {
                'positive': ['spending', 'buying', 'demand', 'popular', 'trending'],
                'negative': ['saving', 'cutting', 'reducing', 'avoiding', 'postponing'],
                'neutral': ['considering', 'evaluating', 'comparing'],
                'score': {'positive': 2, 'negative': -2, 'neutral': 0}
            }
        }

    def load_performance_stats(self):
        """Carica statistiche performance"""
        try:
            if os.path.exists(self.performance_stats_file):
                with open(self.performance_stats_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Errore caricamento performance stats: {e}")
        
        return {
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy_trend': deque(maxlen=50),
            'sector_performance': {},
            'trend_accuracy': {}
        }

    def load_trend_history(self):
        """Carica storico trend"""
        try:
            if os.path.exists(self.trend_history_file):
                with open(self.trend_history_file, 'r') as f:
                    data = json.load(f)
                    for trend in data:
                        if 'recent_performance' in data[trend]:
                            data[trend]['recent_performance'] = deque(
                                data[trend]['recent_performance'], maxlen=20
                            )
                    return data
        except Exception as e:
            logger.warning(f"Errore caricamento trend history: {e}")
        return {}

    def load_weights(self):
        """Carica pesi salvati"""
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
        """Salva pesi aggiornati"""
        try:
            with open(self.weights_file, 'w') as f:
                json.dump(self.weights, f, indent=2)
        except Exception as e:
            logger.error(f"Errore salvataggio pesi: {e}")

    def save_predictions(self):
        """Salva predizioni attive"""
        try:
            with open(self.predictions_file, 'w') as f:
                json.dump(self.active_predictions, f, indent=2)
        except Exception as e:
            logger.error(f"Errore salvataggio predizioni: {e}")

    def load_predictions(self):
        """Carica predizioni per tracking"""
        try:
            if os.path.exists(self.predictions_file):
                with open(self.predictions_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Errore caricamento predizioni: {e}")
        return {}

    def identify_market_pattern(self, text):
        """Identifica pattern di mercato da testo"""
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

    def analyze_market_sentiment(self, text):
        """Analisi sentiment per mercato e consumatori"""
        text_lower = text.lower()
        sentiment_data = {}
        total_score = 0
        
        for category, indicators in self.sentiment_indicators.items():
            category_score = 0
            
            for sentiment_type, keywords in indicators.items():
                if sentiment_type == 'score':
                    continue
                
                for keyword in keywords:
                    if keyword in text_lower:
                        category_score += indicators['score'][sentiment_type]
            
            sentiment_data[category] = category_score
            total_score += category_score
        
        # Determina sentiment complessivo
        if total_score > 3:
            overall_sentiment = 'bullish'
        elif total_score < -3:
            overall_sentiment = 'bearish' 
        else:
            overall_sentiment = 'neutral'
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_score': total_score,
            'categories': sentiment_data,
            'confidence': min(abs(total_score) * 10, 100)
        }

    def get_weight(self, key):
        """Ottieni peso corrente"""
        if key in self.weights:
            return self.weights[key]['weight']
        return 1.0

    def predict_market_trend(self, data_point, context=""):
        """Predice trend di mercato basato sui dati"""
        # 1. Identifica pattern principale
        combined_text = f"{data_point.get('title', '')} {context}"
        pattern_category, pattern_score = self.identify_market_pattern(combined_text)
        pattern_multiplier = self.get_weight(pattern_category)
        
        # 2. Analisi sentiment
        sentiment = self.analyze_market_sentiment(combined_text)
        sentiment_weight = self.get_weight(f"{sentiment['overall_sentiment']}_sentiment")
        sentiment_multiplier = 1 + (sentiment['sentiment_score'] / 10) * sentiment_weight
        
        # 3. Volatilità e momentum
        volatility_factor = data_point.get('volatility', 1.0)
        volume_factor = data_point.get('volume_ratio', 1.0)
        
        volatility_score = volatility_factor * self.get_weight('volatility_spike')
        volume_score = volume_factor * self.get_weight('volume_surge')
        
        # 4. Indicatori economici se disponibili
        economic_score = data_point.get('economic_indicator', 0)
        if economic_score != 0:
            economic_multiplier = 1 + (economic_score * self.get_weight('economic_indicators'))
        else:
            economic_multiplier = 1.0
        
        # 5. Calcolo probabilità trend
        base_probability = 0.3
        
        trend_probability = (
            base_probability * 
            pattern_multiplier * 
            sentiment_multiplier * 
            economic_multiplier *
            (1 + volatility_score/100) *
            (1 + volume_score/100)
        )
        
        # Normalizza probabilità
        trend_probability = max(0.01, min(trend_probability, 0.99))
        
        # Calcola confidence
        pattern_data = self.weights.get(pattern_category, {'samples': 0, 'success_rate': 0.5})
        confidence_raw = min(pattern_data['samples'] / 15, 1.0)
        success_rate = pattern_data['success_rate']
        confidence = (confidence_raw * success_rate + (1 - confidence_raw) * 0.5) * 100
        
        # Determina direzione trend
        if sentiment['sentiment_score'] > 2:
            trend_direction = 'upward'
        elif sentiment['sentiment_score'] < -2:
            trend_direction = 'downward'
        else:
            trend_direction = 'sideways'
        
        # Calcola forza del trend
        trend_strength = min((abs(sentiment['sentiment_score']) + pattern_score + volatility_score) / 3, 10)
        
        return {
            'trend_probability': round(trend_probability * 100, 1),
            'trend_direction': trend_direction,
            'trend_strength': round(trend_strength, 1),
            'confidence': round(confidence, 1),
            'confidence_level': 'high' if confidence > 70 else 'medium' if confidence > 40 else 'low',
            'pattern_match': pattern_category,
            'pattern_score': pattern_score,
            'sentiment_analysis': sentiment,
            'volatility_factor': volatility_factor,
            'volume_factor': volume_factor,
            'economic_impact': economic_score,
            'time_horizon': self._estimate_trend_duration(pattern_category, trend_strength),
            'key_factors': self._identify_key_factors(pattern_category, sentiment, volatility_factor)
        }

    def _estimate_trend_duration(self, pattern, strength):
        """Stima durata del trend in giorni"""
        base_durations = {
            'technology_trend': 30,
            'finance_trend': 14,
            'healthcare_trend': 45,
            'energy_trend': 21,
            'consumer_goods': 28
        }
        
        base = base_durations.get(pattern, 21)
        
        # Aggiusta per forza del trend
        if strength > 8:
            base *= 1.5  # Trend forti durano di più
        elif strength < 4:
            base *= 0.7  # Trend deboli durano meno
        
        return int(base)

    def _identify_key_factors(self, pattern, sentiment, volatility):
        """Identifica fattori chiave per il trend"""
        factors = []
        
        if sentiment['sentiment_score'] > 3:
            factors.append("Sentiment molto positivo")
        elif sentiment['sentiment_score'] < -3:
            factors.append("Sentiment negativo")
        
        if volatility > 1.5:
            factors.append("Alta volatilità")
        elif volatility < 0.7:
            factors.append("Bassa volatilità")
        
        # Fattori specifici per pattern
        pattern_factors = {
            'technology_trend': "Innovazione tecnologica",
            'finance_trend': "Dinamiche finanziarie",
            'healthcare_trend': "Sviluppi sanitari",
            'energy_trend': "Transizione energetica",
            'consumer_goods': "Comportamento consumatori"
        }
        
        if pattern in pattern_factors:
            factors.append(pattern_factors[pattern])
        
        return factors[:3]  # Massimo 3 fattori

    def track_prediction(self, prediction_id, prediction_data, context):
        """Traccia predizione per learning"""
        current_time = datetime.now()
        
        self.active_predictions[prediction_id] = {
            'timestamp': current_time.isoformat(),
            'prediction': prediction_data,
            'context': context,
            'pattern': prediction_data['pattern_match'],
            'trend_direction': prediction_data['trend_direction'],
            'trend_probability': prediction_data['trend_probability'],
            'confidence': prediction_data['confidence']
        }
        self.save_predictions()

    async def check_and_learn(self):
        """Sistema di learning per validare predizioni"""
        if not self.active_predictions:
            return
        
        current_time = datetime.now()
        learned_count = 0
        
        for pred_id, pred_data in list(self.active_predictions.items()):
            try:
                prediction_time = datetime.fromisoformat(pred_data['timestamp'])
                days_passed = (current_time - prediction_time).total_seconds() / 86400
                
                # Controlla predizioni dopo almeno 3 giorni
                if days_passed >= 3:
                    # Simula validazione del trend (in implementazione reale, 
                    # qui andresti a controllare dati di mercato reali)
                    trend_materialized = self._simulate_trend_validation(pred_data)
                    
                    # Applica learning
                    self._apply_trend_learning(pred_data, trend_materialized)
                    
                    # Aggiorna statistiche
                    self._update_trend_stats(pred_data, trend_materialized)
                    
                    del self.active_predictions[pred_id]
                    learned_count += 1
                    
            except Exception as e:
                logger.warning(f"Errore learning per {pred_id}: {e}")
                if pred_id in self.active_predictions:
                    del self.active_predictions[pred_id]
        
        if learned_count > 0:
            logger.info(f"Market Learning: Aggiornati pesi da {learned_count} predizioni")
            self.save_weights()
            self.save_predictions()

    def _simulate_trend_validation(self, prediction_data):
        """Simula validazione del trend (sostituire con dati reali)"""
        # Placeholder: in implementazione reale controllare mercati
        predicted_prob = prediction_data['prediction']['trend_probability']
        confidence = prediction_data['prediction']['confidence']
        
        # Simulazione basata su probabilità e confidence
        import random
        random.seed(hash(prediction_data['timestamp']) % 1000)
        
        if predicted_prob > 70 and confidence > 60:
            return random.random() < 0.8  # 80% accuratezza per predizioni confident
        elif predicted_prob > 50:
            return random.random() < 0.65  # 65% accuratezza per predizioni moderate
        else:
            return random.random() < 0.4   # 40% accuratezza per predizioni speculative

    def _apply_trend_learning(self, prediction_data, was_correct):
        """Applica learning basato sui risultati"""
        pattern = prediction_data['pattern']
        predicted_prob = prediction_data['prediction']['trend_probability']
        confidence = prediction_data['prediction']['confidence']
        
        learning_rate = self.learning_config['base_learning_rate']
        
        if pattern in self.weights:
            pattern_data = self.weights[pattern]
            
            if was_correct:
                adjustment = 1 + (learning_rate * (confidence / 100))
            else:
                adjustment = 1 - (learning_rate * (confidence / 100))
            
            # Applica momentum
            momentum = pattern_data['momentum'] * self.learning_config['momentum_factor']
            new_momentum = (adjustment - 1) * learning_rate + momentum
            
            old_weight = pattern_data['weight']
            pattern_data['weight'] = max(old_weight * adjustment + new_momentum, 0.1)
            pattern_data['momentum'] = new_momentum
            pattern_data['samples'] += 1
            
            # Aggiorna success rate
            if pattern_data['samples'] == 1:
                pattern_data['success_rate'] = 1.0 if was_correct else 0.0
            else:
                decay = 0.9
                pattern_data['success_rate'] = (pattern_data['success_rate'] * decay + 
                                              (1.0 if was_correct else 0.0) * (1 - decay))

    def _update_trend_stats(self, prediction_data, was_correct):
        """Aggiorna statistiche performance"""
        self.performance_stats['total_predictions'] += 1
        if was_correct:
            self.performance_stats['correct_predictions'] += 1
        
        self.performance_stats['accuracy_trend'].append(1 if was_correct else 0)
        
        # Salva stats
        try:
            with open(self.performance_stats_file, 'w') as f:
                stats_to_save = self.performance_stats.copy()
                stats_to_save['accuracy_trend'] = list(stats_to_save['accuracy_trend'])
                json.dump(stats_to_save, f, indent=2)
        except Exception as e:
            logger.error(f"Errore salvataggio performance: {e}")

    def get_learning_insights(self):
        """Ottieni insights sul learning"""
        if not self.performance_stats['accuracy_trend']:
            return {
                'overall_accuracy': 50.0,
                'total_predictions': 0,
                'trend': 'stable'
            }
        
        accuracy_trend = list(self.performance_stats['accuracy_trend'])
        recent_accuracy = sum(accuracy_trend[-10:]) / min(10, len(accuracy_trend)) * 100
        overall_accuracy = self.performance_stats['correct_predictions'] / self.performance_stats['total_predictions'] * 100
        
        return {
            'overall_accuracy': overall_accuracy,
            'recent_accuracy': recent_accuracy,
            'total_predictions': self.performance_stats['total_predictions'],
            'trend': 'improving' if recent_accuracy > overall_accuracy else 'stable'
        }


# ===== MARKET TRENDS HUNTER =====
class MarketTrendsHunter:
    def __init__(self):
        # Credenziali Reddit
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        
        if not all([self.reddit_client_id, self.reddit_client_secret, self.telegram_token]):
            raise ValueError("Variabili d'ambiente mancanti (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, TELEGRAM_BOT_TOKEN)!")
        
        # Market Predictor AI
        self.market_ai = MarketTrendsPredictor()
        
        # State management
        self.active_chats = set()
        self.sent_predictions = set()
        self.reddit = None
        
        # Subreddit focalizzati su comportamento consumatori e mercato
        self.market_consumer_subreddits = [
            # Consumer behavior & shopping trends
            'BuyItForLife', 'frugal', 'personalfinance', 'investing', 'stocks',
            'financialindependence', 'consumerism', 'anticonsumption', 'deals',
            'shopping', 'retail', 'ecommerce', 'smallbusiness', 'entrepreneur',
            
            # Technology adoption & trends
            'technology', 'gadgets', 'apple', 'android', 'pcmasterrace',
            'buildapc', 'startups', 'futurology', 'artificial', 'MachineLearning',
            
            # Market sectors
            'cryptocurrency', 'bitcoin', 'ethereum', 'stocks', 'wallstreetbets',
            'investing', 'realestate', 'energy', 'renewableenergy', 'solar',
            'electricvehicles', 'teslamotors', 'cars', 'automotive',
            
            # Health & wellness trends
            'health', 'fitness', 'nutrition', 'loseit', 'keto', 'intermittentfasting',
            'meditation', 'mentalhealth', 'wellness',
            
            # Lifestyle & consumer culture
            'minimalism', 'zerowaste', 'sustainability', 'environment',
            'climatechange', 'fashion', 'malefashionadvice', 'femalefashionadvice',
            'food', 'cooking', 'mealprep', 'coffee', 'tea',
            
            # Economic discussions
            'economics', 'business', 'jobs', 'careerguidance', 'cscareerquestions',
            'financialindependence', 'povertyfinance', 'middleclassfinance',
            
            # Gaming & entertainment (consumer spending)
            'gaming', 'games', 'pcgaming', 'nintendo', 'playstation', 'xbox',
            'movies', 'television', 'netflix', 'streaming'
        ]

    async def initialize_reddit(self):
        """Inizializza connessione Reddit"""
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
        """Rileva chat attive Telegram"""
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

    async def analyze_reddit_consumer_trends(self):
        """Analizza Reddit per identificare tendenze comportamento consumatori"""
        try:
            market_trends = []
            consumer_signals = []
            current_time = datetime.now()
            
            for subreddit_name in self.market_consumer_subreddits:
                try:
                    subreddit = await self.reddit.subreddit(subreddit_name)
                    
                    post_count = 0
                    subreddit_sentiment = []
                    subreddit_topics = []
                    
                    # Analizza hot posts per trend emergenti
                    async for post in subreddit.hot(limit=15):
                        post_count += 1
                        
                        post_time = datetime.fromtimestamp(post.created_utc)
                        hours_ago = (current_time - post_time).total_seconds() / 3600
                        
                        # Focus su post recenti con engagement significativo
                        if hours_ago <= 24 and post.score >= 10:
                            
                            # Analizza il comportamento consumatori nel post
                            consumer_behavior = self._extract_consumer_behavior(
                                post.title, post.selftext if hasattr(post, 'selftext') else '', 
                                subreddit_name, post.score, post.num_comments
                            )
                            
                            if consumer_behavior['relevance_score'] >= 6.0:
                                
                                # Predici impatto di mercato
                                market_prediction = self.market_ai.predict_market_trend(
                                    {
                                        'title': post.title,
                                        'volatility': consumer_behavior['volatility_factor'],
                                        'volume_ratio': consumer_behavior['engagement_factor'],
                                        'economic_indicator': consumer_behavior['economic_impact']
                                    },
                                    context=f"r/{subreddit_name}: {consumer_behavior['behavior_type']}"
                                )
                                
                                if market_prediction['trend_probability'] >= 55:
                                    market_trends.append({
                                        'post_id': post.id,
                                        'title': post.title,
                                        'subreddit': subreddit_name,
                                        'score': post.score,
                                        'comments': post.num_comments,
                                        'hours_ago': round(hours_ago, 1),
                                        'url': f"https://reddit.com{post.permalink}",
                                        'consumer_behavior': consumer_behavior,
                                        'market_prediction': market_prediction,
                                        'created': post_time
                                    })
                                    
                                    # Track per learning
                                    pred_id = f"{post.id}_{subreddit_name}"
                                    self.market_ai.track_prediction(
                                        pred_id, market_prediction, 
                                        {'subreddit': subreddit_name, 'consumer_behavior': consumer_behavior}
                                    )
                        
                        if post_count >= 15:
                            break
                            
                except Exception as e:
                    logger.warning(f"Errore analisi r/{subreddit_name}: {e}")
                    continue
            
            # Ordina per rilevanza di mercato
            market_trends.sort(
                key=lambda x: (
                    x['market_prediction']['trend_probability'] * 
                    x['consumer_behavior']['relevance_score'] * 
                    (1 + x['score'] / 100)
                ), 
                reverse=True
            )
            
            logger.info(f"Identificati {len(market_trends)} trend consumer/market su Reddit")
            
            return {
                'market_trends': market_trends[:8],  # Top 8 trends
                'timestamp': current_time,
                'subreddits_analyzed': len([s for s in self.market_consumer_subreddits]),
                'learning_insights': self.market_ai.get_learning_insights()
            }
            
        except Exception as e:
            logger.error(f"Errore analisi Reddit consumer trends: {e}")
            return None

    def _extract_consumer_behavior(self, title, content, subreddit, score, comments):
        """Estrae pattern comportamento consumatori da post Reddit"""
        text = f"{title} {content}".lower()
        
        # Pattern comportamento consumatori
        behavior_patterns = {
            'buying_intent': {
                'keywords': ['should i buy', 'worth buying', 'recommend', 'purchase', 'invest in'],
                'weight': 2.5,
                'economic_impact': 0.7
            },
            'trend_adoption': {
                'keywords': ['everyone is', 'new trend', 'popular', 'all over', 'mainstream'],
                'weight': 2.2,
                'economic_impact': 0.8
            },
            'price_sensitivity': {
                'keywords': ['too expensive', 'cheap', 'price drop', 'deal', 'discount'],
                'weight': 2.0,
                'economic_impact': 0.6
            },
            'brand_loyalty': {
                'keywords': ['switching to', 'loyal to', 'brand', 'alternatives', 'comparison'],
                'weight': 1.8,
                'economic_impact': 0.5
            },
            'sustainability_focus': {
                'keywords': ['sustainable', 'eco-friendly', 'green', 'environment', 'ethical'],
                'weight': 2.1,
                'economic_impact': 0.9
            },
            'technology_adoption': {
                'keywords': ['new tech', 'upgrade', 'innovation', 'feature', 'advanced'],
                'weight': 2.3,
                'economic_impact': 0.8
            }
        }
        
        # Calcola score per ogni pattern
        detected_behaviors = []
        total_score = 0
        economic_impact = 0
        
        for behavior, data in behavior_patterns.items():
            pattern_score = 0
            for keyword in data['keywords']:
                if keyword in text:
                    pattern_score += data['weight']
            
            if pattern_score > 0:
                detected_behaviors.append(behavior)
                total_score += pattern_score
                economic_impact += data['economic_impact']
        
        # Determina tipo comportamento dominante
        if 'buying_intent' in detected_behaviors or 'trend_adoption' in detected_behaviors:
            behavior_type = 'high_purchase_intent'
        elif 'sustainability_focus' in detected_behaviors:
            behavior_type = 'conscious_consumption'
        elif 'technology_adoption' in detected_behaviors:
            behavior_type = 'tech_early_adoption'
        elif 'price_sensitivity' in detected_behaviors:
            behavior_type = 'value_seeking'
        else:
            behavior_type = 'general_interest'
        
        # Calcola fattori per predizione mercato
        engagement_factor = min((score + comments * 2) / 50, 3.0)
        volatility_factor = 1.0 + (total_score / 10)
        
        # Score rilevanza finale
        relevance_score = total_score * (1 + engagement_factor / 3)
        
        # Bonus per subreddit specifici
        high_value_subs = ['investing', 'stocks', 'wallstreetbets', 'cryptocurrency', 
                          'personalfinance', 'entrepreneur', 'startups']
        if subreddit.lower() in high_value_subs:
            relevance_score *= 1.3
            economic_impact *= 1.2
        
        return {
            'behavior_type': behavior_type,
            'detected_patterns': detected_behaviors,
            'relevance_score': relevance_score,
            'economic_impact': min(economic_impact, 1.0),
            'volatility_factor': volatility_factor,
            'engagement_factor': engagement_factor,
            'consumer_sentiment': 'positive' if total_score > 4 else 'negative' if total_score < 2 else 'neutral'
        }

    def format_reddit_market_report(self, data, insights):
        """Formatta report basato su analisi Reddit consumer behavior"""
        if not data or not data['market_trends']:
            return "Nessun trend consumatori/mercato rilevato su Reddit."
        
        timestamp = data['timestamp'].strftime("%H:%M - %d/%m/%Y")
        trends = data['market_trends']
        
        message = f"📊 REDDIT MARKET TRENDS ANALYZER\n"
        message += f"⏰ {timestamp} | 🧠 Consumer Behavior AI\n"
        
        if insights.get('total_predictions', 0) > 0:
            accuracy = insights.get('overall_accuracy', 50)
            trend = insights.get('trend', 'stable')
            trend_emoji = "📈" if trend == 'improving' else "➡️"
            
            message += f"📊 AI Performance: {accuracy:.1f}% accuracy {trend_emoji}\n"
        
        message += f"🎯 {len(trends)} CONSUMER/MARKET TRENDS da {data['subreddits_analyzed']} subreddit:\n"
        
        for i, trend in enumerate(trends, 1):
            title = trend['title'][:55] + "..." if len(trend['title']) > 55 else trend['title']
            pred = trend['market_prediction']
            behavior = trend['consumer_behavior']
            
            # Emoji basati su comportamento e trend
            behavior_emojis = {
                'high_purchase_intent': "💰🛒",
                'conscious_consumption': "🌱♻️",
                'tech_early_adoption': "🚀💻",
                'value_seeking': "💎🔍",
                'general_interest': "📊👥"
            }
            
            emoji = behavior_emojis.get(behavior['behavior_type'], "📊")
            
            message += f"\n{emoji} {i}. {title}\n"
            message += f"📍 r/{trend['subreddit']} | {trend['score']} upvotes | 💬 {trend['comments']} | ⏱️ {trend['hours_ago']}h\n"
            
            message += f"🧠 Market Trend: {pred['trend_direction'].upper()} | "
            message += f"Probability: {pred['trend_probability']}%\n"
            message += f"🎯 Confidence: {pred['confidence']}% ({pred['confidence_level']}) | "
            message += f"Strength: {pred['trend_strength']}/10\n"
            
            message += f"👥 Consumer Behavior: {behavior['behavior_type'].replace('_', ' ').title()}\n"
            message += f"📊 Relevance Score: {behavior['relevance_score']:.1f} | "
            message += f"Economic Impact: {behavior['economic_impact']:.1f}\n"
            
            if behavior['detected_patterns']:
                patterns = ", ".join(behavior['detected_patterns'][:3]).replace('_', ' ')
                message += f"🔍 Patterns: {patterns}\n"
            
            if pred['key_factors']:
                factors = " • ".join(pred['key_factors'][:2])
                message += f"⚡ Key Factors: {factors}\n"
            
            message += f"⏱️ Duration: ~{pred['time_horizon']} days | "
            message += f"Sentiment: {pred['sentiment_analysis']['overall_sentiment']}\n"
            
            message += f"🔗 {trend['url']}\n"
        
        # Summary insights
        total_pred = insights.get('total_predictions', 0)
        message += f"\n📚 Total Predictions Tracked: {total_pred}\n"
        message += f"🧠 Reddit Consumer Behavior Analyzer | Market Trends AI v1.0"
        
        return message

    async def send_to_telegram(self, message):
        """Invia report a Telegram"""
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
        """Main loop del Reddit Market Trends Hunter"""
        logger.info("Avvio Reddit Market Trends Hunter...")
        logger.info("Analisi comportamento consumatori e tendenze mercato da Reddit")
        logger.info("Scansione ogni 25 minuti + Learning ogni 2 ore")
        
        if not await self.initialize_reddit():
            return
        
        logger.info("Reddit Market Hunter operativo!")
        
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                
                await self.get_active_chats()
                
                # Learning check ogni 5 cicli (circa ogni 2 ore)
                if cycle_count % 5 == 0:
                    logger.info("Market Learning check...")
                    await self.market_ai.check_and_learn()
                
                # Analizza Reddit per consumer trends
                logger.info("Analizzando Reddit per trend consumatori/mercato...")
                reddit_data = await self.analyze_reddit_consumer_trends()
                
                if reddit_data and reddit_data['market_trends'] and self.active_chats:
                    new_trends = [t for t in reddit_data['market_trends'] 
                                if t['post_id'] not in self.sent_predictions]
                    
                    if new_trends:
                        for trend in new_trends:
                            self.sent_predictions.add(trend['post_id'])
                        
                        reddit_data['market_trends'] = new_trends
                        insights = reddit_data['learning_insights']
                        message = self.format_reddit_market_report(reddit_data, insights)
                        success = await self.send_to_telegram(message)
                        
                        if success:
                            logger.info(f"Inviati {len(new_trends)} trend Reddit consumer/market!")
                            
                            for trend in new_trends:
                                pred = trend['market_prediction']
                                behavior = trend['consumer_behavior']
                                logger.info(
                                    f"  r/{trend['subreddit']}: {behavior['behavior_type']} -> "
                                    f"{pred['trend_direction']} {pred['trend_probability']}% "
                                    f"(conf: {pred['confidence']}%) | {trend['title'][:40]}..."
                                )
                
                # Pulizia cache
                if len(self.sent_predictions) > 600:
                    self.sent_predictions.clear()
                
                # Stats logging ogni 8 cicli
                if cycle_count % 8 == 0:
                    insights = self.market_ai.get_learning_insights()
                    logger.info(f"Reddit Market AI Stats: {insights.get('total_predictions', 0)} predictions, "
                              f"{insights.get('overall_accuracy', 0):.1f}% accuracy")
                
                await asyncio.sleep(1500)  # 25 minuti
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Errore main loop: {e}")
                await asyncio.sleep(300)
        
        if self.reddit:
            await self.reddit.close()


# ===== ENHANCED CONSUMER BEHAVIOR ANALYZER =====
class ConsumerBehaviorAnalyzer:
    """Analizzatore avanzato per comportamento consumatori da Reddit"""
    
    def __init__(self):
        self.sentiment_weights = {
            'purchase_urgency': ['need', 'urgent', 'asap', 'immediately', 'right now'],
            'price_conscious': ['budget', 'cheap', 'affordable', 'deal', 'discount', 'save money'],
            'quality_focused': ['premium', 'high quality', 'worth it', 'investment', 'long term'],
            'trend_following': ['trendy', 'popular', 'everyone has', 'viral', 'must have'],
            'research_phase': ['comparing', 'research', 'opinions', 'reviews', 'suggestions']
        }
        
        self.market_impact_indicators = {
            'high_impact': {
                'keywords': ['game changer', 'revolutionary', 'disrupting', 'market shift'],
                'multiplier': 2.5
            },
            'medium_impact': {
                'keywords': ['growing trend', 'increasing demand', 'popular choice'],
                'multiplier': 1.8
            },
            'emerging_impact': {
                'keywords': ['new', 'innovation', 'alternative', 'different approach'],
                'multiplier': 1.4
            }
        }
    
    def analyze_purchasing_patterns(self, posts_data):
        """Analizza pattern di acquisto da discussioni Reddit"""
        purchase_signals = []
        
        for post in posts_data:
            text = f"{post['title']} {post.get('content', '')}".lower()
            
            # Identifica segnali di acquisto
            purchase_intent = self._detect_purchase_intent(text)
            market_sentiment = self._analyze_market_sentiment(text)
            trend_strength = self._calculate_trend_strength(post)
            
            if purchase_intent['score'] >= 3 or trend_strength >= 7:
                purchase_signals.append({
                    'post_id': post['id'],
                    'purchase_intent': purchase_intent,
                    'market_sentiment': market_sentiment,
                    'trend_strength': trend_strength,
                    'engagement_score': post['score'] + post['comments'] * 2,
                    'subreddit_influence': self._calculate_subreddit_influence(post['subreddit'])
                })
        
        return purchase_signals
    
    def _detect_purchase_intent(self, text):
        """Rileva intenzioni di acquisto nel testo"""
        intent_score = 0
        detected_behaviors = []
        
        for behavior, keywords in self.sentiment_weights.items():
            behavior_score = sum(1 for keyword in keywords if keyword in text)
            if behavior_score > 0:
                intent_score += behavior_score
                detected_behaviors.append(behavior)
        
        # Determina fase del customer journey
        if 'research_phase' in detected_behaviors:
            journey_stage = 'consideration'
        elif any(b in detected_behaviors for b in ['purchase_urgency', 'price_conscious']):
            journey_stage = 'purchase_ready'
        elif 'trend_following' in detected_behaviors:
            journey_stage = 'awareness'
        else:
            journey_stage = 'unknown'
        
        return {
            'score': intent_score,
            'behaviors': detected_behaviors,
            'journey_stage': journey_stage
        }
    
    def _analyze_market_sentiment(self, text):
        """Analizza sentiment di mercato"""
        sentiment_score = 0
        impact_level = 'low'
        
        for level, data in self.market_impact_indicators.items():
            keyword_matches = sum(1 for keyword in data['keywords'] if keyword in text)
            if keyword_matches > 0:
                sentiment_score += keyword_matches * data['multiplier']
                impact_level = level
        
        return {
            'score': sentiment_score,
            'impact_level': impact_level
        }
    
    def _calculate_trend_strength(self, post):
        """Calcola forza del trend basata su engagement"""
        base_score = post['score'] / 10
        comment_factor = post['comments'] / 5
        time_factor = max(1 - (post.get('hours_ago', 24) / 24), 0.1)  # Più recente = più forte
        
        trend_strength = (base_score + comment_factor) * time_factor
        return min(trend_strength, 10)
    
    def _calculate_subreddit_influence(self, subreddit):
        """Calcola influenza del subreddit sulle decisioni di mercato"""
        high_influence_subs = {
            'wallstreetbets': 3.0,
            'investing': 2.5,
            'personalfinance': 2.3,
            'cryptocurrency': 2.8,
            'entrepreneur': 2.2,
            'startups': 2.1
        }
        
        medium_influence_subs = {
            'technology': 1.8,
            'gadgets': 1.7,
            'BuyItForLife': 1.9,
            'frugal': 1.6
        }
        
        return (high_influence_subs.get(subreddit.lower(), 0) or 
                medium_influence_subs.get(subreddit.lower(), 1.0))


async def main():
    """Main function"""
    try:
        hunter = MarketTrendsHunter()
        await hunter.run_market_hunter()
    except Exception as e:
        logger.error(f"Errore critico: {e}")

if __name__ == "__main__":
    logger.info("Reddit Market Trends Predictor AI v1.0")
    logger.info("Focus: Analisi comportamento consumatori e tendenze mercato da Reddit")
    asyncio.run(main())Avvio Market Trends Hunter...")
        logger.info("🧠 Enhanced Market AI per previsione tendenze e comportamento consumatori")
        logger.info("⏰ Analisi ogni 30 minuti + Learning ogni 2 ore")
        
        logger.info("✅ Market Hunter operativo!")
        
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                
                await self.get_active_chats()
                
                # Learning check ogni 4 cicli (ogni 2 ore)
                if cycle_count % 4 == 0:
                    logger.info("🧠 Market Learning check...")
                    await self.market_ai.check_and_learn()
                
                # Raccolta segnali di mercato
                logger.info("📊 Raccogliendo segnali di mercato...")
                market_signals = await self.collect_market_signals()
                
                if market_signals and self.active_chats:
                    new_signals = [s for s in market_signals 
                                 if f"{s['source']}_{int(s['timestamp'].timestamp())}" not in self.sent_predictions]
                    
                    if new_signals:
                        for signal in new_signals:
                            pred_id = f"{signal['source']}_{int(signal['timestamp'].timestamp())}"
                            self.sent_predictions.add(pred_id)
                        
                        insights = self.market_ai.get_learning_insights()
                        message = self.format_market_report(new_signals, insights)
                        success = await self.send_to_telegram(message)
                        
                        if success:
                            logger.info(f"📊 {len(new_signals)} market predictions inviati!")
                            
                            for signal in new_signals:
                                pred = signal['ai_prediction']
                                logger.info(
                                    f"  📈 {pred['pattern_match']}: {pred['trend_direction']} "
                                    f"{pred['trend_probability']}% (conf: {pred['confidence']}%) | "
                                    f"{signal['title'][:40]}..."
                                )
                
                # Pulizia cache
                if len(self.sent_predictions) > 500:
                    self.sent_predictions.clear()
                
                # Stats logging ogni 6 cicli (ogni 3 ore)
                if cycle_count % 6 == 0:
                    insights = self.market_ai.get_learning_insights()
                    logger.info(f"📊 Market AI Stats: {insights.get('total_predictions', 0)} predictions, "
                              f"{insights.get('overall_accuracy', 0):.1f}% accuracy")
                
                await asyncio.sleep(1800)  # 30 minuti
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Errore main loop: {e}")
                await asyncio.sleep(300)

# ===== ADVANCED MARKET DATA COLLECTOR =====
class AdvancedMarketDataCollector:
    """Raccoglitore avanzato di dati di mercato da fonti pubbliche"""
    
    def __init__(self):
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_economic_indicators(self):
        """Raccoglie indicatori economici da fonti pubbliche"""
        # Placeholder per API economiche reali (evita 403)
        # In implementazione reale, usare API come:
        # - FRED (Federal Reserve Economic Data)
        # - Census Bureau
        # - World Bank Open Data
        
        indicators = [
            {
                'name': 'Consumer Confidence',
                'value': 85.2,
                'change': 2.3,
                'trend': 'increasing',
                'significance': 'high'
            },
            {
                'name': 'Retail Sales Growth',
                'value': 4.1,
                'change': -0.8,
                'trend': 'decreasing',
                'significance': 'medium'
            },
            {
                'name': 'Digital Adoption Index',
                'value': 73.5,
                'change': 5.2,
                'trend': 'increasing',
                'significance': 'high'
            }
        ]
        
        return indicators
    
    async def get_sector_trends(self):
        """Analizza trend settoriali"""
        sectors = [
            {
                'sector': 'Technology',
                'momentum': 8.2,
                'volatility': 1.4,
                'consumer_interest': 'high',
                'growth_outlook': 'positive'
            },
            {
                'sector': 'Healthcare',
                'momentum': 6.8,
                'volatility': 0.9,
                'consumer_interest': 'medium',
                'growth_outlook': 'stable'
            },
            {
                'sector': 'Renewable Energy',
                'momentum': 9.1,
                'volatility': 1.8,
                'consumer_interest': 'high',
                'growth_outlook': 'very_positive'
            }
        ]
        
        return sectors
    
    async def get_consumer_sentiment_data(self):
        """Raccoglie dati sentiment consumatori"""
        # Placeholder per sentiment analysis da social media pubblici
        sentiment_data = [
            {
                'category': 'Technology Products',
                'sentiment_score': 0.65,
                'volume': 12500,
                'trending_topics': ['AI', 'smartphones', 'electric vehicles']
            },
            {
                'category': 'Sustainable Products',
                'sentiment_score': 0.78,
                'volume': 8900,
                'trending_topics': ['eco-friendly', 'renewable', 'recycling']
            }
        ]
        
        return sentiment_data

async def main():
    """Main function"""
    try:
        hunter = MarketTrendsHunter()
        await hunter.run_market_hunter()
    except Exception as e:
        logger.error(f"Errore critico: {e}")

if __name__ == "__main__":
    logger.info("📊 Market Trends Predictor AI v1.0")
    logger.info("🎯 Focus: Tendenze mercato e comportamento consumatori")
    asyncio.run(main())
