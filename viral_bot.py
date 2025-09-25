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

# ===== MARKET TREND AI ENGINE =====
class MarketTrendAI:
    def __init__(self):
        # File per salvare i pesi appresi
        self.weights_file = 'market_trend_weights.json'
        self.predictions_file = 'market_predictions_track.json'
        self.performance_stats_file = 'market_performance_stats.json'
        self.pattern_history_file = 'market_pattern_history.json'
        
        # Performance tracking per migliorare learning
        self.performance_stats = self.load_performance_stats()
        self.pattern_success_history = self.load_pattern_history()
        
        # Learning parameters migliorati
        self.learning_config = {
            'base_learning_rate': 0.03,
            'adaptive_learning_rate': True,
            'momentum_factor': 0.1,
            'confidence_threshold': 0.7,
            'min_samples_for_pattern': 3,
            'pattern_decay_rate': 0.99,
            'success_boost_adaptive': True,
            'failure_penalty_adaptive': True
        }
        
        # Pesi specifici per trend di mercato
        self.default_weights = {
            # Categorie di trend di mercato
            'consumer_behavior': {'weight': 2.5, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'product_trends': {'weight': 2.2, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'brand_sentiment': {'weight': 2.8, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'shopping_trends': {'weight': 2.4, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'market_research': {'weight': 1.8, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'pricing_trends': {'weight': 2.1, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'industry_analysis': {'weight': 1.9, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'consumer_insights': {'weight': 2.3, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'retail_trends': {'weight': 2.0, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'demand_analysis': {'weight': 2.6, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            
            # Sentiment specifici per mercato
            'purchase_intent': {'weight': 2.7, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'brand_loyalty': {'weight': 1.7, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'price_sensitivity': {'weight': 2.2, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'market_adoption': {'weight': 2.4, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            
            # Velocity thresholds adaptive
            'trend_explosive_threshold': {'weight': 25.0, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'trend_fast_threshold': {'weight': 12.0, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'trend_steady_threshold': {'weight': 5.0, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
        }
        
        # Carica pesi salvati o usa default
        self.weights = self.load_weights()
        
        # Tracking predizioni per feedback migliorato
        self.active_predictions = self.load_predictions()
        
        # Pattern keywords per trend di mercato
        self.market_pattern_keywords = {
            'consumer_behavior': {
                'primary': ['consumer behavior', 'buying habits', 'shopping patterns', 'purchase decision'],
                'secondary': ['customer preference', 'shopping behavior', 'consumer trend', 'buying pattern'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'product_trends': {
                'primary': ['product trend', 'new product', 'product launch', 'best selling'],
                'secondary': ['product review', 'product comparison', 'popular product', 'trending product'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'brand_sentiment': {
                'primary': ['brand sentiment', 'brand reputation', 'customer satisfaction', 'brand loyalty'],
                'secondary': ['brand review', 'brand comparison', 'brand perception', 'customer feedback'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'shopping_trends': {
                'primary': ['shopping trend', 'retail trend', 'ecommerce', 'online shopping'],
                'secondary': ['black friday', 'cyber monday', 'shopping season', 'holiday shopping'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'market_research': {
                'primary': ['market research', 'market analysis', 'industry report', 'consumer survey'],
                'secondary': ['market study', 'research report', 'analysis shows', 'survey results'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'pricing_trends': {
                'primary': ['pricing trend', 'price change', 'discount', 'price drop'],
                'secondary': ['price increase', 'sale price', 'competitive pricing', 'price strategy'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'industry_analysis': {
                'primary': ['industry analysis', 'sector trend', 'market outlook', 'industry report'],
                'secondary': ['market forecast', 'industry insight', 'sector analysis', 'market prediction'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'consumer_insights': {
                'primary': ['consumer insight', 'customer feedback', 'user review', 'customer opinion'],
                'secondary': ['consumer survey', 'feedback analysis', 'review analysis', 'customer voice'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'retail_trends': {
                'primary': ['retail trend', 'store traffic', 'retail sales', 'shopping mall'],
                'secondary': ['brick and mortar', 'retail analysis', 'store performance', 'retail data'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'demand_analysis': {
                'primary': ['demand analysis', 'consumer demand', 'market demand', 'product demand'],
                'secondary': ['demand forecast', 'demand trend', 'demand pattern', 'seasonal demand'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'purchase_intent': {
                'primary': ['purchase intent', 'buying intention', 'shopping list', 'plan to buy'],
                'secondary': ['considering purchase', 'intend to buy', 'shopping cart', 'wishlist'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'brand_loyalty': {
                'primary': ['brand loyalty', 'customer loyalty', 'repeat purchase', 'loyal customer'],
                'secondary': ['brand switching', 'customer retention', 'loyalty program', 'preferred brand'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'price_sensitivity': {
                'primary': ['price sensitive', 'price elasticity', 'budget conscious', 'value seeking'],
                'secondary': ['price comparison', 'affordability', 'price point', 'cost conscious'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'market_adoption': {
                'primary': ['market adoption', 'product adoption', 'early adopter', 'technology adoption'],
                'secondary': ['adoption rate', 'market penetration', 'user adoption', 'consumer adoption'],
                'score': {'primary': 3, 'secondary': 2}
            }
        }
        
        # Sentiment keywords specifici per mercato
        self.market_sentiment_keywords = {
            'purchase_urgency': {
                'extreme': ['must buy', 'urgent purchase', 'limited time', 'last chance'],
                'high': ['need to buy', 'shopping now', 'buy today', 'immediate purchase'],
                'medium': ['planning to buy', 'considering purchase', 'shopping soon'],
                'score': {'extreme': 4, 'high': 3, 'medium': 2}
            },
            'value_perception': {
                'extreme': ['great value', 'excellent price', 'amazing deal', 'best value'],
                'high': ['good price', 'fair price', 'reasonable cost', 'worth it'],
                'medium': ['affordable', 'budget friendly', 'cost effective'],
                'score': {'extreme': 4, 'high': 3, 'medium': 2}
            },
            'quality_sentiment': {
                'extreme': ['excellent quality', 'premium product', 'top notch', 'outstanding'],
                'high': ['good quality', 'reliable', 'durable', 'well made'],
                'medium': ['decent quality', 'acceptable', 'satisfactory'],
                'score': {'extreme': 4, 'high': 3, 'medium': 2}
            },
            'brand_engagement': {
                'extreme': ['love this brand', 'brand ambassador', 'loyal customer', 'fan of'],
                'high': ['recommend this brand', 'satisfied customer', 'happy with'],
                'medium': ['like this brand', 'prefer this brand', 'good experience'],
                'score': {'extreme': 4, 'high': 3, 'medium': 2}
            }
        }
    
    def load_performance_stats(self):
        """Carica statistiche performance per learning adaptivo"""
        try:
            if os.path.exists(self.performance_stats_file):
                with open(self.performance_stats_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Errore caricamento performance stats: {e}")
        
        return {
            'total_predictions': 0,
            'correct_predictions': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'accuracy_trend': deque(maxlen=50),
            'pattern_performance': {},
            'hourly_accuracy': {},
            'subreddit_performance': {}
        }
    
    def load_pattern_history(self):
        """Carica storico performance pattern"""
        try:
            if os.path.exists(self.pattern_history_file):
                with open(self.pattern_history_file, 'r') as f:
                    data = json.load(f)
                    for pattern in data:
                        if 'recent_performance' in data[pattern]:
                            data[pattern]['recent_performance'] = deque(
                                data[pattern]['recent_performance'], maxlen=20
                            )
                    return data
        except Exception as e:
            logger.warning(f"Errore caricamento pattern history: {e}")
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
                    
                    logger.info(f"Caricati pesi Market Trend AI da {self.weights_file}")
                    return weights
        except Exception as e:
            logger.warning(f"Errore caricamento pesi: {e}")
        
        logger.info("Inizializzo pesi Market Trend AI default")
        return {k: v.copy() for k, v in self.default_weights.items()}
    
    def save_weights(self):
        """Salva pesi aggiornati"""
        try:
            with open(self.weights_file, 'w') as f:
                json.dump(self.weights, f, indent=2)
            logger.debug("Pesi Market Trend AI salvati")
        except Exception as e:
            logger.error(f"Errore salvataggio pesi: {e}")
    
    def save_performance_stats(self):
        """Salva statistiche performance"""
        try:
            stats_to_save = self.performance_stats.copy()
            stats_to_save['accuracy_trend'] = list(stats_to_save['accuracy_trend'])
            
            with open(self.performance_stats_file, 'w') as f:
                json.dump(stats_to_save, f, indent=2)
        except Exception as e:
            logger.error(f"Errore salvataggio performance stats: {e}")
    
    def save_pattern_history(self):
        """Salva storico pattern"""
        try:
            history_to_save = {}
            for pattern, data in self.pattern_success_history.items():
                history_to_save[pattern] = data.copy()
                if 'recent_performance' in data:
                    history_to_save[pattern]['recent_performance'] = list(data['recent_performance'])
            
            with open(self.pattern_history_file, 'w') as f:
                json.dump(history_to_save, f, indent=2)
        except Exception as e:
            logger.error(f"Errore salvataggio pattern history: {e}")
    
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
    
    def identify_market_trend_category(self, title, subreddit):
        """Identifica categoria di trend di mercato"""
        title_lower = title.lower()
        subreddit_lower = subreddit.lower()
        
        pattern_scores = {}
        
        for pattern, keywords in self.market_pattern_keywords.items():
            score = 0
            
            for keyword in keywords['primary']:
                if keyword in title_lower or keyword in subreddit_lower:
                    score += keywords['score']['primary']
            
            for keyword in keywords['secondary']:
                if keyword in title_lower or keyword in subreddit_lower:
                    score += keywords['score']['secondary']
            
            if score > 0:
                pattern_scores[pattern] = score
        
        if pattern_scores:
            best_pattern = max(pattern_scores.items(), key=lambda x: x[1])
            return best_pattern[0], best_pattern[1]
        
        return 'consumer_behavior', 0
    
    def analyze_market_sentiment(self, title):
        """Analisi sentiment specifica per mercato"""
        text = title.lower()
        sentiment_data = {}
        total_score = 0
        
        for category, levels in self.market_sentiment_keywords.items():
            category_score = 0
            
            for level, keywords in levels.items():
                if level == 'score':
                    continue
                
                for keyword in keywords:
                    if keyword in text:
                        category_score += levels['score'][level]
            
            sentiment_data[category] = category_score
            total_score += category_score
        
        if total_score >= 20:
            intensity = 'extreme'
        elif total_score >= 15:
            intensity = 'high'
        elif total_score >= 10:
            intensity = 'medium'
        else:
            intensity = 'low'
        
        return {
            'total_score': min(total_score, 100),
            'categories': sentiment_data,
            'intensity': intensity,
            'weighted_score': total_score
        }
    
    def get_weight(self, key):
        """Ottieni peso corrente"""
        if key in self.weights:
            return self.weights[key]['weight']
        return 1.0
    
    def calculate_adaptive_learning_rate(self, pattern):
        """Calcola learning rate adaptivo"""
        base_rate = self.learning_config['base_learning_rate']
        
        if pattern not in self.weights:
            return base_rate
        
        pattern_data = self.weights[pattern]
        samples = pattern_data.get('samples', 0)
        success_rate = pattern_data.get('success_rate', 0.5)
        
        sample_factor = 1 / (1 + samples * 0.1)
        performance_factor = 2 - success_rate if success_rate < 0.5 else 1
        
        adaptive_rate = base_rate * sample_factor * performance_factor
        return min(adaptive_rate, base_rate * 2)
    
    def predict_trend_growth(self, post, subreddit, minutes_ago):
        """Predizione crescita trend di mercato"""
        
        # 1. Identifica pattern di mercato
        pattern_category, pattern_score = self.identify_market_trend_category(post.title, subreddit)
        pattern_multiplier = self.get_weight(pattern_category)
        
        if pattern_score >= 6:
            pattern_multiplier *= 1.3
        elif pattern_score >= 4:
            pattern_multiplier *= 1.2
        
        # 2. Analisi sentiment di mercato
        sentiment = self.analyze_market_sentiment(post.title)
        sentiment_multiplier = 1 + (sentiment['weighted_score'] / 100)
        
        # 3. Velocità di crescita
        velocity = post.score / max(minutes_ago, 1)
        explosive_threshold = self.get_weight('trend_explosive_threshold')
        fast_threshold = self.get_weight('trend_fast_threshold')
        steady_threshold = self.get_weight('trend_steady_threshold')
        
        if velocity >= explosive_threshold:
            velocity_score = 100
        elif velocity >= fast_threshold:
            velocity_score = 80
        elif velocity >= steady_threshold:
            velocity_score = 50
        else:
            velocity_score = min(velocity * 3, 40)
        
        velocity_multiplier = 1 + (velocity_score / 100)
        
        # 4. Engagement analysis
        if post.score > 0:
            comment_ratio = post.num_comments / post.score
            optimal_ratio = 0.1  # 10% ottimale per discussioni di mercato
            ratio_penalty = abs(comment_ratio - optimal_ratio) / optimal_ratio
            engagement_multiplier = max(1 + (comment_ratio * 2) - ratio_penalty, 0.5)
        else:
            engagement_multiplier = 1.0
        
        # 5. Time decay per trend di mercato
        if minutes_ago > 360:  # >6 ore
            time_multiplier = 0.6
        elif minutes_ago > 180:  # >3 ore
            time_multiplier = 0.8
        else:
            time_multiplier = 1.2  # Bonus per trend recenti
        
        # 6. Calcolo probabilità trend
        base_probability = 0.2
        
        final_probability = (
            base_probability * 
            pattern_multiplier * 
            sentiment_multiplier * 
            velocity_multiplier * 
            engagement_multiplier * 
            time_multiplier
        )
        
        final_probability = max(0.01, min(final_probability, 0.95))
        
        # Calcola confidence
        pattern_data = self.weights.get(pattern_category, {'samples': 0, 'success_rate': 0.5})
        confidence_raw = min(pattern_data['samples'] / 10, 1.0)
        success_rate = pattern_data['success_rate']
        
        confidence = (confidence_raw * success_rate + (1 - confidence_raw) * 0.5) * 100
        
        # Predici crescita trend
        growth_factor = final_probability * 20
        predicted_engagement = int(post.score * (1 + growth_factor))
        
        # Tempo di picco per trend di mercato
        peak_hours = self._calculate_trend_peak_time(pattern_category, velocity, sentiment['intensity'])
        
        return {
            'trend_probability': round(final_probability * 100, 1),
            'confidence': round(confidence, 1),
            'confidence_level': 'high' if confidence > 70 else 'medium' if confidence > 40 else 'low',
            'predicted_peak_hours': peak_hours,
            'predicted_engagement': predicted_engagement,
            'trend_category': pattern_category,
            'pattern_score': pattern_score,
            'pattern_samples': pattern_data['samples'],
            'pattern_success_rate': round(pattern_data['success_rate'] * 100, 1),
            'market_sentiment': sentiment,
            'velocity_score': velocity_score,
            'velocity_raw': round(velocity, 2),
            'pattern_multiplier': round(pattern_multiplier, 2),
            'reasoning': self.generate_trend_reasoning(
                pattern_category, pattern_score, sentiment, velocity_score, 
                final_probability * 100, confidence
            )
        }
    
    def _calculate_trend_peak_time(self, pattern, velocity, intensity):
        """Calcola tempo di picco per trend di mercato"""
        base_times = {
            'shopping_trends': 4,
            'consumer_behavior': 6,
            'product_trends': 8,
            'brand_sentiment': 12,
            'pricing_trends': 3,
            'demand_analysis': 10,
            'purchase_intent': 2
        }
        
        base = base_times.get(pattern, 6)
        
        if velocity > 60:
            base *= 0.7
        elif velocity > 30:
            base *= 0.9
        
        if intensity == 'extreme':
            base *= 0.8
        elif intensity == 'high':
            base *= 0.9
        
        return max(base, 1)
    
    def generate_trend_reasoning(self, pattern, pattern_score, sentiment, velocity, probability, confidence):
        """Genera spiegazione per trend"""
        reasons = []
        
        if pattern_score >= 6:
            reasons.append(f"Trend {pattern} molto forte (score: {pattern_score})")
        elif pattern_score >= 3:
            reasons.append(f"Trend {pattern} rilevato (score: {pattern_score})")
        
        if sentiment['intensity'] in ['extreme', 'high']:
            reasons.append(f"Alta intensità sentiment ({sentiment['intensity']})")
        
        if velocity >= 60:
            reasons.append("Crescita trend esplosiva")
        elif velocity >= 30:
            reasons.append("Velocità trend elevata")
        
        if confidence > 70:
            reasons.append(f"Alta affidabilità ({confidence:.0f}%)")
        elif confidence < 30:
            reasons.append(f"Analisi esplorativa ({confidence:.0f}%)")
        
        return " • ".join(reasons[:4])
    
    def track_trend_prediction(self, post_id, prediction_data, post_score, subreddit, title):
        """Traccia predizione trend"""
        current_time = datetime.now()
        
        self.active_predictions[post_id] = {
            'timestamp': current_time.isoformat(),
            'prediction': prediction_data,
            'original_score': post_score,
            'subreddit': subreddit,
            'title': title,
            'trend_category': prediction_data['trend_category'],
            'pattern_score': prediction_data['pattern_score'],
            'predicted_probability': prediction_data['trend_probability'],
            'predicted_engagement': prediction_data['predicted_engagement'],
            'confidence': prediction_data['confidence'],
            'sentiment_intensity': prediction_data['market_sentiment']['intensity'],
            'velocity_score': prediction_data['velocity_score'],
            'hour_of_day': current_time.hour
        }
        self.save_predictions()
    
    async def check_and_learn_trends(self, reddit):
        """Sistema di learning per trend"""
        if not self.active_predictions:
            return
        
        current_time = datetime.now()
        learned_count = 0
        
        for post_id, prediction_data in list(self.active_predictions.items()):
            try:
                prediction_time = datetime.fromisoformat(prediction_data['timestamp'])
                hours_passed = (current_time - prediction_time).total_seconds() / 3600
                
                if hours_passed >= 6:  # 6 ore per trend di mercato
                    estimated_trend = self.estimate_trend_outcome(prediction_data, hours_passed)
                    
                    self.apply_trend_learning(prediction_data, estimated_trend)
                    self.update_performance_stats(prediction_data, estimated_trend)
                    
                    del self.active_predictions[post_id]
                    learned_count += 1
                    
            except Exception as e:
                logger.warning(f"Errore trend learning per {post_id}: {e}")
                if post_id in self.active_predictions:
                    del self.active_predictions[post_id]
        
        if learned_count > 0:
            logger.info(f"Trend Learning: Aggiornati pesi da {learned_count} predizioni")
            self.save_weights()
            self.save_predictions()
            self.save_performance_stats()
            self.save_pattern_history()
            
            accuracy = self.calculate_current_accuracy()
            logger.info(f"📊 Accuracy trend attuale: {accuracy:.1f}%")
    
    def estimate_trend_outcome(self, prediction_data, hours_passed):
        """Stima se il trend si è materializzato"""
        original_score = prediction_data['original_score']
        predicted_prob = prediction_data['predicted_probability']
        predicted_engagement = prediction_data['predicted_engagement']
        confidence = prediction_data['confidence']
        pattern = prediction_data['trend_category']
        
        # Moltiplicatori per diversi tipi di trend
        trend_multipliers = {
            'shopping_trends': 8,
            'consumer_behavior': 6,
            'product_trends': 10,
            'brand_sentiment': 12,
            'pricing_trends': 15,
            'demand_analysis': 7,
            'purchase_intent': 20
        }
        
        multiplier = trend_multipliers.get(pattern, 5)
        estimated_engagement = original_score * multiplier
        
        # Criteri per trend significativo
        trend_threshold = 1000
        strong_trend_threshold = 2500
        
        if predicted_prob > 65 and confidence > 60:
            return estimated_engagement > trend_threshold
        elif predicted_prob > 45 and confidence > 40:
            return estimated_engagement > strong_trend_threshold
        elif predicted_prob > 30:
            return estimated_engagement > trend_threshold * 1.5
        else:
            return False
    
    def apply_trend_learning(self, prediction_data, actual_trend):
        """Applica learning per trend"""
        pattern = prediction_data['trend_category']
        predicted_prob = prediction_data['predicted_probability']
        confidence = prediction_data['confidence']
        pattern_score = prediction_data['pattern_score']
        
        was_correct = (predicted_prob > 50 and actual_trend) or (predicted_prob <= 50 and not actual_trend)
        
        learning_rate = self.calculate_adaptive_learning_rate(pattern)
        
        if pattern in self.weights:
            pattern_data = self.weights[pattern]
            
            if was_correct:
                adjustment = 1 + (learning_rate * (confidence / 100))
                if pattern_score >= 6:
                    adjustment *= 1.1
            else:
                adjustment = 1 - (learning_rate * (confidence / 100))
                if predicted_prob > 75 and not actual_trend:
                    adjustment *= 0.9
            
            momentum = pattern_data['momentum'] * self.learning_config['momentum_factor']
            new_momentum = (adjustment - 1) * learning_rate + momentum
            
            old_weight = pattern_data['weight']
            pattern_data['weight'] = max(old_weight * adjustment + new_momentum, 0.1)
            pattern_data['momentum'] = new_momentum
            pattern_data['samples'] += 1
            
            current_success_rate = pattern_data['success_rate']
            if pattern_data['samples'] == 1:
                pattern_data['success_rate'] = 1.0 if was_correct else 0.0
            else:
                decay = 0.9
                pattern_data['success_rate'] = (current_success_rate * decay + 
                                              (1.0 if was_correct else 0.0) * (1 - decay))
            
            logger.debug(f"Trend {pattern}: {old_weight:.3f} -> {pattern_data['weight']:.3f}")
    
    def update_performance_stats(self, prediction_data, actual_trend):
        """Aggiorna statistiche performance"""
        predicted_prob = prediction_data['predicted_probability']
        pattern = prediction_data['trend_category']
        subreddit = prediction_data['subreddit']
        hour = prediction_data['hour_of_day']
        
        was_correct = (predicted_prob > 50 and actual_trend) or (predicted_prob <= 50 and not actual_trend)
        
        self.performance_stats['total_predictions'] += 1
        if was_correct:
            self.performance_stats['correct_predictions'] += 1
        
        if predicted_prob > 50 and not actual_trend:
            self.performance_stats['false_positives'] += 1
        elif predicted_prob <= 50 and actual_trend:
            self.performance_stats['false_negatives'] += 1
        
        self.performance_stats['accuracy_trend'].append(1 if was_correct else 0)
        
        if pattern not in self.performance_stats['pattern_performance']:
            self.performance_stats['pattern_performance'][pattern] = {
                'total': 0, 'correct': 0, 'accuracy': 0.0
            }
        
        pattern_stats = self.performance_stats['pattern_performance'][pattern]
        pattern_stats['total'] += 1
        if was_correct:
            pattern_stats['correct'] += 1
        pattern_stats['accuracy'] = pattern_stats['correct'] / pattern_stats['total']
    
    def calculate_current_accuracy(self):
        """Calcola accuracy attuale"""
        if self.performance_stats['total_predictions'] == 0:
            return 50.0
        
        return (self.performance_stats['correct_predictions'] / 
                self.performance_stats['total_predictions']) * 100
    
    def get_trend_insights(self):
        """Ottieni insights sui trend"""
        if not self.performance_stats['accuracy_trend']:
            return {
                'overall_accuracy': 50.0,
                'recent_accuracy': 50.0,
                'total_predictions': 0,
                'best_trends': [],
                'trend': 'stable'
            }
        
        recent_accuracy = sum(list(self.performance_stats['accuracy_trend'])[-10:]) / min(10, len(self.performance_stats['accuracy_trend'])) * 100
        overall_accuracy = self.calculate_current_accuracy()
        
        best_trends = []
        for pattern, stats in self.performance_stats['pattern_performance'].items():
            if stats['total'] >= 3:
                best_trends.append((pattern, stats['accuracy']))
        
        best_trends.sort(key=lambda x: x[1], reverse=True)
        
        insights = {
            'overall_accuracy': overall_accuracy,
            'recent_accuracy': recent_accuracy,
            'total_predictions': self.performance_stats['total_predictions'],
            'best_trends': best_trends[:5],
            'trend': 'improving' if recent_accuracy > overall_accuracy else 'stable' if abs(recent_accuracy - overall_accuracy) < 5 else 'declining'
        }
        
        return insights

# ===== MARKET TREND ANALYZER =====
class MarketTrendAnalyzer:
    def __init__(self):
        # Credenziali
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        
        if not all([self.reddit_client_id, self.reddit_client_secret, self.telegram_token]):
            raise ValueError("Variabili d'ambiente mancanti!")
        
        # Market Trend AI
        self.market_ai = MarketTrendAI()
        
        # State management
        self.active_chats = set()
        self.reddit = None
        self.analyzed_posts = set()
        
        # Subreddit ottimizzati per trend di mercato (evitando quelli problematici)
        self.market_subreddits = [
            'business', 'economics', 'investing', 'stocks', 'wallstreetbets',
            'personalfinance', 'finance', 'smallbusiness', 'entrepreneur',
            'marketing', 'advertising', 'sales', 'customerservice',
            'productmanagement', 'supplychain', 'logistics',
            'tech', 'technology', 'gadgets', 'android', 'apple',
            'ecommerce', 'onlinemarketing', 'digitalmarketing',
            'consumer', 'consumers', 'shopping', 'deals', 'discounts',
            'productreviews', 'reviews', 'amazon', 'ebay',
            'brand', 'branding', 'customerexperience',
            'marketresearch', 'dataisbeautiful', 'datascience',
            'startups', 'venturecapital', 'angelinvesting'
        ]
        
    async def initialize(self):
        """Inizializza Reddit connection"""
        try:
            self.reddit = asyncpraw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent='MarketTrendAnalyzer/2.0 (by u/YourUsername)'
            )
            logger.info("✅ Reddit connesso con Market Trend AI")
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
                                        logger.info(f"📱 Nuova chat: {chat_id}")
                            
                            if data['result']:
                                last_update_id = data['result'][-1]['update_id']
                                clear_url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates?offset={last_update_id + 1}"
                                await session.get(clear_url)
                            
                            if new_chats > 0:
                                logger.info(f"📊 {new_chats} nuove chat. Totale: {len(self.active_chats)}")
                        
                        return True
                    return False
                
        except Exception as e:
            logger.error(f"Errore chat: {e}")
            return False
    
    async def analyze_market_trends(self):
        """Analizza trend di mercato e comportamenti consumatori"""
        try:
            trend_posts = []
            current_time = datetime.now()
            
            for subreddit_name in self.market_subreddits:
                try:
                    subreddit = await self.reddit.subreddit(subreddit_name)
                    
                    count = 0
                    async for post in subreddit.hot(limit=15):  # Ridotto limite per evitare 403
                        count += 1
                        
                        post_time = datetime.fromtimestamp(post.created_utc)
                        minutes_ago = (current_time - post_time).total_seconds() / 60
                        
                        # Filtri più selettivi per trend di mercato
                        if (minutes_ago <= 480 and post.score >= 10 and 
                            post.id not in self.analyzed_posts):
                            
                            # Market Trend AI Prediction
                            trend_prediction = self.market_ai.predict_trend_growth(
                                post, subreddit_name, minutes_ago
                            )
                            
                            # Filtra per trend significativi
                            if (trend_prediction['trend_probability'] >= 40 and 
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
                                    'trend_prediction': trend_prediction
                                })
                                
                                # Traccia per learning
                                self.market_ai.track_trend_prediction(
                                    post.id, trend_prediction, post.score, subreddit_name, post.title
                                )
                        
                        if count >= 15:  # Limite per evitare rate limiting
                            break
                            
                except Exception as e:
                    if "403" in str(e):
                        logger.warning(f"⏸️  Subreddit {subreddit_name} non accessibile (403). Saltato.")
                    else:
                        logger.warning(f"Errore {subreddit_name}: {e}")
                    continue
                await asyncio.sleep(1)  # Rate limiting
            
            trend_posts.sort(key=lambda x: x['trend_prediction']['trend_probability'], reverse=True)
            
            logger.info(f"📈 Market Trend AI: {len(trend_posts)} trend rilevati")
            
            return {
                'trend_posts': trend_posts[:5],  # Massimo 5 trend per alert
                'timestamp': current_time,
                'trend_insights': self.market_ai.get_trend_insights()
            }
            
        except Exception as e:
            logger.error(f"Errore analisi trend: {e}")
            return None
    
    def format_trend_alert(self, data):
        """Formatta alert per trend di mercato"""
        if not data or not data['trend_posts']:
            return "📊 Nessun trend significativo rilevato nelle ultime ore."
        
        timestamp = data['timestamp'].strftime("%H:%M - %d/%m/%Y")
        posts = data['trend_posts']
        insights = data.get('trend_insights', {})
        
        message = f"📈 MARKET TREND ANALYZER 📈\n"
        message += f"⏰ {timestamp} | 🧠 AI Consumer Behavior Analysis\n"
        
        if insights.get('total_predictions', 0) > 0:
            accuracy = insights.get('overall_accuracy', 50)
            recent_accuracy = insights.get('recent_accuracy', 50)
            trend = insights.get('trend', 'stable')
            trend_emoji = "📈" if trend == 'improving' else "📉" if trend == 'declining' else "➡️"
            
            message += f"📊 AI Accuracy: {accuracy:.1f}% | Recente: {recent_accuracy:.1f}% {trend_emoji}\n"
        
        message += f"🔥 {len(posts)} TREND EMERGENTI RILEVATI:\n"
        
        for i, post in enumerate(posts, 1):
            title = post['title'][:60] + "..." if len(post['title']) > 60 else post['title']
            title = title.replace('[', '').replace(']', '').replace('*', '')
            
            trend = post['trend_prediction']
            
            # Emoji basate su probabilità trend
            if trend['trend_probability'] >= 70 and trend['confidence'] > 65:
                emoji = "🚀🔥"
                level = "TREND ESPLOSIVO"
            elif trend['trend_probability'] >= 55:
                emoji = "⚡📈"
                level = "TREND FORTE"
            else:
                emoji = "📊🎯"
                level = "TREND EMERGENTE"
            
            message += f"\n{emoji} {i}. {title}\n"
            message += f"📊 Engagement: {post['score']} upvotes ({post['upvotes_per_min']}/min) | 💬 {post['comments']} comments\n"
            
            # Dettagli trend AI
            message += f"🧠 Probabilità Trend: {trend['trend_probability']}% | Affidabilità: {trend['confidence']}% ({trend['confidence_level']})\n"
            message += f"🎯 Categoria: {trend['trend_category'].replace('_', ' ').title()} "
            
            if trend.get('pattern_samples', 0) > 0:
                message += f"| Success Rate: {trend.get('pattern_success_rate', 50)}%\n"
            else:
                message += f"| Nuovo Pattern\n"
            
            message += f"📈 Previsione → {trend['predicted_engagement']:,} engagement in {trend['predicted_peak_hours']:.1f}h\n"
            message += f"⚡ Velocità Crescita: {trend['velocity_raw']}/min | Sentiment: {trend['market_sentiment']['intensity']}\n"
            
            if trend.get('reasoning'):
                message += f"💡 {trend['reasoning']}\n"
            
            message += f"📍 r/{post['subreddit']} | ⏱️ {post['minutes_ago']} min fa\n"
            message += f"🔗 {post['url']}\n"
        
        # Insights summary
        if insights.get('best_trends'):
            top_trend = insights['best_trends'][0]
            message += f"\n🏆 Trend Più Affidabile: {top_trend[0].replace('_', ' ').title()} ({top_trend[1]*100:.0f}% accuracy)\n"
        
        total_pred = insights.get('total_predictions', 0)
        message += f"📚 Trend Analizzati: {total_pred} | Modalità: Monitoraggio Continuo\n"
        message += f"⚡ Market Trend Analyzer v2.0 | Consumer Behavior AI"
        
        return message
    
    async def send_trend_alert(self, message):
        """Invia alert trend a Telegram"""
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
    
    async def run_trend_analyzer(self):
        """Main loop per analisi trend"""
        logger.info("📈 Avvio Market Trend Analyzer...")
        logger.info("🧠 AI Consumer Behavior Analysis con Trend Prediction")
        logger.info("⏰ Scansione ogni 15 minuti + Learning ogni ora")
        
        if not await self.initialize():
            return
        
        logger.info("✅ Market Trend Analyzer operativo!")
        
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                
                await self.get_active_chats()
                
                # Trend Learning check ogni 4 cicli (ogni ora)
                if cycle_count % 4 == 0:
                    logger.info("🧠 Trend Learning check...")
                    await self.market_ai.check_and_learn_trends(self.reddit)
                
                # Analisi trend di mercato
                logger.info("🔍 Analisi trend di mercato...")
                trend_data = await self.analyze_market_trends()
                
                if trend_data and trend_data['trend_posts']:
                    new_trends = [p for p in trend_data['trend_posts'] if p['id'] not in self.analyzed_posts]
                    
                    if new_trends and self.active_chats:
                        for post in new_trends:
                            self.analyzed_posts.add(post['id'])
                        
                        trend_data['trend_posts'] = new_trends
                        message = self.format_trend_alert(trend_data)
                        success = await self.send_trend_alert(message)
                        
                        if success:
                            logger.info(f"🔥 {len(new_trends)} trend alerts inviati!")
                            
                            for post in new_trends:
                                trend = post['trend_prediction']
                                logger.info(
                                    f"  📈 {trend['trend_category']}: {trend['trend_probability']}% "
                                    f"(conf: {trend['confidence']}%) → {trend['predicted_engagement']} | "
                                    f"{post['title'][:35]}..."
                                )
                
                # Pulizia cache
                if len(self.analyzed_posts) > 1000:
                    self.analyzed_posts.clear()
                
                # Stats logging
                if cycle_count % 8 == 0:
                    insights = self.market_ai.get_trend_insights()
                    logger.info(f"📊 Trend AI Stats: {insights.get('total_predictions', 0)} analisi, "
                              f"{insights.get('overall_accuracy', 0):.1f}% accuracy")
                
                await asyncio.sleep(900)  # 15 minuti
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Errore main loop: {e}")
                await asyncio.sleep(300)  # 5 minuti di pausa in caso di errore
        
        if self.reddit:
            await self.reddit.close()

async def main():
    """Main function"""
    try:
        analyzer = MarketTrendAnalyzer()
        await analyzer.run_trend_analyzer()
    except Exception as e:
        logger.error(f"Errore critico: {e}")

if __name__ == "__main__":
    logger.info("📈 Market Trend Analyzer v2.0 - Consumer Behavior AI")
    asyncio.run(main())
