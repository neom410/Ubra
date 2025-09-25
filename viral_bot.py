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
        self.weights_file = 'market_trend_weights.json'
        self.predictions_file = 'market_predictions_track.json'
        self.performance_stats_file = 'market_performance_stats.json'
        self.trend_history_file = 'trend_history.json'
        
        # Performance tracking per migliorare learning
        self.performance_stats = self.load_performance_stats()
        self.trend_success_history = self.load_trend_history()
        
        # Learning parameters migliorati
        self.learning_config = {
            'base_learning_rate': 0.03,
            'adaptive_learning_rate': True,
            'momentum_factor': 0.1,
            'confidence_threshold': 0.7,
            'min_samples_for_trend': 3,
            'trend_decay_rate': 0.99,
            'success_boost_adaptive': True,
            'failure_penalty_adaptive': True
        }
        
        # Pesi specifici per trend di mercato e consumatori
        self.default_weights = {
            # Trend categories con momentum
            'consumer_sentiment': {'weight': 2.8, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'market_volatility': {'weight': 3.2, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'product_trends': {'weight': 2.5, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'brand_sentiment': {'weight': 2.3, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'economic_indicators': {'weight': 2.6, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'shopping_behavior': {'weight': 2.4, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'technology_adoption': {'weight': 2.7, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'investment_trends': {'weight': 3.0, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'sustainability_trends': {'weight': 2.2, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'health_wellness': {'weight': 2.1, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            
            # Market sentiment multipliers
            'bullish_sentiment': {'weight': 1.8, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'bearish_sentiment': {'weight': 2.0, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'uncertainty_weight': {'weight': 1.6, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'adoption_velocity': {'weight': 2.2, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'social_proof_weight': {'weight': 1.9, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            
            # Velocity thresholds adaptive
            'trend_explosive_threshold': {'weight': 35.0, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'trend_fast_threshold': {'weight': 15.0, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'trend_steady_threshold': {'weight': 6.0, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
        }
        
        # Carica pesi salvati o usa default
        self.weights = self.load_weights()
        
        # Tracking predizioni per feedback migliorato
        self.active_predictions = self.load_predictions()
        
        # Trend keywords per analisi di mercato
        self.trend_categories = {
            'consumer_sentiment': {
                'primary': ['buy', 'purchase', 'shopping', 'consumer', 'spending', 'demand'],
                'secondary': ['afford', 'price', 'cost', 'budget', 'expensive', 'cheap'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'market_volatility': {
                'primary': ['volatility', 'crash', 'drop', 'plummet', 'surge', 'rally'],
                'secondary': ['uncertainty', 'risk', 'fluctuation', 'swing', 'turbulence'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'product_trends': {
                'primary': ['product', 'launch', 'release', 'new', 'innovation', 'feature'],
                'secondary': ['upgrade', 'version', 'model', 'design', 'improvement'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'brand_sentiment': {
                'primary': ['brand', 'company', 'corporation', 'reputation', 'trust'],
                'secondary': ['loyalty', 'image', 'perception', 'quality', 'service'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'economic_indicators': {
                'primary': ['economy', 'gdp', 'inflation', 'recession', 'growth', 'employment'],
                'secondary': ['rate', 'percentage', 'index', 'indicator', 'statistic'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'shopping_behavior': {
                'primary': ['shop', 'store', 'online', 'ecommerce', 'cart', 'checkout'],
                'secondary': ['browse', 'compare', 'review', 'rating', 'recommendation'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'technology_adoption': {
                'primary': ['adopt', 'technology', 'digital', 'ai', 'automation', 'smart'],
                'secondary': ['integration', 'implementation', 'deployment', 'usage'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'investment_trends': {
                'primary': ['invest', 'stock', 'portfolio', 'return', 'dividend', 'yield'],
                'secondary': ['asset', 'security', 'equity', 'bond', 'fund'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'sustainability_trends': {
                'primary': ['sustainable', 'green', 'eco', 'environment', 'climate'],
                'secondary': ['renewable', 'carbon', 'emission', 'recycle', 'ethical'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'health_wellness': {
                'primary': ['health', 'wellness', 'fitness', 'nutrition', 'diet', 'exercise'],
                'secondary': ['wellbeing', 'lifestyle', 'organic', 'natural', 'supplement'],
                'score': {'primary': 3, 'secondary': 2}
            }
        }
        
        # Market sentiment keywords
        self.market_sentiment_keywords = {
            'bullish_sentiment': {
                'strong': ['bullish', 'optimistic', 'positive', 'growth', 'profit', 'gain'],
                'moderate': ['recovery', 'improve', 'expand', 'opportunity', 'potential'],
                'score': {'strong': 4, 'moderate': 2}
            },
            'bearish_sentiment': {
                'strong': ['bearish', 'pessimistic', 'negative', 'loss', 'decline', 'drop'],
                'moderate': ['concern', 'worry', 'risk', 'caution', 'uncertain'],
                'score': {'strong': 4, 'moderate': 2}
            },
            'uncertainty': {
                'strong': ['uncertain', 'volatile', 'unpredictable', 'turbulent'],
                'moderate': ['mixed', 'unclear', 'ambiguous', 'debate', 'discussion'],
                'score': {'strong': 4, 'moderate': 2}
            },
            'adoption_velocity': {
                'strong': ['adopt', 'embrace', 'adoption', 'popular', 'mainstream'],
                'moderate': ['growing', 'increasing', 'rising', 'expanding'],
                'score': {'strong': 4, 'moderate': 2}
            },
            'social_proof': {
                'strong': ['popular', 'trending', 'viral', 'bestseller', 'recommended'],
                'moderate': ['review', 'rating', 'feedback', 'testimonial', 'endorsement'],
                'score': {'strong': 4, 'moderate': 2}
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
            'category_performance': {},
            'hourly_accuracy': {},
            'subreddit_performance': {}
        }
    
    def load_trend_history(self):
        """Carica storico performance trend"""
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
    
    def save_trend_history(self):
        """Salva storico trend"""
        try:
            history_to_save = {}
            for trend, data in self.trend_success_history.items():
                history_to_save[trend] = data.copy()
                if 'recent_performance' in data:
                    history_to_save[trend]['recent_performance'] = list(data['recent_performance'])
            
            with open(self.trend_history_file, 'w') as f:
                json.dump(history_to_save, f, indent=2)
        except Exception as e:
            logger.error(f"Errore salvataggio trend history: {e}")
    
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
    
    def identify_trend_category_enhanced(self, title, subreddit):
        """Identifica categoria trend con scoring migliorato"""
        title_lower = title.lower()
        subreddit_lower = subreddit.lower()
        
        trend_scores = {}
        
        for category, keywords in self.trend_categories.items():
            score = 0
            
            for keyword in keywords['primary']:
                if keyword in title_lower or keyword in subreddit_lower:
                    score += keywords['score']['primary']
            
            for keyword in keywords['secondary']:
                if keyword in title_lower or keyword in subreddit_lower:
                    score += keywords['score']['secondary']
            
            if score > 0:
                trend_scores[category] = score
        
        if trend_scores:
            best_trend = max(trend_scores.items(), key=lambda x: x[1])
            return best_trend[0], best_trend[1]
        
        return 'general', 0
    
    def analyze_market_sentiment_enhanced(self, title):
        """Analisi sentiment di mercato con scoring migliorato"""
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
            total_score += category_score * self.get_weight(f'{category}_weight')
        
        # Determina intensità sentiment
        if total_score >= 25:
            intensity = 'strong'
        elif total_score >= 15:
            intensity = 'moderate'
        elif total_score >= 8:
            intensity = 'weak'
        else:
            intensity = 'neutral'
        
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
    
    def calculate_adaptive_learning_rate(self, category):
        """Calcola learning rate adaptivo per categoria"""
        base_rate = self.learning_config['base_learning_rate']
        
        if category not in self.weights:
            return base_rate
        
        category_data = self.weights[category]
        samples = category_data.get('samples', 0)
        success_rate = category_data.get('success_rate', 0.5)
        
        sample_factor = 1 / (1 + samples * 0.1)
        performance_factor = 2 - success_rate if success_rate < 0.5 else 1
        
        adaptive_rate = base_rate * sample_factor * performance_factor
        return min(adaptive_rate, base_rate * 2)
    
    def predict_trend_momentum_enhanced(self, post, subreddit, minutes_ago):
        """CORE: Predizione momentum trend di mercato e consumatori"""
        
        # 1. Identifica categoria trend
        trend_category, trend_score = self.identify_trend_category_enhanced(post.title, subreddit)
        trend_multiplier = self.get_weight(trend_category)
        
        if trend_score >= 6:
            trend_multiplier *= 1.3
        elif trend_score >= 4:
            trend_multiplier *= 1.2
        
        # 2. Analisi sentiment di mercato
        sentiment = self.analyze_market_sentiment_enhanced(post.title)
        sentiment_multiplier = 1 + (sentiment['weighted_score'] / 100)
        
        # 3. Velocità di adozione/diffusione
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
        
        # 4. Engagement analysis per trend
        if post.score > 0:
            comment_ratio = post.num_comments / post.score
            # Per trend di mercato, ratio più alto indica discussione approfondita
            engagement_multiplier = 1 + min(comment_ratio * 3, 2.0)  # Max 3x
        else:
            engagement_multiplier = 1.0
        
        # 5. Time decay per trend (più lento delle news)
        if minutes_ago > 360:  # >6 ore
            time_multiplier = 0.8
        elif minutes_ago > 180:  # >3 ore
            time_multiplier = 0.9
        else:
            time_multiplier = 1.2  # Bonus per trend freschi
        
        # 6. Calcolo probabilità trend
        base_probability = 0.20  # Più conservativo per trend di mercato
        
        final_probability = (
            base_probability * 
            trend_multiplier * 
            sentiment_multiplier * 
            velocity_multiplier * 
            engagement_multiplier * 
            time_multiplier
        )
        
        final_probability = max(0.01, min(final_probability, 0.95))
        
        # Calcola confidence
        trend_data = self.weights.get(trend_category, {'samples': 0, 'success_rate': 0.5})
        confidence_raw = min(trend_data['samples'] / 10, 1.0)
        success_rate = trend_data['success_rate']
        
        confidence = (confidence_raw * success_rate + (1 - confidence_raw) * 0.5) * 100
        
        # Predici crescita trend
        growth_factor = final_probability * 20  # Più conservativo
        volatility = 1 + (sentiment['weighted_score'] / 150)
        predicted_momentum = int(post.score * (1 + growth_factor) * volatility)
        
        # Timeline trend
        trend_timeline = self._calculate_trend_timeline(trend_category, velocity, sentiment['intensity'])
        
        return {
            'trend_probability': round(final_probability * 100, 1),
            'confidence': round(confidence, 1),
            'confidence_level': 'high' if confidence > 70 else 'medium' if confidence > 40 else 'low',
            'predicted_timeline_hours': trend_timeline,
            'predicted_momentum': predicted_momentum,
            'trend_category': trend_category,
            'trend_score': trend_score,
            'trend_samples': trend_data['samples'],
            'trend_success_rate': round(trend_data['success_rate'] * 100, 1),
            'market_sentiment': sentiment,
            'velocity_score': velocity_score,
            'velocity_raw': round(velocity, 2),
            'trend_multiplier': round(trend_multiplier, 2),
            'market_impact': self.assess_market_impact(trend_category, sentiment, velocity_score),
            'reasoning': self.generate_trend_reasoning(
                trend_category, trend_score, sentiment, velocity_score, 
                final_probability * 100, confidence
            )
        }
    
    def _calculate_trend_timeline(self, category, velocity, intensity):
        """Calcola timeline del trend"""
        base_times = {
            'market_volatility': 2,
            'investment_trends': 4,
            'consumer_sentiment': 6,
            'product_trends': 8,
            'technology_adoption': 12,
            'sustainability_trends': 24,
            'health_wellness': 18
        }
        
        base = base_times.get(category, 8)
        
        if velocity > 60:
            base *= 0.6  # Trend veloci hanno timeline più brevi
        elif velocity > 30:
            base *= 0.8
        
        if intensity == 'strong':
            base *= 0.7
        elif intensity == 'moderate':
            base *= 0.9
        
        return max(base, 1)
    
    def assess_market_impact(self, category, sentiment, velocity):
        """Valuta impatto sul mercato"""
        impact_score = 0
        
        # Base impact per categoria
        category_impact = {
            'market_volatility': 90,
            'investment_trends': 80,
            'consumer_sentiment': 70,
            'economic_indicators': 85,
            'brand_sentiment': 60,
            'product_trends': 65,
            'technology_adoption': 75,
            'shopping_behavior': 55,
            'sustainability_trends': 50,
            'health_wellness': 45
        }
        
        impact_score += category_impact.get(category, 40)
        
        # Modifica per sentiment
        if sentiment['intensity'] == 'strong':
            impact_score += 20
        elif sentiment['intensity'] == 'moderate':
            impact_score += 10
        
        # Modifica per velocità
        if velocity > 80:
            impact_score += 15
        elif velocity > 50:
            impact_score += 8
        
        impact_level = 'high' if impact_score >= 80 else 'medium' if impact_score >= 60 else 'low'
        
        return {
            'score': min(impact_score, 100),
            'level': impact_level,
            'description': self.get_impact_description(impact_level, category)
        }
    
    def get_impact_description(self, impact_level, category):
        """Genera descrizione impatto"""
        descriptions = {
            'high': {
                'market_volatility': 'Alto impatto su volatilità mercati',
                'investment_trends': 'Forte influenza su decisioni investimenti',
                'consumer_sentiment': 'Significativo cambiamento comportamenti acquisto',
                'economic_indicators': 'Impatto rilevante su indicatori economici'
            },
            'medium': {
                'market_volatility': 'Moderato impatto su trend mercato',
                'investment_trends': 'Influenza media su strategie investimento',
                'consumer_sentiment': 'Cambiamento graduale preferenze consumatori'
            },
            'low': {
                'market_volatility': 'Impatto limitato sui mercati',
                'investment_trends': 'Minima influenza su investimenti',
                'consumer_sentiment': 'Lieve modifica comportamenti acquisto'
            }
        }
        
        return descriptions.get(impact_level, {}).get(category, f'Impatto {impact_level} sul mercato')
    
    def generate_trend_reasoning(self, category, trend_score, sentiment, velocity, probability, confidence):
        """Genera spiegazione per trend"""
        reasons = []
        
        if trend_score >= 6:
            reasons.append(f"Trend {category} molto forte (score: {trend_score})")
        elif trend_score >= 3:
            reasons.append(f"Trend {category} rilevato (score: {trend_score})")
        
        if sentiment['intensity'] in ['strong', 'moderate']:
            reasons.append(f"Sentiment di mercato {sentiment['intensity']}")
        
        if velocity >= 80:
            reasons.append("Crescita trend esplosiva")
        elif velocity >= 50:
            reasons.append("Velocità diffusione sostenuta")
        
        if confidence > 70:
            reasons.append(f"Alta affidabilità predizione ({confidence:.0f}%)")
        
        return " • ".join(reasons[:4])
    
    def track_trend_prediction_enhanced(self, post_id, prediction_data, post_score, subreddit, title):
        """Traccia predizione trend"""
        current_time = datetime.now()
        
        self.active_predictions[post_id] = {
            'timestamp': current_time.isoformat(),
            'prediction': prediction_data,
            'original_score': post_score,
            'subreddit': subreddit,
            'title': title,
            'trend_category': prediction_data['trend_category'],
            'trend_score': prediction_data['trend_score'],
            'predicted_probability': prediction_data['trend_probability'],
            'predicted_momentum': prediction_data['predicted_momentum'],
            'confidence': prediction_data['confidence'],
            'market_sentiment': prediction_data['market_sentiment']['intensity'],
            'velocity_score': prediction_data['velocity_score'],
            'market_impact': prediction_data['market_impact'],
            'hour_of_day': current_time.hour
        }
        self.save_predictions()
    
    async def check_and_learn_enhanced(self, reddit):
        """Sistema di learning per trend"""
        if not self.active_predictions:
            return
        
        current_time = datetime.now()
        learned_count = 0
        
        for post_id, prediction_data in list(self.active_predictions.items()):
            try:
                prediction_time = datetime.fromisoformat(prediction_data['timestamp'])
                hours_passed = (current_time - prediction_time).total_seconds() / 3600
                
                if hours_passed >= 12:  # Più tempo per valutare trend
                    estimated_trend = self.estimate_trend_outcome_enhanced(prediction_data, hours_passed)
                    
                    self.apply_enhanced_gradient_learning(prediction_data, estimated_trend)
                    self.update_performance_stats(prediction_data, estimated_trend)
                    
                    del self.active_predictions[post_id]
                    learned_count += 1
                    
            except Exception as e:
                logger.warning(f"Errore learning trend per {post_id}: {e}")
                if post_id in self.active_predictions:
                    del self.active_predictions[post_id]
        
        if learned_count > 0:
            logger.info(f"Market Trend Learning: Aggiornati pesi da {learned_count} predizioni")
            self.save_weights()
            self.save_predictions()
            self.save_performance_stats()
            self.save_trend_history()
            
            accuracy = self.calculate_current_accuracy()
            logger.info(f"📈 Accuracy trend attuale: {accuracy:.1f}%")
    
    def estimate_trend_outcome_enhanced(self, prediction_data, hours_passed):
        """Stima se il trend si è materializzato"""
        original_score = prediction_data['original_score']
        predicted_prob = prediction_data['predicted_probability']
        predicted_momentum = prediction_data['predicted_momentum']
        confidence = prediction_data['confidence']
        category = prediction_data['trend_category']
        
        # Multiplicatori per categoria trend
        trend_multipliers = {
            'market_volatility': 8,
            'investment_trends': 10,
            'consumer_sentiment': 12,
            'product_trends': 15,
            'technology_adoption': 20,
            'general': 6
        }
        
        multiplier = trend_multipliers.get(category, 6)
        estimated_momentum = original_score * multiplier
        
        # Criteri per trend significativo
        trend_threshold = 1500
        strong_trend_threshold = 3000
        
        if predicted_prob > 65 and confidence > 60:
            return estimated_momentum > trend_threshold
        elif predicted_prob > 45 and confidence > 40:
            return estimated_momentum > strong_trend_threshold
        elif predicted_prob > 30:
            return estimated_momentum > trend_threshold * 1.5
        else:
            return False
    
    def apply_enhanced_gradient_learning(self, prediction_data, actual_trend):
        """Applica learning per trend"""
        category = prediction_data['trend_category']
        predicted_prob = prediction_data['predicted_probability']
        confidence = prediction_data['confidence']
        trend_score = prediction_data['trend_score']
        
        was_correct = (predicted_prob > 50 and actual_trend) or (predicted_prob <= 50 and not actual_trend)
        
        learning_rate = self.calculate_adaptive_learning_rate(category)
        
        if category in self.weights:
            trend_data = self.weights[category]
            
            if was_correct:
                adjustment = 1 + (learning_rate * (confidence / 100))
                if trend_score >= 6:
                    adjustment *= 1.1
            else:
                adjustment = 1 - (learning_rate * (confidence / 100))
                if predicted_prob > 75 and not actual_trend:
                    adjustment *= 0.9
            
            momentum = trend_data['momentum'] * self.learning_config['momentum_factor']
            new_momentum = (adjustment - 1) * learning_rate + momentum
            
            old_weight = trend_data['weight']
            trend_data['weight'] = max(old_weight * adjustment + new_momentum, 0.1)
            trend_data['momentum'] = new_momentum
            trend_data['samples'] += 1
            
            current_success_rate = trend_data['success_rate']
            if trend_data['samples'] == 1:
                trend_data['success_rate'] = 1.0 if was_correct else 0.0
            else:
                decay = 0.9
                trend_data['success_rate'] = (current_success_rate * decay + 
                                            (1.0 if was_correct else 0.0) * (1 - decay))
            
            logger.debug(f"Trend {category}: {old_weight:.3f} -> {trend_data['weight']:.3f}")
        
        # Aggiorna sentiment weights
        sentiment_intensity = prediction_data['market_sentiment']
        if sentiment_intensity in ['strong', 'moderate'] and abs(predicted_prob - (100 if actual_trend else 0)) > 35:
            self._adjust_sentiment_weights(sentiment_intensity, was_correct, learning_rate)
    
    def _adjust_sentiment_weights(self, intensity, was_correct, learning_rate):
        """Aggiusta pesi sentiment"""
        sentiment_weights = ['bullish_sentiment', 'bearish_sentiment', 'uncertainty_weight']
        
        adjustment = 1 + learning_rate if was_correct else 1 - learning_rate
        
        for weight_key in sentiment_weights:
            if weight_key in self.weights:
                old_weight = self.weights[weight_key]['weight']
                self.weights[weight_key]['weight'] = max(old_weight * adjustment, 0.1)
    
    def update_performance_stats(self, prediction_data, actual_trend):
        """Aggiorna statistiche performance"""
        predicted_prob = prediction_data['predicted_probability']
        category = prediction_data['trend_category']
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
        
        if category not in self.performance_stats['category_performance']:
            self.performance_stats['category_performance'][category] = {
                'total': 0, 'correct': 0, 'accuracy': 0.0
            }
        
        category_stats = self.performance_stats['category_performance'][category]
        category_stats['total'] += 1
        if was_correct:
            category_stats['correct'] += 1
        category_stats['accuracy'] = category_stats['correct'] / category_stats['total']
        
        if str(hour) not in self.performance_stats['hourly_accuracy']:
            self.performance_stats['hourly_accuracy'][str(hour)] = {'total': 0, 'correct': 0}
        
        hour_stats = self.performance_stats['hourly_accuracy'][str(hour)]
        hour_stats['total'] += 1
        if was_correct:
            hour_stats['correct'] += 1
        
        if subreddit not in self.performance_stats['subreddit_performance']:
            self.performance_stats['subreddit_performance'][subreddit] = {
                'total': 0, 'correct': 0, 'accuracy': 0.0
            }
        
        sub_stats = self.performance_stats['subreddit_performance'][subreddit]
        sub_stats['total'] += 1
        if was_correct:
            sub_stats['correct'] += 1
        sub_stats['accuracy'] = sub_stats['correct'] / sub_stats['total']
        
        if category not in self.trend_success_history:
            self.trend_success_history[category] = {
                'recent_performance': deque(maxlen=20),
                'trend': 'stable'
            }
        
        self.trend_success_history[category]['recent_performance'].append(1 if was_correct else 0)
    
    def calculate_current_accuracy(self):
        """Calcola accuracy attuale"""
        if self.performance_stats['total_predictions'] == 0:
            return 50.0
        
        return (self.performance_stats['correct_predictions'] / 
                self.performance_stats['total_predictions']) * 100
    
    def get_learning_insights(self):
        """Ottieni insights sul learning"""
        if not self.performance_stats['accuracy_trend']:
            return {
                'overall_accuracy': 50.0,
                'recent_accuracy': 50.0,
                'total_predictions': 0,
                'best_categories': [],
                'trend': 'stable'
            }
        
        recent_accuracy = sum(list(self.performance_stats['accuracy_trend'])[-10:]) / min(10, len(self.performance_stats['accuracy_trend'])) * 100
        overall_accuracy = self.calculate_current_accuracy()
        
        best_categories = []
        for category, stats in self.performance_stats['category_performance'].items():
            if stats['total'] >= 3:
                best_categories.append((category, stats['accuracy']))
        
        best_categories.sort(key=lambda x: x[1], reverse=True)
        
        insights = {
            'overall_accuracy': overall_accuracy,
            'recent_accuracy': recent_accuracy,
            'total_predictions': self.performance_stats['total_predictions'],
            'best_categories': best_categories[:5],
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
        self.market_ai = EnhancedMarketTrendAI()
        
        # State management
        self.active_chats = set()
        self.reddit = None
        self.analyzed_posts = set()
        
        # Subreddit per analisi trend di mercato
        self.market_subreddits = [
            'stocks', 'investing', 'wallstreetbets', 'stockmarket', 'trading',
            'economics', 'business', 'finance', 'personalfinance', 'financialindependence',
            'cryptocurrency', 'bitcoin', 'ethereum', 'defi', 'cryptomarkets',
            'technology', 'tech', 'gadgets', 'artificial', 'MachineLearning',
            'entrepreneur', 'startups', 'smallbusiness', 'marketing', 'sales',
            'consumer', 'shopping', 'ecommerce', 'amazon', 'shopify',
            'sustainability', 'green', 'climate', 'environment', 'renewableenergy',
            'health', 'fitness', 'nutrition', 'wellness', 'supplements'
        ]
        
    async def initialize(self):
        """Inizializza Reddit connection"""
        try:
            self.reddit = asyncpraw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent='MarketTrendAnalyzer/2.0'
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
        """📈 Analizza trend di mercato e comportamenti consumatori"""
        try:
            trend_analysis = []
            current_time = datetime.now()
            
            for subreddit_name in self.market_subreddits:
                try:
                    subreddit = await self.reddit.subreddit(subreddit_name)
                    
                    count = 0
                    async for post in subreddit.hot(limit=15):
                        count += 1
                        
                        post_time = datetime.fromtimestamp(post.created_utc)
                        minutes_ago = (current_time - post_time).total_seconds() / 60
                        
                        if minutes_ago <= 240 and post.score >= 10:  # 4 ore, >10 upvotes
                            
                            # Market Trend AI Prediction
                            trend_prediction = self.market_ai.predict_trend_momentum_enhanced(
                                post, subreddit_name, minutes_ago
                            )
                            
                            # Filtra per trend significativi
                            if trend_prediction['trend_probability'] >= 40 and post.id not in self.analyzed_posts:
                                trend_analysis.append({
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
                                self.market_ai.track_trend_prediction_enhanced(
                                    post.id, trend_prediction, post.score, subreddit_name, post.title
                                )
                        
                        if count >= 15:
                            break
                            
                except Exception as e:
                    logger.warning(f"Errore {subreddit_name}: {e}")
                    continue
            
            trend_analysis.sort(key=lambda x: x['trend_prediction']['trend_probability'], reverse=True)
            
            logger.info(f"📈 Market Trend AI: {len(trend_analysis)} trend rilevati")
            
            return {
                'trend_analysis': trend_analysis[:8],  # Top 8 trend
                'timestamp': current_time,
                'learning_insights': self.market_ai.get_learning_insights()
            }
            
        except Exception as e:
            logger.error(f"Errore analisi trend: {e}")
            return None
    
    def format_trend_alert(self, data):
        """📊 Formatta alert trend di mercato"""
        if not data or not data['trend_analysis']:
            return "❌ Nessun trend significativo rilevato."
        
        timestamp = data['timestamp'].strftime("%H:%M - %d/%m/%Y")
        trends = data['trend_analysis']
        insights = data.get('learning_insights', {})
        
        message = f"📈 MARKET TREND ANALYZER 📈\n"
        message += f"⏰ {timestamp} | 🧠 Enhanced Market Trend AI\n"
        
        if insights.get('total_predictions', 0) > 0:
            accuracy = insights.get('overall_accuracy', 50)
            recent_accuracy = insights.get('recent_accuracy', 50)
            trend = insights.get('trend', 'stable')
            trend_emoji = "📈" if trend == 'improving' else "📉" if trend == 'declining' else "➡️"
            
            message += f"📊 AI Performance: {accuracy:.1f}% accuracy | Recent: {recent_accuracy:.1f}% {trend_emoji}\n"
        
        message += f"🎯 {len(trends)} TREND EMERGENTI RILEVATI:\n"
        
        for i, trend in enumerate(trends, 1):
            title = trend['title'][:50] + "..." if len(trend['title']) > 50 else trend['title']
            title = title.replace('[', '').replace(']', '').replace('*', '')
            
            prediction = trend['trend_prediction']
            
            # Emoji basati su probabilità e impatto
            if prediction['trend_probability'] >= 70 and prediction['market_impact']['level'] == 'high':
                emoji = "🚀🔥"
                level = "TREND ESPLOSIVO"
            elif prediction['trend_probability'] >= 60:
                emoji = "⚡📈"
                level = "TREND FORTE"
            else:
                emoji = "📊🎯"
                level = "TREND MODERATO"
            
            message += f"\n{emoji} {i}. {title}\n"
            message += f"📊 {trend['score']} upvotes ({trend['upvotes_per_min']}/min) | 💬 {trend['comments']}\n"
            
            # Market Trend AI data
            message += f"🧠 Trend Prob: {prediction['trend_probability']}% | Confidence: {prediction['confidence']}%\n"
            message += f"🎯 Categoria: {prediction['trend_category']} (score: {prediction.get('trend_score', 0)})\n"
            message += f"📈 Predizione → {prediction['predicted_momentum']:,} engagement in {prediction['predicted_timeline_hours']:.1f}h\n"
            message += f"⚡ Velocità: {prediction['velocity_raw']}/min | Sentiment: {prediction['market_sentiment']['intensity']}\n"
            message += f"💼 Impatto Mercato: {prediction['market_impact']['level']} ({prediction['market_impact']['score']}/100)\n"
            message += f"💭 {prediction['market_impact']['description']}\n"
            
            if prediction.get('reasoning'):
                message += f"🔍 {prediction['reasoning']}\n"
            
            message += f"📍 r/{trend['subreddit']} | ⏱️ {trend['minutes_ago']} min fa\n"
            message += f"🔗 {trend['url']}\n"
        
        # Market Insights Summary
        if insights.get('best_categories'):
            top_category = insights['best_categories'][0]
            message += f"\n🏆 Categoria più affidabile: {top_category[0]} ({top_category[1]*100:.0f}% accuracy)\n"
        
        total_pred = insights.get('total_predictions', 0)
        message += f"📚 Trend analizzati: {total_pred} | Learning Status: Active\n"
        message += f"⚡ Market Trend Analyzer v2.0 | Focus: Consumer Behavior & Market Trends"
        
        return message
    
    async def send_to_telegram(self, message):
        """📤 Invia alert a Telegram"""
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
        """🚀 MAIN LOOP - Analisi ogni 15 minuti"""
        logger.info("🚀 Avvio Market Trend Analyzer...")
        logger.info("🧠 Enhanced Market Trend AI per consumer behavior")
        logger.info("⏰ Scansione ogni 15 minuti + Learning ogni 3 ore")
        
        if not await self.initialize():
            return
        
        logger.info("✅ Market Trend Analyzer operativo!")
        
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                
                await self.get_active_chats()
                
                # Learning check ogni 12 cicli (3 ore)
                if cycle_count % 12 == 0:
                    logger.info("🧠 Market Trend Learning check...")
                    await self.market_ai.check_and_learn_enhanced(self.reddit)
                
                # Analisi trend di mercato
                logger.info("🔍 Analisi trend di mercato...")
                trend_data = await self.analyze_market_trends()
                
                if trend_data and trend_data['trend_analysis']:
                    new_trends = [t for t in trend_data['trend_analysis'] if t['id'] not in self.analyzed_posts]
                    
                    if new_trends and self.active_chats:
                        for trend in new_trends:
                            self.analyzed_posts.add(trend['id'])
                        
                        trend_data['trend_analysis'] = new_trends
                        message = self.format_trend_alert(trend_data)
                        success = await self.send_to_telegram(message)
                        
                        if success:
                            logger.info(f"🔥 {len(new_trends)} trend inviati!")
                            
                            for trend in new_trends:
                                pred = trend['trend_prediction']
                                logger.info(
                                    f"  📈 {pred['trend_category']}: {pred['trend_probability']}% "
                                    f"(impact: {pred['market_impact']['level']}) → "
                                    f"{trend['title'][:30]}..."
                                )
                
                # Pulizia cache
                if len(self.analyzed_posts) > 1000:
                    self.analyzed_posts.clear()
                
                # Stats logging ogni 24 cicli (6 ore)
                if cycle_count % 24 == 0:
                    insights = self.market_ai.get_learning_insights()
                    logger.info(f"📊 Market AI Stats: {insights.get('total_predictions', 0)} predictions, "
                              f"{insights.get('overall_accuracy', 0):.1f}% accuracy")
                
                await asyncio.sleep(900)  # 15 minuti
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Errore main loop: {e}")
                await asyncio.sleep(300)  # Riprova dopo 5 minuti
        
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
    logger.info("🚀 Market Trend Analyzer v2.0 - Consumer Behavior & Market Trends")
    asyncio.run(main())
