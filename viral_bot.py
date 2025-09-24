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
    handlers=[logging.StreamHandler(), logging.FileHandler('enhanced_bot.log')]
)
logger = logging.getLogger(__name__)

# ===== ENHANCED GRADIENT LEARNING AI ENGINE =====
class EnhancedGradientLearningAI:
    def __init__(self):
        # File per salvare i pesi appresi
        self.weights_file = 'ai_weights.json'
        self.predictions_file = 'predictions_track.json'
        self.performance_stats_file = 'performance_stats.json'
        self.pattern_history_file = 'pattern_history.json'
        
        # Performance tracking per migliorare learning
        self.performance_stats = self.load_performance_stats()
        self.pattern_success_history = self.load_pattern_history()
        
        # Learning parameters migliorati
        self.learning_config = {
            'base_learning_rate': 0.03,  # Ridotto per stabilità
            'adaptive_learning_rate': True,
            'momentum_factor': 0.1,
            'confidence_threshold': 0.7,
            'min_samples_for_pattern': 3,  # Minimo campioni per considerare pattern affidabile
            'pattern_decay_rate': 0.99,   # Decay graduale per pattern vecchi
            'success_boost_adaptive': True,
            'failure_penalty_adaptive': True
        }
        
        # Pesi con momentum tracking
        self.default_weights = {
            # Pattern specifici con momentum
            'elon_musk': {'weight': 2.8, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'ai_breakthrough': {'weight': 2.0, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'crypto_crash': {'weight': 3.2, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'tech_layoffs': {'weight': 1.8, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'scandal_celebrity': {'weight': 2.5, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'market_crash': {'weight': 3.5, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'space_news': {'weight': 1.6, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'gaming_drama': {'weight': 1.9, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'political_news': {'weight': 2.3, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'health_news': {'weight': 1.7, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'climate_news': {'weight': 1.4, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'sports_news': {'weight': 1.5, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'entertainment': {'weight': 1.8, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'science_discovery': {'weight': 1.9, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'tech_general': {'weight': 1.6, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'business_news': {'weight': 1.5, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'general': {'weight': 1.0, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            
            # Sentiment multipliers con tracking
            'high_emotion_weight': {'weight': 1.6, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'urgency_weight': {'weight': 2.2, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'controversy_weight': {'weight': 2.0, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'numbers_weight': {'weight': 1.4, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'exclusivity_weight': {'weight': 1.7, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            
            # Velocity thresholds adaptive
            'velocity_explosive_threshold': {'weight': 45.0, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'velocity_fast_threshold': {'weight': 18.0, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
            'velocity_steady_threshold': {'weight': 8.0, 'momentum': 0, 'samples': 0, 'success_rate': 0.5},
        }
        
        # Carica pesi salvati o usa default
        self.weights = self.load_weights()
        
        # Tracking predizioni per feedback migliorato
        self.active_predictions = self.load_predictions()
        
        # Pattern keywords con scoring dinamico
        self.pattern_keywords = {
            'elon_musk': {
                'primary': ['elon', 'musk', 'tesla', 'spacex'],
                'secondary': ['neuralink', 'boring company', 'starlink'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'ai_breakthrough': {
                'primary': ['ai', 'artificial intelligence', 'chatgpt', 'gpt', 'openai'],
                'secondary': ['claude', 'robot', 'automation', 'machine learning', 'neural network'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'crypto_crash': {
                'primary': ['bitcoin', 'crypto', 'ethereum', 'crash', 'pump', 'dump'],
                'secondary': ['blockchain', 'defi', 'nft', 'bull', 'bear'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'tech_layoffs': {
                'primary': ['layoffs', 'fired', 'job cuts'],
                'secondary': ['downsizing', 'restructuring', 'redundant'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'scandal_celebrity': {
                'primary': ['scandal', 'controversy', 'exposed', 'arrest'],
                'secondary': ['caught', 'lawsuit', 'divorce', 'affair'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'market_crash': {
                'primary': ['market', 'stock', 'crash', 'plummet'],
                'secondary': ['dow', 'nasdaq', 'sp500', 'bear market', 'recession'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'space_news': {
                'primary': ['space', 'mars', 'moon', 'rocket', 'nasa'],
                'secondary': ['spacex', 'iss', 'satellite', 'astronaut'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'gaming_drama': {
                'primary': ['gaming', 'game', 'streamer', 'twitch'],
                'secondary': ['youtube', 'esports', 'nintendo', 'sony', 'xbox'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'political_news': {
                'primary': ['trump', 'biden', 'election', 'president'],
                'secondary': ['congress', 'senate', 'politics', 'vote', 'policy'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'health_news': {
                'primary': ['covid', 'vaccine', 'pandemic', 'health'],
                'secondary': ['medical', 'doctor', 'hospital', 'disease', 'cure'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'climate_news': {
                'primary': ['climate', 'global warming', 'carbon'],
                'secondary': ['emission', 'green', 'renewable', 'pollution'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'sports_news': {
                'primary': ['football', 'basketball', 'soccer', 'olympics'],
                'secondary': ['championship', 'world cup', 'nfl', 'nba'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'entertainment': {
                'primary': ['movie', 'netflix', 'disney', 'actor'],
                'secondary': ['actress', 'film', 'tv show', 'series', 'music'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'science_discovery': {
                'primary': ['study', 'research', 'scientists', 'discovery'],
                'secondary': ['breakthrough', 'experiment', 'published'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'tech_general': {
                'primary': ['technology', 'tech', 'startup', 'innovation'],
                'secondary': ['app', 'software', 'hardware', 'gadget'],
                'score': {'primary': 3, 'secondary': 2}
            },
            'business_news': {
                'primary': ['business', 'company', 'ceo', 'merger'],
                'secondary': ['acquisition', 'ipo', 'earnings', 'revenue'],
                'score': {'primary': 3, 'secondary': 2}
            }
        }
        
        # Sentiment keywords con scoring più preciso
        self.sentiment_keywords = {
            'high_emotion': {
                'extreme': ['shocking', 'unbelievable', 'insane', 'mind-blowing', 'devastating'],
                'high': ['crazy', 'amazing', 'incredible', 'revolutionary', 'horrific'],
                'medium': ['breakthrough', 'game-changing', 'tragic', 'miraculous'],
                'score': {'extreme': 4, 'high': 3, 'medium': 2}
            },
            'urgency': {
                'extreme': ['breaking', 'urgent', 'emergency', 'alert'],
                'high': ['just in', 'developing', 'live', 'immediate'],
                'medium': ['now', 'today'],
                'score': {'extreme': 4, 'high': 3, 'medium': 2}
            },
            'controversy': {
                'extreme': ['banned', 'illegal', 'scandal', 'exposed'],
                'high': ['censored', 'forbidden', 'controversial', 'outrageous'],
                'medium': ['leaked', 'disputed'],
                'score': {'extreme': 4, 'high': 3, 'medium': 2}
            },
            'numbers': {
                'extreme': ['trillion', 'billion', 'record', 'historic'],
                'high': ['million', 'highest', 'lowest', 'largest'],
                'medium': ['%', '$', 'first', 'biggest'],
                'score': {'extreme': 4, 'high': 3, 'medium': 2}
            },
            'exclusivity': {
                'extreme': ['exclusive', 'never before', 'unprecedented', 'secret'],
                'high': ['only', 'rare', 'hidden', 'revealed'],
                'medium': ['insider', 'special'],
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
            'accuracy_trend': deque(maxlen=50),  # Ultimi 50 risultati
            'pattern_performance': {},
            'hourly_accuracy': {},  # Accuracy per ora del giorno
            'subreddit_performance': {}
        }
    
    def load_pattern_history(self):
        """Carica storico performance pattern"""
        try:
            if os.path.exists(self.pattern_history_file):
                with open(self.pattern_history_file, 'r') as f:
                    data = json.load(f)
                    # Converti deque serializzate
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
        """Carica pesi salvati con struttura migliorata"""
        try:
            if os.path.exists(self.weights_file):
                with open(self.weights_file, 'r') as f:
                    saved_weights = json.load(f)
                    # Merge con default per nuove chiavi e struttura
                    weights = {}
                    for key, default_data in self.default_weights.items():
                        if key in saved_weights:
                            if isinstance(saved_weights[key], dict):
                                weights[key] = saved_weights[key]
                            else:
                                # Converti vecchio formato
                                weights[key] = {
                                    'weight': saved_weights[key],
                                    'momentum': 0,
                                    'samples': 0,
                                    'success_rate': 0.5
                                }
                        else:
                            weights[key] = default_data.copy()
                    
                    logger.info(f"Caricati pesi Enhanced AI da {self.weights_file}")
                    return weights
        except Exception as e:
            logger.warning(f"Errore caricamento pesi: {e}")
        
        logger.info("Inizializzo pesi Enhanced AI default")
        return {k: v.copy() for k, v in self.default_weights.items()}
    
    def save_weights(self):
        """Salva pesi aggiornati"""
        try:
            with open(self.weights_file, 'w') as f:
                json.dump(self.weights, f, indent=2)
            logger.debug("Pesi Enhanced AI salvati")
        except Exception as e:
            logger.error(f"Errore salvataggio pesi: {e}")
    
    def save_performance_stats(self):
        """Salva statistiche performance"""
        try:
            # Prepara dati per serializzazione
            stats_to_save = self.performance_stats.copy()
            stats_to_save['accuracy_trend'] = list(stats_to_save['accuracy_trend'])
            
            with open(self.performance_stats_file, 'w') as f:
                json.dump(stats_to_save, f, indent=2)
        except Exception as e:
            logger.error(f"Errore salvataggio performance stats: {e}")
    
    def save_pattern_history(self):
        """Salva storico pattern"""
        try:
            # Prepara dati per serializzazione
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
    
    def identify_pattern_category_enhanced(self, title, subreddit):
        """Identifica categoria con scoring migliorato"""
        title_lower = title.lower()
        subreddit_lower = subreddit.lower()
        
        pattern_scores = {}
        
        # Calcola score per ogni pattern
        for pattern, keywords in self.pattern_keywords.items():
            score = 0
            
            # Primary keywords
            for keyword in keywords['primary']:
                if keyword in title_lower or keyword in subreddit_lower:
                    score += keywords['score']['primary']
            
            # Secondary keywords
            for keyword in keywords['secondary']:
                if keyword in title_lower or keyword in subreddit_lower:
                    score += keywords['score']['secondary']
            
            if score > 0:
                pattern_scores[pattern] = score
        
        if pattern_scores:
            # Ritorna pattern con score più alto
            best_pattern = max(pattern_scores.items(), key=lambda x: x[1])
            return best_pattern[0], best_pattern[1]
        
        return 'general', 0
    
    def analyze_sentiment_enhanced(self, title):
        """Analisi sentiment con scoring migliorato"""
        text = title.lower()
        sentiment_data = {}
        total_score = 0
        
        # Calcola score per ogni categoria sentiment
        for category, levels in self.sentiment_keywords.items():
            category_score = 0
            
            for level, keywords in levels.items():
                if level == 'score':
                    continue
                
                for keyword in keywords:
                    if keyword in text:
                        category_score += levels['score'][level]
            
            sentiment_data[category] = category_score
            total_score += category_score * self.get_weight(f'{category}_weight')
        
        # Calcola intensità basata su score totale
        if total_score >= 30:
            intensity = 'extreme'
        elif total_score >= 20:
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
        """Ottieni peso corrente con fallback"""
        if key in self.weights:
            return self.weights[key]['weight']
        return 1.0
    
    def calculate_adaptive_learning_rate(self, pattern):
        """Calcola learning rate adaptivo per pattern"""
        base_rate = self.learning_config['base_learning_rate']
        
        if pattern not in self.weights:
            return base_rate
        
        pattern_data = self.weights[pattern]
        samples = pattern_data.get('samples', 0)
        success_rate = pattern_data.get('success_rate', 0.5)
        
        # Riduci learning rate con più campioni (più stabilità)
        sample_factor = 1 / (1 + samples * 0.1)
        
        # Aumenta learning rate per pattern con bassa performance
        performance_factor = 2 - success_rate if success_rate < 0.5 else 1
        
        adaptive_rate = base_rate * sample_factor * performance_factor
        return min(adaptive_rate, base_rate * 2)  # Cap massimo
    
    def predict_viral_trajectory_enhanced(self, post, subreddit, minutes_ago):
        """CORE: Predizione migliorata con Enhanced Learning"""
        
        # 1. Identifica pattern con scoring
        pattern_category, pattern_score = self.identify_pattern_category_enhanced(post.title, subreddit)
        pattern_multiplier = self.get_weight(pattern_category)
        
        # Bonus per pattern score alto
        if pattern_score >= 6:
            pattern_multiplier *= 1.3
        elif pattern_score >= 4:
            pattern_multiplier *= 1.2
        
        # 2. Analisi sentiment migliorata
        sentiment = self.analyze_sentiment_enhanced(post.title)
        sentiment_multiplier = 1 + (sentiment['weighted_score'] / 100)
        
        # 3. Velocità con soglie adaptive
        velocity = post.score / max(minutes_ago, 1)
        explosive_threshold = self.get_weight('velocity_explosive_threshold')
        fast_threshold = self.get_weight('velocity_fast_threshold')
        steady_threshold = self.get_weight('velocity_steady_threshold')
        
        if velocity >= explosive_threshold:
            velocity_score = 100
        elif velocity >= fast_threshold:
            velocity_score = 80
        elif velocity >= steady_threshold:
            velocity_score = 50
        else:
            velocity_score = min(velocity * 3, 40)
        
        velocity_multiplier = 1 + (velocity_score / 100)
        
        # 4. Engagement analysis migliorato
        if post.score > 0:
            comment_ratio = post.num_comments / post.score
            optimal_ratio = 0.08  # 8% ottimale per viral content
            ratio_penalty = abs(comment_ratio - optimal_ratio) / optimal_ratio
            engagement_multiplier = max(1 + (comment_ratio * 2) - ratio_penalty, 0.5)
        else:
            engagement_multiplier = 1.0
        
        # 5. Time decay migliorato
        if minutes_ago > 240:  # >4 ore
            time_multiplier = 0.7
        elif minutes_ago > 120:  # >2 ore
            time_multiplier = 0.9
        else:
            time_multiplier = 1.1  # Bonus per content fresco
        
        # 6. Calcolo probabilità con confidence
        base_probability = 0.25
        
        final_probability = (
            base_probability * 
            pattern_multiplier * 
            sentiment_multiplier * 
            velocity_multiplier * 
            engagement_multiplier * 
            time_multiplier
        )
        
        # Clamp con confidence adjustment
        final_probability = max(0.01, min(final_probability, 0.98))
        
        # Calcola confidence basata su samples del pattern
        pattern_data = self.weights.get(pattern_category, {'samples': 0, 'success_rate': 0.5})
        confidence_raw = min(pattern_data['samples'] / 10, 1.0)  # Max confidence con 10+ samples
        success_rate = pattern_data['success_rate']
        
        # Confidence finale combina samples e success rate
        confidence = (confidence_raw * success_rate + (1 - confidence_raw) * 0.5) * 100
        
        # Predici score finale con volatilità
        growth_factor = final_probability * 25
        volatility = 1 + (sentiment['weighted_score'] / 200)  # Volatilità da sentiment
        predicted_final_score = int(post.score * (1 + growth_factor) * volatility)
        
        # Peak time basato su pattern e velocità
        base_peak_hours = self._calculate_peak_time(pattern_category, velocity, sentiment['intensity'])
        
        return {
            'viral_probability': round(final_probability * 100, 1),
            'confidence': round(confidence, 1),
            'confidence_level': 'high' if confidence > 70 else 'medium' if confidence > 40 else 'low',
            'predicted_peak_hours': base_peak_hours,
            'predicted_final_score': predicted_final_score,
            'pattern_match': pattern_category,
            'pattern_score': pattern_score,
            'pattern_samples': pattern_data['samples'],
            'pattern_success_rate': round(pattern_data['success_rate'] * 100, 1),
            'sentiment_analysis': sentiment,
            'velocity_score': velocity_score,
            'velocity_raw': round(velocity, 2),
            'pattern_multiplier': round(pattern_multiplier, 2),
            'reasoning': self.generate_enhanced_reasoning(
                pattern_category, pattern_score, sentiment, velocity_score, 
                final_probability * 100, confidence
            )
        }
    
    def _calculate_peak_time(self, pattern, velocity, intensity):
        """Calcola tempo di picco basato su pattern e caratteristiche"""
        base_times = {
            'market_crash': 1.5,
            'crypto_crash': 2,
            'elon_musk': 3,
            'scandal_celebrity': 4,
            'ai_breakthrough': 8,
            'space_news': 12,
            'science_discovery': 16
        }
        
        base = base_times.get(pattern, 6)
        
        # Aggiusta per velocità
        if velocity > 80:
            base *= 0.7  # Più veloce = picco prima
        elif velocity > 50:
            base *= 0.9
        
        # Aggiusta per intensità
        if intensity == 'extreme':
            base *= 0.8
        elif intensity == 'high':
            base *= 0.9
        
        return max(base, 0.5)
    
    def generate_enhanced_reasoning(self, pattern, pattern_score, sentiment, velocity, probability, confidence):
        """Genera spiegazione migliorata"""
        reasons = []
        
        # Pattern reasoning con score
        if pattern_score >= 6:
            reasons.append(f"Pattern {pattern} molto forte (score: {pattern_score})")
        elif pattern_score >= 3:
            reasons.append(f"Pattern {pattern} rilevato (score: {pattern_score})")
        
        # Sentiment reasoning
        if sentiment['intensity'] in ['extreme', 'high']:
            reasons.append(f"Alto carico emotivo ({sentiment['intensity']}: {sentiment['total_score']})")
        
        # Velocity reasoning
        if velocity >= 80:
            reasons.append("Crescita esplosiva in corso")
        elif velocity >= 50:
            reasons.append("Velocità sostenuta elevata")
        
        # Confidence reasoning
        if confidence > 70:
            reasons.append(f"Alta affidabilità predizione ({confidence:.0f}%)")
        elif confidence < 30:
            reasons.append(f"Predizione esplorativa ({confidence:.0f}%)")
        
        return " • ".join(reasons[:4])  # Max 4 reasons
    
    def track_prediction_enhanced(self, post_id, prediction_data, post_score, subreddit, title):
        """Traccia predizione con metadati migliorati"""
        current_time = datetime.now()
        
        self.active_predictions[post_id] = {
            'timestamp': current_time.isoformat(),
            'prediction': prediction_data,
            'original_score': post_score,
            'subreddit': subreddit,
            'title': title,
            'pattern': prediction_data['pattern_match'],
            'pattern_score': prediction_data['pattern_score'],
            'predicted_probability': prediction_data['viral_probability'],
            'predicted_final_score': prediction_data['predicted_final_score'],
            'confidence': prediction_data['confidence'],
            'sentiment_intensity': prediction_data['sentiment_analysis']['intensity'],
            'velocity_score': prediction_data['velocity_score'],
            'hour_of_day': current_time.hour
        }
        self.save_predictions()
    
    async def check_and_learn_enhanced(self, reddit):
        """Sistema di learning migliorato"""
        if not self.active_predictions:
            return
        
        current_time = datetime.now()
        learned_count = 0
        
        for post_id, prediction_data in list(self.active_predictions.items()):
            try:
                # Controlla predizioni di almeno 8 ore fa per maggiore accuratezza
                prediction_time = datetime.fromisoformat(prediction_data['timestamp'])
                hours_passed = (current_time - prediction_time).total_seconds() / 3600
                
                if hours_passed >= 8:
                    # Stima risultato finale
                    estimated_viral = self.estimate_viral_outcome_enhanced(prediction_data, hours_passed)
                    
                    # Applica Enhanced Gradient Learning
                    self.apply_enhanced_gradient_learning(prediction_data, estimated_viral)
                    
                    # Aggiorna statistiche performance
                    self.update_performance_stats(prediction_data, estimated_viral)
                    
                    # Rimuovi da tracking
                    del self.active_predictions[post_id]
                    learned_count += 1
                    
            except Exception as e:
                logger.warning(f"Errore enhanced learning per {post_id}: {e}")
                if post_id in self.active_predictions:
                    del self.active_predictions[post_id]
        
        if learned_count > 0:
            logger.info(f"Enhanced Learning: Aggiornati pesi da {learned_count} predizioni")
            self.save_weights()
            self.save_predictions()
            self.save_performance_stats()
            self.save_pattern_history()
            
            # Log performance stats
            accuracy = self.calculate_current_accuracy()
            logger.info(f"🧠 Accuracy attuale: {accuracy:.1f}%")
    
    def estimate_viral_outcome_enhanced(self, prediction_data, hours_passed):
        """Stima migliorata se il contenuto è diventato virale"""
        original_score = prediction_data['original_score']
        predicted_prob = prediction_data['predicted_probability']
        predicted_final = prediction_data['predicted_final_score']
        confidence = prediction_data['confidence']
        pattern = prediction_data['pattern']
        
        # Stima score finale basata su crescita tipica per pattern
        pattern_multipliers = {
            'elon_musk': 15,
            'crypto_crash': 12,
            'market_crash': 18,
            'scandal_celebrity': 20,
            'ai_breakthrough': 8,
            'political_news': 10,
            'space_news': 6,
            'general': 4
        }
        
        multiplier = pattern_multipliers.get(pattern, 4)
        estimated_final_score = original_score * multiplier
        
        # Criteri per viral (più precisi)
        viral_threshold = 2000
        strong_viral_threshold = 5000
        
        # Logica migliorata
        if predicted_prob > 70 and confidence > 60:
            return estimated_final_score > viral_threshold
        elif predicted_prob > 50 and confidence > 40:
            return estimated_final_score > strong_viral_threshold
        elif predicted_prob > 30:
            return estimated_final_score > viral_threshold * 2
        else:
            return False
    
    def apply_enhanced_gradient_learning(self, prediction_data, actual_viral):
        """Applica Enhanced Gradient Learning con momentum e adaptive rates"""
        pattern = prediction_data['pattern']
        predicted_prob = prediction_data['predicted_probability']
        confidence = prediction_data['confidence']
        pattern_score = prediction_data['pattern_score']
        
        was_correct = (predicted_prob > 50 and actual_viral) or (predicted_prob <= 50 and not actual_viral)
        
        # Calcola adaptive learning rate
        learning_rate = self.calculate_adaptive_learning_rate(pattern)
        
        # Aggiorna pattern weights con momentum
        if pattern in self.weights:
            pattern_data = self.weights[pattern]
            
            # Calcola adjustment
            if was_correct:
                adjustment = 1 + (learning_rate * (confidence / 100))
                if pattern_score >= 6:  # Bonus per pattern forti
                    adjustment *= 1.1
            else:
                adjustment = 1 - (learning_rate * (confidence / 100))
                if predicted_prob > 80 and not actual_viral:  # Penalità per false positive confident
                    adjustment *= 0.9
            
            # Applica momentum
            momentum = pattern_data['momentum'] * self.learning_config['momentum_factor']
            new_momentum = (adjustment - 1) * learning_rate + momentum
            
            # Aggiorna peso
            old_weight = pattern_data['weight']
            pattern_data['weight'] = max(old_weight * adjustment + new_momentum, 0.1)
            pattern_data['momentum'] = new_momentum
            pattern_data['samples'] += 1
            
            # Aggiorna success rate (media mobile)
            current_success_rate = pattern_data['success_rate']
            if pattern_data['samples'] == 1:
                pattern_data['success_rate'] = 1.0 if was_correct else 0.0
            else:
                # Media mobile con decay
                decay = 0.9
                pattern_data['success_rate'] = (current_success_rate * decay + 
                                              (1.0 if was_correct else 0.0) * (1 - decay))
            
            logger.debug(f"Pattern {pattern}: {old_weight:.3f} -> {pattern_data['weight']:.3f} "
                        f"(success_rate: {pattern_data['success_rate']:.3f})")
        
        # Aggiorna sentiment weights se necessario
        sentiment_intensity = prediction_data['sentiment_intensity']
        if sentiment_intensity in ['extreme', 'high'] and abs(predicted_prob - (100 if actual_viral else 0)) > 40:
            self._adjust_sentiment_weights(sentiment_intensity, was_correct, learning_rate)
        
        # Aggiorna velocity thresholds
        velocity_score = prediction_data['velocity_score']
        if velocity_score > 80:
            self._adjust_velocity_thresholds(was_correct, learning_rate)
    
    def _adjust_sentiment_weights(self, intensity, was_correct, learning_rate):
        """Aggiusta pesi sentiment basandosi sui risultati"""
        sentiment_weights = ['high_emotion_weight', 'urgency_weight', 'controversy_weight']
        
        adjustment = 1 + learning_rate if was_correct else 1 - learning_rate
        
        for weight_key in sentiment_weights:
            if weight_key in self.weights:
                old_weight = self.weights[weight_key]['weight']
                self.weights[weight_key]['weight'] = max(old_weight * adjustment, 0.1)
    
    def _adjust_velocity_thresholds(self, was_correct, learning_rate):
        """Aggiusta soglie velocità"""
        threshold_keys = ['velocity_explosive_threshold', 'velocity_fast_threshold', 'velocity_steady_threshold']
        
        # Se la predizione era sbagliata per alta velocità, aggiusta soglie
        if not was_correct:
            adjustment = 1.05  # Aumenta soglie se false positive
            for key in threshold_keys:
                if key in self.weights:
                    self.weights[key]['weight'] *= adjustment
    
    def update_performance_stats(self, prediction_data, actual_viral):
        """Aggiorna statistiche performance complete"""
        predicted_prob = prediction_data['predicted_probability']
        pattern = prediction_data['pattern']
        subreddit = prediction_data['subreddit']
        hour = prediction_data['hour_of_day']
        
        was_correct = (predicted_prob > 50 and actual_viral) or (predicted_prob <= 50 and not actual_viral)
        
        # Statistiche generali
        self.performance_stats['total_predictions'] += 1
        if was_correct:
            self.performance_stats['correct_predictions'] += 1
        
        if predicted_prob > 50 and not actual_viral:
            self.performance_stats['false_positives'] += 1
        elif predicted_prob <= 50 and actual_viral:
            self.performance_stats['false_negatives'] += 1
        
        # Accuracy trend (ultimi 50)
        self.performance_stats['accuracy_trend'].append(1 if was_correct else 0)
        
        # Performance per pattern
        if pattern not in self.performance_stats['pattern_performance']:
            self.performance_stats['pattern_performance'][pattern] = {
                'total': 0, 'correct': 0, 'accuracy': 0.0
            }
        
        pattern_stats = self.performance_stats['pattern_performance'][pattern]
        pattern_stats['total'] += 1
        if was_correct:
            pattern_stats['correct'] += 1
        pattern_stats['accuracy'] = pattern_stats['correct'] / pattern_stats['total']
        
        # Performance per ora del giorno
        if str(hour) not in self.performance_stats['hourly_accuracy']:
            self.performance_stats['hourly_accuracy'][str(hour)] = {'total': 0, 'correct': 0}
        
        hour_stats = self.performance_stats['hourly_accuracy'][str(hour)]
        hour_stats['total'] += 1
        if was_correct:
            hour_stats['correct'] += 1
        
        # Performance per subreddit
        if subreddit not in self.performance_stats['subreddit_performance']:
            self.performance_stats['subreddit_performance'][subreddit] = {
                'total': 0, 'correct': 0, 'accuracy': 0.0
            }
        
        sub_stats = self.performance_stats['subreddit_performance'][subreddit]
        sub_stats['total'] += 1
        if was_correct:
            sub_stats['correct'] += 1
        sub_stats['accuracy'] = sub_stats['correct'] / sub_stats['total']
        
        # Aggiorna pattern history
        if pattern not in self.pattern_success_history:
            self.pattern_success_history[pattern] = {
                'recent_performance': deque(maxlen=20),
                'trend': 'stable'
            }
        
        self.pattern_success_history[pattern]['recent_performance'].append(1 if was_correct else 0)
        
        # Calcola trend pattern
        recent_perf = list(self.pattern_success_history[pattern]['recent_performance'])
        if len(recent_perf) >= 10:
            first_half = sum(recent_perf[:len(recent_perf)//2]) / (len(recent_perf)//2)
            second_half = sum(recent_perf[len(recent_perf)//2:]) / (len(recent_perf) - len(recent_perf)//2)
            
            if second_half > first_half + 0.2:
                self.pattern_success_history[pattern]['trend'] = 'improving'
            elif second_half < first_half - 0.2:
                self.pattern_success_history[pattern]['trend'] = 'declining'
            else:
                self.pattern_success_history[pattern]['trend'] = 'stable'
    
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
                'best_patterns': [],
                'trend': 'stable'
            }
        
        recent_accuracy = sum(list(self.performance_stats['accuracy_trend'])[-10:]) / min(10, len(self.performance_stats['accuracy_trend'])) * 100
        overall_accuracy = self.calculate_current_accuracy()
        
        # Top performing patterns
        best_patterns = []
        for pattern, stats in self.performance_stats['pattern_performance'].items():
            if stats['total'] >= 3:  # Almeno 3 samples
                best_patterns.append((pattern, stats['accuracy']))
        
        best_patterns.sort(key=lambda x: x[1], reverse=True)
        
        insights = {
            'overall_accuracy': overall_accuracy,
            'recent_accuracy': recent_accuracy,
            'total_predictions': self.performance_stats['total_predictions'],
            'best_patterns': best_patterns[:5],
            'trend': 'improving' if recent_accuracy > overall_accuracy else 'stable' if abs(recent_accuracy - overall_accuracy) < 5 else 'declining'
        }
        
        return insights

# ===== ENHANCED VIRAL NEWS HUNTER =====
class EnhancedViralNewsHunter:
    def __init__(self):
        # Credenziali
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        
        if not all([self.reddit_client_id, self.reddit_client_secret, self.telegram_token]):
            raise ValueError("Variabili d'ambiente mancanti!")
        
        # Enhanced Gradient Learning AI
        self.enhanced_ai = EnhancedGradientLearningAI()
        
        # State management
        self.active_chats = set()
        self.reddit = None
        self.sent_posts = set()
        
        # Subreddit ottimizzati per viral detection
        self.viral_subreddits = [
            'news', 'worldnews', 'breakingnews', 'nottheonion', 'offbeat',
            'technology', 'gadgets', 'Futurology', 'singularity', 'artificial',
            'MachineLearning', 'OpenAI', 'ChatGPT', 'cryptocurrency', 'bitcoin', 
            'ethereum', 'business', 'economics', 'stocks', 'wallstreetbets', 
            'investing', 'todayilearned', 'interestingasfuck', 'nextfuckinglevel', 
            'Damnthatsinteresting', 'mildlyinteresting', 'showerthoughts', 
            'explainlikeimfive', 'science', 'space', 'physics', 'biology', 
            'medicine', 'health', 'movies', 'television', 'gaming', 'music', 
            'books', 'politics', 'worldpolitics'
        ]
        
    async def initialize(self):
        """Inizializza Reddit connection"""
        try:
            self.reddit = asyncpraw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent='EnhancedViralNewsHunter/3.0'
            )
            logger.info("✅ Reddit connesso con Enhanced Gradient AI")
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
    
    async def hunt_viral_news_enhanced(self):
        """🧠 Cerca notizie con Enhanced Gradient Learning AI"""
        try:
            viral_posts = []
            current_time = datetime.now()
            
            for subreddit_name in self.viral_subreddits:
                try:
                    subreddit = await self.reddit.subreddit(subreddit_name)
                    
                    count = 0
                    async for post in subreddit.hot(limit=20):
                        count += 1
                        
                        post_time = datetime.fromtimestamp(post.created_utc)
                        minutes_ago = (current_time - post_time).total_seconds() / 60
                        
                        if minutes_ago <= 300 and post.score >= 15:  # 5 ore, >15 upvotes
                            
                            # Enhanced AI Prediction
                            ai_prediction = self.enhanced_ai.predict_viral_trajectory_enhanced(
                                post, subreddit_name, minutes_ago
                            )
                            
                            # Filtra per probabilità significativa
                            if ai_prediction['viral_probability'] >= 45 and post.id not in self.sent_posts:
                                viral_posts.append({
                                    'id': post.id,
                                    'title': post.title,
                                    'score': post.score,
                                    'subreddit': subreddit_name,
                                    'url': f"https://reddit.com{post.permalink}",
                                    'comments': post.num_comments,
                                    'created': post_time,
                                    'minutes_ago': round(minutes_ago),
                                    'upvotes_per_min': round(post.score / max(minutes_ago, 1), 1),
                                    'ai_prediction': ai_prediction
                                })
                                
                                # Traccia per learning
                                self.enhanced_ai.track_prediction_enhanced(
                                    post.id, ai_prediction, post.score, subreddit_name, post.title
                                )
                        
                        if count >= 20:
                            break
                            
                except Exception as e:
                    logger.warning(f"Errore {subreddit_name}: {e}")
                    continue
            
            viral_posts.sort(key=lambda x: x['ai_prediction']['viral_probability'], reverse=True)
            
            logger.info(f"🧠 Enhanced AI: {len(viral_posts)} viral candidates rilevati")
            
            return {
                'viral_posts': viral_posts[:6],
                'timestamp': current_time,
                'learning_insights': self.enhanced_ai.get_learning_insights()
            }
            
        except Exception as e:
            logger.error(f"Errore Enhanced hunt: {e}")
            return None
    
    def format_enhanced_message(self, data):
        """📱 Formatta messaggio con Enhanced AI insights"""
        if not data or not data['viral_posts']:
            return "❌ Nessuna notizia virale rilevata."
        
        timestamp = data['timestamp'].strftime("%H:%M - %d/%m/%Y")
        posts = data['viral_posts']
        insights = data.get('learning_insights', {})
        
        message = f"🚀 ENHANCED VIRAL NEWS HUNTER 🚀\n"
        message += f"⏰ {timestamp} | 🧠 Enhanced Gradient Learning AI\n"
        
        if insights.get('total_predictions', 0) > 0:
            accuracy = insights.get('overall_accuracy', 50)
            recent_accuracy = insights.get('recent_accuracy', 50)
            trend = insights.get('trend', 'stable')
            trend_emoji = "📈" if trend == 'improving' else "📉" if trend == 'declining' else "➡️"
            
            message += f"📊 AI Performance: {accuracy:.1f}% accuracy | Recent: {recent_accuracy:.1f}% {trend_emoji}\n"
        
        message += f"🎯 {len(posts)} PREDIZIONI VIRALI HIGH-CONFIDENCE:\n"
        
        for i, post in enumerate(posts, 1):
            title = post['title'][:55] + "..." if len(post['title']) > 55 else post['title']
            title = title.replace('[', '').replace(']', '').replace('*', '')
            
            ai = post['ai_prediction']
            
            # Emoji basati su probabilità e confidence
            if ai['viral_probability'] >= 75 and ai['confidence'] > 60:
                emoji = "🚀🔥"
                level = "EXPLOSIVE"
            elif ai['viral_probability'] >= 65:
                emoji = "⚡📈"
                level = "HIGH"
            else:
                emoji = "📊🎯"
                level = "MODERATE"
            
            message += f"\n{emoji} {i}. {title}\n"
            message += f"📊 {post['score']} upvotes ({post['upvotes_per_min']}/min) | 💬 {post['comments']}\n"
            
            # Enhanced AI data
            message += f"🧠 Viral Prob: {ai['viral_probability']}% | Confidence: {ai['confidence']}% ({ai['confidence_level']})\n"
            message += f"🎯 Pattern: {ai['pattern_match']} (score: {ai.get('pattern_score', 0)}) "
            
            if ai.get('pattern_samples', 0) > 0:
                message += f"| Success Rate: {ai.get('pattern_success_rate', 50)}%\n"
            else:
                message += f"| New Pattern\n"
            
            message += f"📈 Predizione → {ai['predicted_final_score']:,} upvotes in {ai['predicted_peak_hours']:.1f}h\n"
            message += f"⚡ Velocità: {ai['velocity_raw']}/min | Sentiment: {ai['sentiment_analysis']['intensity']}\n"
            
            if ai.get('reasoning'):
                message += f"💭 {ai['reasoning']}\n"
            
            message += f"📍 r/{post['subreddit']} | ⏱️ {post['minutes_ago']} min fa\n"
            message += f"🔗 {post['url']}\n"
        
        # Enhanced AI Summary
        if insights.get('best_patterns'):
            top_pattern = insights['best_patterns'][0]
            message += f"\n🏆 Best Performing Pattern: {top_pattern[0]} ({top_pattern[1]*100:.0f}% accuracy)\n"
        
        total_pred = insights.get('total_predictions', 0)
        message += f"📚 Total Predictions Tracked: {total_pred} | Learning Status: Active\n"
        message += f"⚡ Enhanced Gradient Learning AI v3.0 | Adaptive & Self-Improving"
        
        return message
    
    async def send_to_telegram(self, message):
        """📤 Invia a Telegram"""
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
    
    async def run_enhanced_hunter(self):
        """🚀 MAIN LOOP Enhanced"""
        logger.info("🚀 Avvio Enhanced Viral News Hunter...")
        logger.info("🧠 Enhanced Gradient Learning AI con Momentum & Adaptive Rates")
        logger.info("⏰ Scansione ogni 20 minuti + Enhanced Learning ogni ora")
        
        if not await self.initialize():
            return
        
        logger.info("✅ Enhanced Hunter operativo!")
        
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                
                await self.get_active_chats()
                
                # Enhanced Learning check ogni 3 cicli (ogni ora)
                if cycle_count % 3 == 0:
                    logger.info("🧠 Enhanced Learning check...")
                    await self.enhanced_ai.check_and_learn_enhanced(self.reddit)
                
                # Hunt viral news
                logger.info("🔍 Enhanced viral hunt...")
                viral_data = await self.hunt_viral_news_enhanced()
                
                if viral_data and viral_data['viral_posts']:
                    new_viral = [p for p in viral_data['viral_posts'] if p['id'] not in self.sent_posts]
                    
                    if new_viral and self.active_chats:
                        for post in new_viral:
                            self.sent_posts.add(post['id'])
                        
                        viral_data['viral_posts'] = new_viral
                        message = self.format_enhanced_message(viral_data)
                        success = await self.send_to_telegram(message)
                        
                        if success:
                            logger.info(f"🔥 {len(new_viral)} Enhanced predictions inviati!")
                            
                            for post in new_viral:
                                ai = post['ai_prediction']
                                logger.info(
                                    f"  🧠 {ai['pattern_match']}: {ai['viral_probability']}% "
                                    f"(conf: {ai['confidence']}%) → {ai['predicted_final_score']} | "
                                    f"{post['title'][:30]}..."
                                )
                
                # Pulizia cache
                if len(self.sent_posts) > 800:
                    self.sent_posts.clear()
                
                # Stats logging ogni 6 cicli (ogni 2 ore)
                if cycle_count % 6 == 0:
                    insights = self.enhanced_ai.get_learning_insights()
                    logger.info(f"📊 Enhanced AI Stats: {insights.get('total_predictions', 0)} predictions, "
                              f"{insights.get('overall_accuracy', 0):.1f}% accuracy")
                
                await asyncio.sleep(1200)  # 20 minuti
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Errore main loop: {e}")
                await asyncio.sleep(180)
        
        if self.reddit:
            await self.reddit.close()

async def main():
    """Main function"""
    try:
        bot = EnhancedViralNewsHunter()
        await bot.run_enhanced_hunter()
    except Exception as e:
        logger.error(f"Errore critico: {e}")

if __name__ == "__main__":
    logger.info("🚀 Enhanced Gradient Learning AI Viral News Hunter v3.0")
    asyncio.run(main())
