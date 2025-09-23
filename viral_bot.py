import asyncpraw
import asyncio
import time
import os
from datetime import datetime, timedelta
import logging
from collections import Counter, defaultdict
import aiohttp
import re
import json
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

# Configurazione logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('trend_bot.log')]
)
logger = logging.getLogger(__name__)

# ===== ENHANCED EXPONENTIAL GROWTH ANALYZER =====
class ExponentialGrowthAnalyzer:
    def __init__(self):
        self.growth_history_file = 'growth_history.json'
        self.trend_patterns_file = 'trend_patterns.json'
        self.growth_history = self.load_growth_history()
        self.trend_patterns = self.load_trend_patterns()
        
        # Parametri per analisi esponenziale
        self.min_data_points = 3
        self.exponential_threshold = 1.5  # Soglia per crescita esponenziale
        self.trend_confirmation_time = 45  # minuti per confermare trend
        self.viral_threshold = 2000  # score minimo per considerare virale
        
    def load_growth_history(self) -> Dict:
        """Carica storico crescite"""
        try:
            if os.path.exists(self.growth_history_file):
                with open(self.growth_history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Errore caricamento growth history: {e}")
        return {}
    
    def save_growth_history(self):
        """Salva storico crescite"""
        try:
            # Mantieni solo ultimi 7 giorni di dati
            cutoff = datetime.now() - timedelta(days=7)
            cleaned_history = {}
            
            for post_id, data in self.growth_history.items():
                if 'data_points' in data and data['data_points']:
                    last_timestamp = datetime.fromisoformat(data['data_points'][-1]['timestamp'])
                    if last_timestamp > cutoff:
                        cleaned_history[post_id] = data
            
            self.growth_history = cleaned_history
            
            with open(self.growth_history_file, 'w') as f:
                json.dump(self.growth_history, f, indent=2)
        except Exception as e:
            logger.error(f"Errore salvataggio growth history: {e}")
    
    def load_trend_patterns(self) -> Dict:
        """Carica pattern di trend identificati"""
        try:
            if os.path.exists(self.trend_patterns_file):
                with open(self.trend_patterns_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Errore caricamento trend patterns: {e}")
        return {
            'successful_patterns': [],
            'failed_patterns': [],
            'keyword_weights': {},
            'subreddit_multipliers': {}
        }
    
    def save_trend_patterns(self):
        """Salva pattern di trend"""
        try:
            with open(self.trend_patterns_file, 'w') as f:
                json.dump(self.trend_patterns, f, indent=2)
        except Exception as e:
            logger.error(f"Errore salvataggio trend patterns: {e}")
    
    def track_post_growth(self, post_id: str, score: int, comments: int, subreddit: str, title: str):
        """Traccia la crescita di un post nel tempo"""
        current_time = datetime.now()
        
        if post_id not in self.growth_history:
            self.growth_history[post_id] = {
                'subreddit': subreddit,
                'title': title,
                'data_points': [],
                'trend_status': 'monitoring',
                'first_seen': current_time.isoformat()
            }
        
        # Aggiungi nuovo data point
        data_point = {
            'timestamp': current_time.isoformat(),
            'score': score,
            'comments': comments,
            'minutes_since_start': 0
        }
        
        # Calcola minuti dal primo avvistamento
        first_seen = datetime.fromisoformat(self.growth_history[post_id]['first_seen'])
        data_point['minutes_since_start'] = (current_time - first_seen).total_seconds() / 60
        
        self.growth_history[post_id]['data_points'].append(data_point)
        
        # Mantieni solo ultimi 20 data points per efficienza
        if len(self.growth_history[post_id]['data_points']) > 20:
            self.growth_history[post_id]['data_points'] = self.growth_history[post_id]['data_points'][-20:]
    
    def analyze_exponential_growth(self, post_id: str) -> Optional[Dict]:
        """Analizza se un post sta crescendo esponenzialmente"""
        if post_id not in self.growth_history:
            return None
        
        data_points = self.growth_history[post_id]['data_points']
        if len(data_points) < self.min_data_points:
            return None
        
        # Estrai dati per analisi
        times = [dp['minutes_since_start'] for dp in data_points]
        scores = [dp['score'] for dp in data_points]
        
        if len(times) < 3 or max(scores) < 50:  # Dati insufficienti
            return None
        
        try:
            # Calcola tasso di crescita tra punti consecutivi
            growth_rates = []
            for i in range(1, len(scores)):
                if scores[i-1] > 0 and times[i] != times[i-1]:
                    rate = (scores[i] - scores[i-1]) / (times[i] - times[i-1])
                    growth_rates.append(max(rate, 0))
            
            if not growth_rates:
                return None
            
            # Analizza accelerazione della crescita
            acceleration = 0
            if len(growth_rates) >= 2:
                recent_avg = sum(growth_rates[-2:]) / 2
                early_avg = sum(growth_rates[:2]) / 2 if len(growth_rates) > 2 else growth_rates[0]
                if early_avg > 0:
                    acceleration = recent_avg / early_avg
            
            # Calcola velocità media recente
            recent_velocity = sum(growth_rates[-3:]) / min(3, len(growth_rates))
            
            # Stima crescita esponenziale usando regressione semplificata
            exponential_score = 0
            
            # Controlla se la crescita accelera
            if acceleration > self.exponential_threshold:
                exponential_score += 30
            
            # Controlla velocità assoluta
            if recent_velocity > 10:  # >10 upvotes/minuto
                exponential_score += 40
            elif recent_velocity > 5:
                exponential_score += 20
            
            # Controlla consistenza della crescita
            if len(growth_rates) >= 3:
                consistency = 1 - (np.std(growth_rates[-3:]) / (np.mean(growth_rates[-3:]) + 1))
                exponential_score += consistency * 30
            
            # Predici score futuro (semplificato)
            if recent_velocity > 0:
                current_score = scores[-1]
                predicted_1h = current_score + (recent_velocity * 60)
                predicted_6h = current_score + (recent_velocity * 360 * 0.7)  # Decay factor
            else:
                predicted_1h = scores[-1]
                predicted_6h = scores[-1]
            
            return {
                'exponential_score': min(exponential_score, 100),
                'growth_acceleration': acceleration,
                'recent_velocity': recent_velocity,
                'current_score': scores[-1],
                'predicted_1h': int(predicted_1h),
                'predicted_6h': int(predicted_6h),
                'data_points_count': len(data_points),
                'tracking_time_minutes': times[-1],
                'is_exponential': exponential_score > 50,
                'trend_strength': 'explosive' if exponential_score > 80 else 'strong' if exponential_score > 60 else 'moderate' if exponential_score > 40 else 'weak'
            }
            
        except Exception as e:
            logger.error(f"Errore analisi esponenziale per {post_id}: {e}")
            return None

# ===== EMERGING TREND DETECTOR =====
class EmergingTrendDetector:
    def __init__(self):
        self.weights_file = 'trend_weights.json'
        self.trending_keywords_file = 'trending_keywords.json'
        
        # Carica pesi e keywords
        self.weights = self.load_weights()
        self.trending_keywords = self.load_trending_keywords()
        
        # Pattern per trend emergenti
        self.emerging_patterns = {
            'tech_disruption': {
                'keywords': ['breakthrough', 'revolutionary', 'game-changer', 'disruption', 'innovation', 'first time', 'never before'],
                'weight': 2.5
            },
            'crisis_emerging': {
                'keywords': ['crisis', 'emergency', 'urgent', 'critical', 'disaster', 'unprecedented', 'breaking'],
                'weight': 3.0
            },
            'viral_moment': {
                'keywords': ['viral', 'trending', 'everywhere', 'internet', 'meme', 'blowing up', 'taking over'],
                'weight': 2.0
            },
            'market_disruption': {
                'keywords': ['crash', 'surge', 'record', 'historic', 'all-time', 'massive', 'collapse'],
                'weight': 2.8
            },
            'social_phenomenon': {
                'keywords': ['phenomenon', 'movement', 'wave', 'spreading', 'everyone', 'millions', 'worldwide'],
                'weight': 2.2
            }
        }
        
        # Subreddit con alta probabilità di trend emergenti
        self.trend_source_subreddits = {
            'technology': 1.5,
            'Futurology': 1.4,
            'singularity': 1.6,
            'artificial': 1.3,
            'MachineLearning': 1.2,
            'cryptocurrency': 1.8,
            'wallstreetbets': 2.0,
            'science': 1.3,
            'space': 1.4,
            'news': 1.2,
            'worldnews': 1.3,
            'breakingnews': 2.5,
            'todayilearned': 1.1,
            'Showerthoughts': 1.0,
            'interestingasfuck': 1.2,
            'nextfuckinglevel': 1.3,
            'Damnthatsinteresting': 1.2
        }
    
    def load_weights(self) -> Dict:
        """Carica pesi per trend detection"""
        default_weights = {
            'exponential_weight': 0.4,
            'velocity_weight': 0.25,
            'engagement_weight': 0.2,
            'pattern_weight': 0.15,
            'time_penalty': 0.1,
            'subreddit_bonus': 0.1,
            'keyword_density_weight': 0.3,
            'comment_ratio_optimal': 0.05  # 5% comment ratio è ottimale
        }
        
        try:
            if os.path.exists(self.weights_file):
                with open(self.weights_file, 'r') as f:
                    saved_weights = json.load(f)
                    default_weights.update(saved_weights)
        except Exception as e:
            logger.warning(f"Errore caricamento trend weights: {e}")
        
        return default_weights
    
    def load_trending_keywords(self) -> Dict:
        """Carica keywords trending dinamiche"""
        try:
            if os.path.exists(self.trending_keywords_file):
                with open(self.trending_keywords_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Errore caricamento trending keywords: {e}")
        
        return {
            'hot_keywords': {},
            'last_update': datetime.now().isoformat(),
            'keyword_performance': {}
        }
    
    def analyze_emerging_trend_potential(self, post, subreddit_name: str, growth_analysis: Dict) -> Dict:
        """Analizza il potenziale di trend emergente"""
        title = post.title.lower()
        
        # 1. Score base da crescita esponenziale
        exponential_score = growth_analysis.get('exponential_score', 0)
        
        # 2. Analisi pattern di trend emergente
        pattern_score = 0
        matched_patterns = []
        
        for pattern_name, pattern_data in self.emerging_patterns.items():
            pattern_matches = sum(1 for keyword in pattern_data['keywords'] if keyword in title)
            if pattern_matches > 0:
                pattern_score += pattern_matches * pattern_data['weight'] * 10
                matched_patterns.append(pattern_name)
        
        # 3. Bonus subreddit
        subreddit_multiplier = self.trend_source_subreddits.get(subreddit_name, 1.0)
        
        # 4. Analisi engagement quality
        if post.score > 0:
            comment_ratio = post.num_comments / post.score
            # Ratio ottimale intorno al 5%
            optimal_ratio = self.weights['comment_ratio_optimal']
            engagement_quality = 1 - abs(comment_ratio - optimal_ratio) / optimal_ratio
            engagement_quality = max(0, min(engagement_quality, 1))
        else:
            engagement_quality = 0
        
        # 5. Velocity score
        velocity_score = min(growth_analysis.get('recent_velocity', 0) * 5, 100)
        
        # 6. Time factor (più recente = meglio per trend emergenti)
        tracking_time = growth_analysis.get('tracking_time_minutes', 0)
        if tracking_time < 60:  # Meno di 1 ora
            time_factor = 1.2
        elif tracking_time < 180:  # Meno di 3 ore
            time_factor = 1.0
        else:
            time_factor = 0.8
        
        # Calcola score finale
        final_score = (
            exponential_score * self.weights['exponential_weight'] +
            velocity_score * self.weights['velocity_weight'] +
            engagement_quality * 100 * self.weights['engagement_weight'] +
            pattern_score * self.weights['pattern_weight']
        ) * subreddit_multiplier * time_factor
        
        # Determina livello di trend
        if final_score >= 80:
            trend_level = "🚀 EXPLOSIVE EMERGING"
            confidence = "ALTISSIMA"
        elif final_score >= 65:
            trend_level = "⚡ STRONG EMERGING"
            confidence = "ALTA"
        elif final_score >= 50:
            trend_level = "📈 MODERATE EMERGING"
            confidence = "MEDIA"
        elif final_score >= 35:
            trend_level = "📊 WEAK EMERGING"
            confidence = "BASSA"
        else:
            trend_level = "📱 MONITORING"
            confidence = "MOLTO BASSA"
        
        return {
            'trend_score': min(final_score, 100),
            'trend_level': trend_level,
            'confidence': confidence,
            'exponential_component': exponential_score,
            'pattern_component': pattern_score,
            'velocity_component': velocity_score,
            'engagement_quality': round(engagement_quality * 100, 1),
            'subreddit_multiplier': subreddit_multiplier,
            'time_factor': time_factor,
            'matched_patterns': matched_patterns,
            'is_emerging_trend': final_score >= 50,
            'predicted_viral_potential': growth_analysis.get('predicted_6h', 0) > self.weights.get('viral_threshold', 2000)
        }

# ===== ENHANCED VIRAL TREND HUNTER =====
class ViralTrendHunter:
    def __init__(self):
        # Credenziali
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        
        if not all([self.reddit_client_id, self.reddit_client_secret, self.telegram_token]):
            raise ValueError("❌ Variabili d'ambiente mancanti! Aggiungi REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, TELEGRAM_BOT_TOKEN")
        
        # Componenti AI
        self.growth_analyzer = ExponentialGrowthAnalyzer()
        self.trend_detector = EmergingTrendDetector()
        
        # State management
        self.active_chats = set()
        self.reddit = None
        self.sent_alerts = set()
        self.last_cleanup = datetime.now()
        
        # Subreddit ottimizzati per trend emergenti
        self.trend_subreddits = [
            # News e Breaking
            'news', 'worldnews', 'breakingnews', 'nottheonion',
            # Tech e Innovation
            'technology', 'gadgets', 'Futurology', 'singularity', 'artificial',
            'MachineLearning', 'OpenAI', 'ChatGPT',
            # Finance e Crypto
            'cryptocurrency', 'bitcoin', 'ethereum', 'wallstreetbets', 'investing',
            'stocks', 'business', 'economics',
            # Science e Discovery
            'science', 'space', 'physics', 'biology', 'medicine',
            # Social e Viral
            'todayilearned', 'interestingasfuck', 'nextfuckinglevel',
            'Damnthatsinteresting', 'mildlyinteresting',
            # Culture e Entertainment
            'movies', 'television', 'gaming', 'music',
            # Discussioni
            'explainlikeimfive', 'Showerthoughts', 'unpopularopinion'
        ]
    
    async def initialize(self):
        """Inizializza connessioni"""
        try:
            self.reddit = asyncpraw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent='EmergingTrendHunter/3.0'
            )
            logger.info("✅ Reddit connesso - Trend Hunter attivo")
            return True
        except Exception as e:
            logger.error(f"❌ Errore Reddit: {e}")
            return False
    
    async def get_active_chats(self):
        """Rileva e gestisce chat attive"""
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
                                        logger.info(f"📱 Nuova chat attiva: {chat_id}")
                            
                            # Pulisci updates
                            if data['result']:
                                last_update_id = data['result'][-1]['update_id']
                                clear_url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates?offset={last_update_id + 1}"
                                await session.get(clear_url)
                            
                            if new_chats > 0:
                                logger.info(f"📊 {new_chats} nuove chat. Totale attive: {len(self.active_chats)}")
                        
                        return len(self.active_chats) > 0
                    else:
                        logger.error(f"Errore Telegram API: {response.status}")
                        return False
                
        except Exception as e:
            logger.error(f"Errore gestione chat: {e}")
            return False
    
    async def scan_for_emerging_trends(self):
        """🔍 Scansiona per trend emergenti"""
        try:
            trending_posts = []
            scanned_count = 0
            
            for subreddit_name in self.trend_subreddits:
                try:
                    subreddit = await self.reddit.subreddit(subreddit_name)
                    
                    # Scansiona hot + rising per massima copertura
                    async for post in subreddit.hot(limit=15):
                        await self._analyze_post(post, subreddit_name, trending_posts)
                        scanned_count += 1
                    
                    async for post in subreddit.rising(limit=10):
                        await self._analyze_post(post, subreddit_name, trending_posts)
                        scanned_count += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️ Errore scansione r/{subreddit_name}: {e}")
                    continue
            
            # Ordina per trend score
            trending_posts.sort(key=lambda x: x['trend_analysis']['trend_score'], reverse=True)
            
            logger.info(f"🔍 Scansionati {scanned_count} post, trovati {len(trending_posts)} trend emergenti")
            
            return {
                'trending_posts': trending_posts[:10],  # Top 10
                'total_scanned': scanned_count,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ Errore scansione trend: {e}")
            return None
    
    async def _analyze_post(self, post, subreddit_name: str, trending_posts: List):
        """Analizza singolo post per trend emergente"""
        try:
            # Filtra post troppo vecchi (>6 ore)
            post_time = datetime.fromtimestamp(post.created_utc)
            hours_ago = (datetime.now() - post_time).total_seconds() / 3600
            
            if hours_ago > 6 or post.score < 20:  # Troppo vecchio o troppo pochi upvotes
                return
            
            # Traccia crescita
            self.growth_analyzer.track_post_growth(
                post.id, post.score, post.num_comments, subreddit_name, post.title
            )
            
            # Analizza crescita esponenziale
            growth_analysis = self.growth_analyzer.analyze_exponential_growth(post.id)
            
            if not growth_analysis or not growth_analysis['is_exponential']:
                return
            
            # Analizza potenziale trend emergente
            trend_analysis = self.trend_detector.analyze_emerging_trend_potential(
                post, subreddit_name, growth_analysis
            )
            
            # Filtra solo trend emergenti significativi
            if trend_analysis['is_emerging_trend'] and post.id not in self.sent_alerts:
                trending_posts.append({
                    'id': post.id,
                    'title': post.title,
                    'score': post.score,
                    'comments': post.num_comments,
                    'subreddit': subreddit_name,
                    'url': f"https://reddit.com{post.permalink}",
                    'created_time': post_time,
                    'hours_ago': round(hours_ago, 1),
                    'growth_analysis': growth_analysis,
                    'trend_analysis': trend_analysis,
                    'upvotes_per_hour': round(post.score / max(hours_ago, 0.1), 1)
                })
        
        except Exception as e:
            logger.debug(f"Errore analisi post {getattr(post, 'id', 'unknown')}: {e}")
    
    def format_trend_alert(self, data) -> str:
        """📱 Formatta alert per trend emergenti"""
        if not data or not data['trending_posts']:
            return "🔍 Nessun trend emergente rilevato al momento."
        
        timestamp = data['timestamp'].strftime("%H:%M - %d/%m/%Y")
        posts = data['trending_posts']
        
        message = f"🚀 TREND EMERGENTI RILEVATI 🚀\n"
        message += f"⏰ {timestamp} | 🔍 {data['total_scanned']} post analizzati\n"
        message += f"📊 Analisi Esponenziale + Pattern Recognition\n\n"
        
        message += f"📈 {len(posts)} TREND IN RAPIDA CRESCITA:\n"
        
        for i, post in enumerate(posts[:6], 1):  # Max 6 per evitare messaggi troppo lunghi
            # Tronca titolo
            title = post['title'][:60] + "..." if len(post['title']) > 60 else post['title']
            title = title.replace('[', '').replace(']', '').replace('*', '')
            
            growth = post['growth_analysis']
            trend = post['trend_analysis']
            
            # Header con emoji trend level
            if "EXPLOSIVE" in trend['trend_level']:
                emoji = "🚀🔥"
            elif "STRONG" in trend['trend_level']:
                emoji = "⚡📈"
            else:
                emoji = "📊🎯"
            
            message += f"\n{emoji} {i}. {title}\n"
            
            # Statistiche attuali
            message += f"📊 {post['score']} upvotes ({post['upvotes_per_hour']}/h) | 💬 {post['comments']}\n"
            
            # Analisi crescita esponenziale
            message += f"📈 Crescita: {growth['trend_strength'].upper()} "
            message += f"(Score: {growth['exponential_score']:.0f})\n"
            message += f"⚡ Velocità: {growth['recent_velocity']:.1f} upvotes/min\n"
            
            # Predizioni
            message += f"🔮 Predizione 1h: {growth['predicted_1h']:,} | 6h: {growth['predicted_6h']:,}\n"
            
            # Trend analysis
            message += f"🎯 {trend['trend_level']} ({trend['confidence']})\n"
            if trend['matched_patterns']:
                patterns = ', '.join(trend['matched_patterns'][:2])
                message += f"🔍 Pattern: {patterns}\n"
            
            # Info post
            message += f"📍 r/{post['subreddit']} | ⏱️ {post['hours_ago']}h fa\n"
            message += f"🔗 {post['url']}\n"
        
        # Summary stats
        avg_trend_score = sum(p['trend_analysis']['trend_score'] for p in posts) / len(posts)
        explosive_count = sum(1 for p in posts if "EXPLOSIVE" in p['trend_analysis']['trend_level'])
        
        message += f"\n📊 RIEPILOGO TREND:\n"
        message += f"⚡ {explosive_count} trend esplosivi | Confidence media: {avg_trend_score:.0f}%\n"
        message += f"🔍 Prossima scansione tra 20 minuti\n"
        message += f"🧠 AI Engine: Exponential Growth Analyzer v3.0"
        
        return message
    
    async def send_trend_alert(self, message: str):
        """📤 Invia alert trend su Telegram"""
        if not self.active_chats:
            logger.warning("📱 Nessuna chat attiva per invio alert")
            return False
        
        success_count = 0
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            for chat_id in list(self.active_chats):  # Copy per evitare modifiche durante iterazione
                try:
                    url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                    payload = {
                        'chat_id': chat_id,
                        'text': message,
                        'disable_web_page_preview': True,
                        'parse_mode': 'HTML'
                    }
                    
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            success_count += 1
                            logger.info(f"📤 Alert inviato a chat {chat_id}")
                        else:
                            error_text = await response.text()
                            logger.error(f"❌ Errore invio chat {chat_id}: {response.status} - {error_text}")
                            
                            # Rimuovi chat inattive
                            if response.status in [400, 403, 404]:
                                self.active_chats.discard(chat_id)
                                logger.info(f"🗑️ Rimossa chat inattiva: {chat_id}")
                                
                except Exception as e:
                    logger.error(f"❌ Errore invio chat {chat_id}: {e}")
        
        logger.info(f"📊 Alert inviato a {success_count}/{len(self.active_chats)} chat")
        return success_count > 0
    
    async def cleanup_data(self):
        """🧹 Pulizia periodica dei dati"""
        try:
            # Pulisci sent_alerts (mantieni solo ultime 24h)
            if len(self.sent_alerts) > 500:
                self.sent_alerts.clear()
                logger.info("🧹 Cache alert pulita")
            
            # Salva dati analyzer
            self.growth_analyzer.save_growth_history()
            self.trend_detector.save_weights()
            
            # Update timestamp
            self.last_cleanup = datetime.now()
            
            logger.info("🧹 Cleanup completato")
            
        except Exception as e:
            logger.error(f"❌ Errore cleanup: {e}")
    
    async def update_trend_learning(self):
        """🧠 Aggiorna apprendimento sui trend"""
        try:
            # Controlla predizioni passate e aggiorna pesi
            verified_trends = 0
            
            for post_id, data in list(self.growth_analyzer.growth_history.items()):
                if len(data['data_points']) >= 4:  # Dati sufficienti
                    first_score = data['data_points'][0]['score']
                    last_score = data['data_points'][-1]['score']
                    
                    # Se il post è cresciuto significativamente, era un vero trend
                    if last_score > first_score * 3 and last_score > 1000:
                        verified_trends += 1
                        
                        # Aggiorna pattern weights basandosi sui successi
                        title_lower = data['title'].lower()
                        for pattern_name, pattern_data in self.trend_detector.emerging_patterns.items():
                            for keyword in pattern_data['keywords']:
                                if keyword in title_lower:
                                    # Rinforza pattern di successo
                                    pattern_data['weight'] = min(pattern_data['weight'] * 1.02, 5.0)
            
            if verified_trends > 0:
                logger.info(f"🧠 Apprendimento: {verified_trends} trend verificati, pesi aggiornati")
                self.trend_detector.save_weights()
            
        except Exception as e:
            logger.error(f"❌ Errore update learning: {e}")
    
    async def run_trend_hunter(self):
        """🚀 MAIN LOOP - Trend Hunter Principale"""
        logger.info("🚀 Avvio Enhanced Trend Hunter...")
        logger.info("🎯 Focus: Anticipare fenomeni emergenti con analisi esponenziale")
        logger.info("⏰ Scansione ogni 20 minuti + apprendimento automatico")
        
        if not await self.initialize():
            logger.error("❌ Impossibile inizializzare Reddit!")
            return
        
        logger.info("✅ Trend Hunter operativo!")
        
        cycle_count = 0
        
        while True:
            try:
                cycle_start = time.time()
                cycle_count += 1
                
                logger.info(f"🔄 Ciclo {cycle_count} - Scansione trend emergenti...")
                
                # 1. Verifica chat attive
                has_chats = await self.get_active_chats()
                
                # 2. Scansiona per trend emergenti
                trend_data = await self.scan_for_emerging_trends()
                
                if trend_data and trend_data['trending_posts']:
                    new_trends = [
                        p for p in trend_data['trending_posts'] 
                        if p['id'] not in self.sent_alerts
                    ]
                    
                    if new_trends and has_chats:
                        # Aggiorna sent alerts
                        for post in new_trends:
                            self.sent_alerts.add(post['id'])
                        
                        # Invia alert solo per nuovi trend
                        trend_data['trending_posts'] = new_trends
                        message = self.format_trend_alert(trend_data)
                        
                        success = await self.send_trend_alert(message)
                        
                        if success:
                            logger.info(f"🔥 {len(new_trends)} nuovi trend emergenti inviati!")
                            
                            # Log trend details
                            for post in new_trends[:3]:  # Log primi 3
                                trend = post['trend_analysis']
                                growth = post['growth_analysis']
                                logger.info(
                                    f"  🚀 {trend['trend_level']}: {trend['trend_score']:.0f}% | "
                                    f"Velocity: {growth['recent_velocity']:.1f}/min | "
                                    f"{post['title'][:40]}..."
                                )
                        else:
                            logger.warning("⚠️ Errore invio alert trend")
                    
                    elif not has_chats:
                        logger.info("📱 Trend rilevati ma nessuna chat attiva")
                    else:
                        logger.info("🔍 Nessun nuovo trend emergente")
                else:
                    logger.info("📊 Nessun trend emergente significativo rilevato")
                
                # 3. Apprendimento e pulizia periodica
                if cycle_count % 6 == 0:  # Ogni 2 ore (6 cicli * 20 min)
                    logger.info("🧠 Aggiornamento apprendimento...")
                    await self.update_trend_learning()
                
                if cycle_count % 12 == 0:  # Ogni 4 ore
                    logger.info("🧹 Pulizia dati periodica...")
                    await self.cleanup_data()
                
                # 4. Statistiche periodiche
                if cycle_count % 18 == 0:  # Ogni 6 ore
                    total_tracked = len(self.growth_analyzer.growth_history)
                    active_trends = sum(
                        1 for data in self.growth_analyzer.growth_history.values()
                        if len(data['data_points']) >= 3
                    )
                    
                    logger.info(f"📊 Stats: {total_tracked} post tracciati, {active_trends} trend attivi")
                    logger.info(f"📱 {len(self.active_chats)} chat attive, {len(self.sent_alerts)} alert inviati")
                
                # 5. Calcola tempo di attesa
                cycle_time = time.time() - cycle_start
                wait_time = max(1200 - cycle_time, 60)  # 20 min - tempo ciclo, min 1 min
                
                logger.info(f"⏱️ Ciclo completato in {cycle_time:.1f}s. Prossima scansione tra {wait_time/60:.1f} min")
                
                # 6. Attendi prossimo ciclo
                await asyncio.sleep(wait_time)
                
            except KeyboardInterrupt:
                logger.info("🛑 Trend Hunter fermato dall'utente")
                break
                
            except Exception as e:
                logger.error(f"❌ Errore main loop: {e}")
                logger.info("🔄 Riavvio tra 3 minuti...")
                await asyncio.sleep(180)
        
        # Cleanup finale
        try:
            if self.reddit:
                await self.reddit.close()
            
            await self.cleanup_data()
            logger.info("✅ Trend Hunter terminato correttamente")
            
        except Exception as e:
            logger.error(f"Errore cleanup finale: {e}")

# ===== STARTUP E MONITORAGGIO =====
async def health_check():
    """Verifica stato servizi"""
    try:
        # Test variabili ambiente
        required_vars = ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'TELEGRAM_BOT_TOKEN']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"❌ Variabili mancanti: {', '.join(missing_vars)}")
            return False
        
        # Test connessione Telegram
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        url = f"https://api.telegram.org/bot{token}/getMe"
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"❌ Telegram bot non raggiungibile: {response.status}")
                    return False
        
        logger.info("✅ Health check superato")
        return True
        
    except Exception as e:
        logger.error(f"❌ Health check fallito: {e}")
        return False

async def main():
    """Funzione principale con gestione errori avanzata"""
    logger.info("🚀 Avvio Enhanced Reddit Trend Hunter v3.0")
    logger.info("🎯 Specializzato in rilevamento trend emergenti")
    logger.info("📊 Analisi esponenziale + Pattern recognition")
    
    # Health check iniziale
    if not await health_check():
        logger.error("❌ Health check fallito - terminazione")
        return
    
    retry_count = 0
    max_retries = 5
    
    while retry_count < max_retries:
        try:
            hunter = ViralTrendHunter()
            await hunter.run_trend_hunter()
            break  # Uscita normale
            
        except Exception as e:
            retry_count += 1
            logger.error(f"❌ Errore critico (tentativo {retry_count}/{max_retries}): {e}")
            
            if retry_count < max_retries:
                wait_time = min(300 * retry_count, 1800)  # Max 30 min
                logger.info(f"🔄 Riavvio tra {wait_time/60:.1f} minuti...")
                await asyncio.sleep(wait_time)
            else:
                logger.error("❌ Numero massimo tentativi raggiunto")
                break
    
    logger.info("🔚 Trend Hunter terminato")

# ===== ENTRY POINT =====
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 ENHANCED REDDIT TREND HUNTER v3.0")
    logger.info("🎯 Anticipazione Fenomeni Emergenti")
    logger.info("📊 Exponential Growth Analysis + AI Learning")
    logger.info("⚡ Ottimizzato per Fly.io + Piano Free")
    logger.info("=" * 60)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Terminazione forzata")
    except Exception as e:
        logger.error(f"❌ Errore fatale: {e}")
        exit(1)
