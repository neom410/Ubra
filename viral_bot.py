import os
import requests
import praw
from datetime import datetime, timedelta
import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
import json
from collections import defaultdict
import math

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedditTrendPredictor:
    def __init__(self):
        # Carica variabili d'ambiente
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'RedditTrendBot v2.0')
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        
        if not all([self.reddit_client_id, self.reddit_client_secret, self.telegram_bot_token]):
            raise ValueError("Variabili d'ambiente mancanti")
        
        # Inizializza Reddit
        self.reddit = praw.Reddit(
            client_id=self.reddit_client_id,
            client_secret=self.reddit_client_secret,
            user_agent=self.reddit_user_agent
        )
        
        # Cache
        self.chat_id = None
        self.historical_data = defaultdict(list)  # {post_id: [snapshots]}
        self.keyword_performance = defaultdict(lambda: {'viral_count': 0, 'total_count': 0})
        self.subreddit_stats = defaultdict(lambda: {'avg_viral_score': 0, 'post_count': 0})
        
    def get_telegram_chat_id(self):
        """Recupera automaticamente il chat ID dal bot"""
        if self.chat_id:
            return self.chat_id
            
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/getUpdates"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data['ok'] and data['result']:
                last_update = data['result'][-1]
                self.chat_id = last_update['message']['chat']['id']
                logger.info(f"Chat ID trovato: {self.chat_id}")
                return self.chat_id
            else:
                logger.warning("Nessun messaggio ricevuto dal bot. Invia un messaggio al bot prima.")
                return None
                
        except Exception as e:
            logger.error(f"Errore nel recupero chat ID: {e}")
            return None
    
    def calculate_velocity(self, post):
        """Calcola la velocità di crescita del post"""
        age_hours = (datetime.now() - datetime.fromtimestamp(post.created_utc)).total_seconds() / 3600
        if age_hours < 0.1:
            age_hours = 0.1
        
        score_velocity = post.score / age_hours
        comment_velocity = post.num_comments / age_hours
        
        return score_velocity, comment_velocity
    
    def extract_keywords(self, title):
        """Estrae keyword significative dal titolo"""
        # Parole comuni da ignorare
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'this', 'that', 'these', 'those'}
        
        words = title.lower().split()
        keywords = [w.strip('.,!?()[]{}') for w in words if len(w) > 3 and w not in stopwords]
        return keywords[:5]  # Prime 5 keyword
    
    def calculate_virality_score(self, post, historical_snapshots=None):
        """Calcola uno score predittivo di viralità"""
        # Metriche base
        score_velocity, comment_velocity = self.calculate_velocity(post)
        
        # Età del post in ore
        age_hours = (datetime.now() - datetime.fromtimestamp(post.created_utc)).total_seconds() / 3600
        
        # Ratio commenti/score (engagement)
        engagement_ratio = (post.num_comments / (post.score + 1)) * 100
        
        # Fattore temporale (post giovani hanno più potenziale)
        time_factor = max(0, 1 - (age_hours / 24))
        
        # Score base
        base_score = (score_velocity * 0.4 + comment_velocity * 0.3 + engagement_ratio * 0.3) * time_factor
        
        # Bonus da keyword storiche
        keywords = self.extract_keywords(post.title)
        keyword_bonus = 0
        for kw in keywords:
            if kw in self.keyword_performance:
                perf = self.keyword_performance[kw]
                if perf['total_count'] > 0:
                    viral_rate = perf['viral_count'] / perf['total_count']
                    keyword_bonus += viral_rate * 10
        
        # Bonus da storico subreddit
        subreddit_bonus = 0
        sr_name = post.subreddit.display_name
        if sr_name in self.subreddit_stats:
            sr_stats = self.subreddit_stats[sr_name]
            if sr_stats['post_count'] > 5:
                subreddit_bonus = sr_stats['avg_viral_score'] * 0.2
        
        # Accelerazione (se abbiamo dati storici)
        acceleration_bonus = 0
        if historical_snapshots and len(historical_snapshots) >= 2:
            # Calcola se il post sta accelerando
            recent_growth = historical_snapshots[-1]['score'] - historical_snapshots[-2]['score']
            time_diff = historical_snapshots[-1]['timestamp'] - historical_snapshots[-2]['timestamp']
            if time_diff > 0:
                acceleration = recent_growth / time_diff
                acceleration_bonus = min(acceleration * 5, 20)
        
        final_score = base_score + keyword_bonus + subreddit_bonus + acceleration_bonus
        
        return {
            'virality_score': final_score,
            'score_velocity': score_velocity,
            'comment_velocity': comment_velocity,
            'engagement_ratio': engagement_ratio,
            'time_factor': time_factor,
            'keyword_bonus': keyword_bonus,
            'subreddit_bonus': subreddit_bonus,
            'acceleration_bonus': acceleration_bonus
        }
    
    def update_historical_data(self, post, virality_metrics):
        """Aggiorna i dati storici per tracking"""
        post_id = post.id
        
        snapshot = {
            'timestamp': datetime.now().timestamp(),
            'score': post.score,
            'comments': post.num_comments,
            'virality_score': virality_metrics['virality_score']
        }
        
        self.historical_data[post_id].append(snapshot)
        
        # Mantieni solo ultimi 10 snapshot per post (ottimizzazione memoria)
        if len(self.historical_data[post_id]) > 10:
            self.historical_data[post_id] = self.historical_data[post_id][-10:]
        
        # Pulizia post vecchi (oltre 24 ore)
        current_time = datetime.now().timestamp()
        to_remove = []
        for pid, snapshots in self.historical_data.items():
            if snapshots and (current_time - snapshots[-1]['timestamp']) > 86400:
                to_remove.append(pid)
        
        for pid in to_remove:
            del self.historical_data[pid]
    
    def update_learning_data(self, post, virality_metrics):
        """Aggiorna le statistiche di apprendimento"""
        is_viral = virality_metrics['virality_score'] > 50
        
        # Aggiorna performance keyword
        keywords = self.extract_keywords(post.title)
        for kw in keywords:
            self.keyword_performance[kw]['total_count'] += 1
            if is_viral:
                self.keyword_performance[kw]['viral_count'] += 1
        
        # Aggiorna statistiche subreddit
        sr_name = post.subreddit.display_name
        self.subreddit_stats[sr_name]['post_count'] += 1
        
        # Media mobile delle performance del subreddit
        current_avg = self.subreddit_stats[sr_name]['avg_viral_score']
        count = self.subreddit_stats[sr_name]['post_count']
        new_avg = ((current_avg * (count - 1)) + virality_metrics['virality_score']) / count
        self.subreddit_stats[sr_name]['avg_viral_score'] = new_avg
    
    def get_reddit_trends(self):
        """Analizza le tendenze su Reddit con predizione"""
        trends = []
        subreddits = ['all', 'popular', 'worldnews', 'technology']  # Ridotto per ottimizzare
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Analizza solo top 30 post per subreddit (ottimizzazione)
                for post in subreddit.hot(limit=30):
                    post_time = datetime.fromtimestamp(post.created_utc)
                    time_diff = (datetime.now() - post_time).total_seconds() / 60
                    
                    # Post degli ultimi 120 minuti
                    if time_diff <= 120:
                        # Recupera snapshot storici
                        historical = self.historical_data.get(post.id, [])
                        
                        # Calcola metriche di viralità
                        virality_metrics = self.calculate_virality_score(post, historical)
                        
                        # Aggiorna dati storici e di apprendimento
                        self.update_historical_data(post, virality_metrics)
                        self.update_learning_data(post, virality_metrics)
                        
                        trends.append({
                            'title': post.title[:200],
                            'subreddit': post.subreddit.display_name,
                            'score': post.score,
                            'comments': post.num_comments,
                            'url': f"https://reddit.com{post.permalink}",
                            'created_utc': post.created_utc,
                            'age_minutes': int(time_diff),
                            'virality_score': virality_metrics['virality_score'],
                            'score_velocity': virality_metrics['score_velocity'],
                            'prediction': self.get_prediction_label(virality_metrics['virality_score'])
                        })
                        
            except Exception as e:
                logger.error(f"Errore nell'analisi di r/{subreddit_name}: {e}")
                continue
        
        # Ordina per virality score
        trends.sort(key=lambda x: x['virality_score'], reverse=True)
        return trends[:10]
    
    def get_prediction_label(self, score):
        """Restituisce etichetta predittiva"""
        if score > 80:
            return "🔥 VIRALE"
        elif score > 60:
            return "📈 ALTA CRESCITA"
        elif score > 40:
            return "⚡ CRESCITA"
        elif score > 20:
            return "📊 MODERATO"
        else:
            return "📉 BASSO"
    
    def format_alert_message(self, trends):
        """Formatta il messaggio con predizioni"""
        if not trends:
            return "🔍 Nessuna tendenza significativa rilevata"
        
        message = "🤖 **REDDIT TREND PREDICTOR** 🤖\n"
        message += f"📊 Analisi predittiva basata su pattern storici\n\n"
        
        for i, trend in enumerate(trends[:5], 1):
            emoji = "🔥" if trend['virality_score'] > 60 else "📈" if trend['virality_score'] > 40 else "📊"
            
            message += f"{emoji} **#{i} {trend['title'][:80]}...**\n"
            message += f"   🎯 Predizione: {trend['prediction']}\n"
            message += f"   📊 Score Viralità: {trend['virality_score']:.1f}/100\n"
            message += f"   ⚡ Velocità: {trend['score_velocity']:.1f} pts/h\n"
            message += f"   💬 {trend['comments']} commenti | 👍 {trend['score']} upvotes\n"
            message += f"   ⏱️ Età: {trend['age_minutes']} min | 📍 r/{trend['subreddit']}\n"
            message += f"   🔗 [Vai al post]({trend['url']})\n\n"
        
        # Statistiche di apprendimento
        total_keywords = len(self.keyword_performance)
        total_subreddits = len(self.subreddit_stats)
        
        message += f"🧠 **Apprendimento Attivo**\n"
        message += f"   • {total_keywords} keyword analizzate\n"
        message += f"   • {total_subreddits} subreddit monitorati\n"
        message += f"   • {len(self.historical_data)} post tracciati\n\n"
        
        message += f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
        return message
    
    def send_telegram_alert(self, message):
        """Invia alert su Telegram"""
        chat_id = self.get_telegram_chat_id()
        if not chat_id:
            logger.error("Chat ID non disponibile")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, json=payload, timeout=10)
            if response.json().get('ok', False):
                logger.info("✅ Alert inviato con successo")
                return True
            else:
                logger.error("❌ Errore nell'invio Telegram")
                return False
                
        except Exception as e:
            logger.error(f"❌ Errore nell'invio messaggio Telegram: {e}")
            return False
    
    def check_and_alert(self):
        """Esegue il controllo e invia l'alert"""
        logger.info("🔍 Avvio analisi predittiva tendenze Reddit...")
        
        try:
            trends = self.get_reddit_trends()
            message = self.format_alert_message(trends)
            
            if self.send_telegram_alert(message):
                logger.info(f"✅ Analisi completata - {len(trends)} trend analizzati")
            else:
                logger.error("❌ Errore nell'invio dell'alert")
                
        except Exception as e:
            logger.error(f"❌ Errore durante il controllo: {e}")

def main():
    try:
        bot = RedditTrendPredictor()
        
        # Configura scheduler
        scheduler = BlockingScheduler()
        
        # Esegui immediatamente e poi ogni 15 minuti
        bot.check_and_alert()
        scheduler.add_job(
            bot.check_and_alert,
            trigger=IntervalTrigger(minutes=15),
            id='reddit_trend_prediction',
            name='Reddit Trend Predictor - ogni 15 minuti'
        )
        
        logger.info("🤖 Reddit Trend Predictor avviato - Analisi ogni 15 minuti")
        logger.info("📊 Sistema di apprendimento attivo per predizioni accurate")
        scheduler.start()
        
    except Exception as e:
        logger.error(f"❌ Errore nell'avvio del bot: {e}")

if __name__ == "__main__":
    main()
