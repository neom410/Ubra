import asyncpraw
import asyncio
import time
import os
from datetime import datetime, timedelta
import logging
from collections import Counter
import aiohttp
import re

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ViralNewsHunter:
    def __init__(self):
        # Legge le credenziali dalle variabili d'ambiente di Render
        self.reddit_client_id = os.environ.get('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
        self.telegram_token = os.environ.get('TELEGRAM_BOT_TOKEN')
        
        # Verifica che tutte le variabili siano presenti
        if not all([self.reddit_client_id, self.reddit_client_secret, self.telegram_token]):
            raise ValueError("❌ Variabili d'ambiente mancanti! Controlla su Render.com → Settings → Environment")
        
        # Variabili per il tracking delle chat
        self.active_chats = set()
        self.reddit = None
        self.sent_posts = set()  # Per evitare duplicati
        
        # SUBREDDIT per notizie virali e breaking news
        self.viral_subreddits = [
            # Breaking news generali
            'news', 'worldnews', 'breakingnews', 'nottheonion', 'offbeat',
            
            # Tech e innovazione
            'technology', 'gadgets', 'Futurology', 'singularity', 'artificial',
            'MachineLearning', 'cryptocurrency', 'bitcoin', 'ethereum',
            
            # Business e finanza
            'business', 'economics', 'stocks', 'wallstreetbets', 'investing',
            
            # Cultura e trending
            'todayilearned', 'interestingasfuck', 'nextfuckinglevel', 'Damnthatsinteresting',
            'mildlyinteresting', 'showerthoughts', 'explainlikeimfive',
            
            # Social e viral content
            'facepalm', 'publicfreakout', 'instant_regret', 'whatcouldgowrong',
            'therewasanattempt', 'crappydesign', 'assholedesign',
            
            # Science e discovery
            'science', 'space', 'physics', 'biology', 'medicine', 'health',
            
            # Entertainment
            'movies', 'television', 'gaming', 'music', 'books',
            
            # Reddit meta (spesso virale)
            'bestof', 'announcements', 'blog'
        ]
        
        # KEYWORDS che indicano viralità
        self.viral_indicators = [
            # Urgenza e breaking
            'breaking', 'urgent', 'just in', 'developing', 'live', 'happening now',
            'alert', 'emergency', 'crisis', 'scandal', 'exposed', 'leaked',
            
            # Record e superlativi
            'record', 'highest', 'lowest', 'biggest', 'smallest', 'first time',
            'never before', 'historic', 'unprecedented', 'revolutionary',
            'groundbreaking', 'game changing', 'massive', 'huge', 'enormous',
            
            # Shock e sorpresa
            'shocking', 'unbelievable', 'incredible', 'amazing', 'stunning',
            'bizarre', 'weird', 'crazy', 'insane', 'wild', 'unexpected',
            'plot twist', 'you wont believe', 'mind blowing',
            
            # Trend e virale
            'viral', 'trending', 'everyone is talking', 'internet is going crazy',
            'twitter is losing it', 'blowing up', 'everywhere',
            
            # Numeri impressionanti
            'million', 'billion', 'trillion', '%', '$', '000', 'x increase',
            'jumped', 'soared', 'plummeted', 'crashed', 'exploded',
            
            # Celebrity e figure pubbliche
            'elon musk', 'jeff bezos', 'bill gates', 'mark zuckerberg',
            'donald trump', 'joe biden', 'pope', 'queen', 'celebrity',
            
            # Tech buzzwords
            'ai', 'chatgpt', 'artificial intelligence', 'robot', 'automation',
            'tesla', 'spacex', 'apple', 'google', 'microsoft', 'meta', 'openai'
        ]
        
    async def initialize(self):
        """Inizializza le connessioni async"""
        try:
            self.reddit = asyncpraw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent='ViralNewsHunter/1.0'
            )
            logger.info("✅ Connessione Reddit: OK")
            return True
            
        except Exception as e:
            logger.error(f"❌ Errore Reddit: {e}")
            return False
    
    async def get_active_chats(self):
        """Rileva automaticamente le chat attive utilizzando gli aggiornamenti del bot"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data['ok'] and data['result']:
                            for update in data['result']:
                                if 'message' in update:
                                    chat_id = update['message']['chat']['id']
                                    self.active_chats.add(chat_id)
                                    logger.info(f"Chat rilevata: {chat_id}")
                            
                            # Marca aggiornamenti come letti
                            last_update_id = data['result'][-1]['update_id']
                            clear_url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates?offset={last_update_id + 1}"
                            await session.get(clear_url)
                        
                        return True
                    else:
                        logger.error(f"Errore Telegram API: {response.status}")
                        return False
                
        except Exception as e:
            logger.error(f"Errore nel rilevamento chat: {e}")
            return False
    
    def calculate_viral_score(self, post, subreddit, minutes_ago):
        """Calcola il potenziale virale di un post"""
        score = 0
        title_lower = post.title.lower()
        
        # Base score da upvotes e velocità
        if minutes_ago > 0:
            upvotes_per_minute = post.score / minutes_ago
            score += min(upvotes_per_minute * 2, 100)  # Max 100 punti da velocità
        
        # Bonus per numero assoluto di upvotes
        if post.score > 1000:
            score += 50
        elif post.score > 500:
            score += 30
        elif post.score > 100:
            score += 15
        
        # Bonus per commenti (engagement)
        if post.num_comments > 500:
            score += 40
        elif post.num_comments > 200:
            score += 25
        elif post.num_comments > 50:
            score += 10
        
        # Ratio upvotes/commenti (controversia = viralità)
        if post.num_comments > 0:
            ratio = post.score / post.num_comments
            if ratio < 5:  # Molti commenti vs upvotes = controverso
                score += 20
        
        # Bonus per parole chiave virali
        viral_keywords_found = 0
        for keyword in self.viral_indicators:
            if keyword in title_lower:
                score += 25
                viral_keywords_found += 1
        
        # Bonus extra per multiple keywords virali
        if viral_keywords_found > 2:
            score += 30
        
        # Bonus per subreddit ad alto potenziale virale
        high_viral_subs = ['news', 'worldnews', 'todayilearned', 'interestingasfuck', 
                          'nextfuckinglevel', 'technology', 'wallstreetbets']
        if subreddit in high_viral_subs:
            score += 20
        
        # Bonus per numeri nel titolo (statistiche = viralità)
        numbers = re.findall(r'\d+[%$]?|\d+\.\d+[%$]?', title_lower)
        score += len(numbers) * 10
        
        # Bonus per ALL CAPS (urgenza)
        caps_words = re.findall(r'\b[A-Z]{3,}\b', post.title)
        score += len(caps_words) * 5
        
        # Penalità per post troppo vecchi
        if minutes_ago > 180:  # Più di 3 ore
            score *= 0.5
        
        return int(score)
    
    def categorize_viral_post(self, title, subreddit):
        """Categorizza il tipo di notizia virale"""
        title_lower = title.lower()
        
        # Tech e innovazione
        if any(word in title_lower for word in ['ai', 'robot', 'tech', 'innovation', 'breakthrough']):
            if 'elon' in title_lower or 'tesla' in title_lower:
                return '🚗 ELON/TESLA'
            elif any(word in title_lower for word in ['ai', 'chatgpt', 'artificial', 'robot']):
                return '🤖 AI/TECH'
            else:
                return '💻 TECNOLOGIA'
        
        # Business e finanza
        elif any(word in title_lower for word in ['stock', 'market', 'bitcoin', 'crypto', '$', 'billion']):
            return '💰 FINANZA'
        
        # Breaking news
        elif any(word in title_lower for word in ['breaking', 'urgent', 'developing', 'crisis']):
            return '🚨 BREAKING'
        
        # Science e discovery
        elif any(word in title_lower for word in ['study', 'research', 'scientists', 'discovery']):
            return '🔬 SCIENZA'
        
        # Celebrity e personaggi pubblici  
        elif any(word in title_lower for word in ['trump', 'biden', 'celebrity', 'famous']):
            return '⭐ CELEBRITY'
        
        # Weird e bizzarro
        elif any(word in title_lower for word in ['bizarre', 'weird', 'crazy', 'unbelievable']):
            return '🤯 BIZZARRO'
        
        # Fail e fails
        elif subreddit in ['facepalm', 'therewasanattempt', 'instant_regret']:
            return '🤦 FAIL'
        
        # TIL e facts
        elif subreddit == 'todayilearned':
            return '📚 TIL'
        
        else:
            return '🔥 VIRALE'
    
    async def hunt_viral_news(self):
        """Cerca notizie che stanno diventando virali"""
        try:
            viral_posts = []
            current_time = datetime.now()
            
            for subreddit_name in self.viral_subreddits:
                try:
                    subreddit = await self.reddit.subreddit(subreddit_name)
                    
                    # Analizza i post più HOT (viralità in corso)
                    count = 0
                    async for post in subreddit.hot(limit=25):
                        count += 1
                        
                        post_time = datetime.fromtimestamp(post.created_utc)
                        minutes_ago = (current_time - post_time).total_seconds() / 60
                        
                        # Focus su post delle ultime 6 ore
                        if minutes_ago <= 360:  
                            viral_score = self.calculate_viral_score(post, subreddit_name, minutes_ago)
                            
                            # Soglia per essere considerato "virale"
                            if viral_score >= 50 and post.id not in self.sent_posts:
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
                                    'upvotes_per_min': round(post.score / max(minutes_ago, 1), 1)
                                })
                        
                        if count >= 25:
                            break
                            
                except Exception as e:
                    logger.warning(f"Errore nel subreddit {subreddit_name}: {e}")
                    continue
            
            # Ordina per viral_score
            viral_posts.sort(key=lambda x: x['viral_score'], reverse=True)
            
            return {
                'viral_posts': viral_posts[:10],  # Top 10 più virali
                'timestamp': current_time
            }
            
        except Exception as e:
            logger.error(f"Errore nella caccia virale: {e}")
            return None
    
    def format_viral_message(self, data):
        """Formatta il messaggio per notizie virali"""
        if not data or not data['viral_posts']:
            return "❌ Nessuna notizia virale rilevata in questo momento."
        
        timestamp = data['timestamp'].strftime("%H:%M - %d/%m/%Y")
        
        message = f"🔥 NOTIZIE VIRALI DELL'ULTIMA ORA 🔥\n"
        message += f"⏰ Scansione: {timestamp}\n\n"
        
        # Top notizie virali
        message += "📈 TOP NOTIZIE CHE STANNO DIVENTANDO VIRALI:\n"
        
        for i, post in enumerate(data['viral_posts'][:8], 1):
            title = post['title'][:80] + "..." if len(post['title']) > 80 else post['title']
            # Pulisci caratteri problematici
            title = title.replace('[', '').replace(']', '').replace('*', '').replace('_', '')
            
            message += f"\n{post['category']} {i}. {title}\n"
            message += f"🚀 Viral Score: {post['viral_score']} | "
            message += f"👍 {post['score']} ({post['upvotes_per_min']}/min) | "
            message += f"💬 {post['comments']}\n"
            message += f"📍 r/{post['subreddit']} | ⏱️ {post['minutes_ago']} min fa\n"
            message += f"🔗 {post['url']}\n"
        
        # Statistiche finali
        total_viral = len(data['viral_posts'])
        avg_score = sum(p['viral_score'] for p in data['viral_posts']) // max(total_viral, 1)
        
        message += f"\n📊 Trovate {total_viral} notizie virali\n"
        message += f"📈 Viral Score medio: {avg_score}\n"
        message += f"🎯 Scansionati {len(self.viral_subreddits)} subreddit"
        
        return message
    
    async def send_to_telegram(self, message):
        """Invia messaggio a tutte le chat attive"""
        if not self.active_chats:
            logger.warning("Nessuna chat attiva trovata")
            return
        
        async with aiohttp.ClientSession() as session:
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
                        else:
                            logger.error(f"Errore nell'invio alla chat {chat_id}: {response.status}")
                            if response.status in [400, 403]:
                                self.active_chats.discard(chat_id)
                                
                except Exception as e:
                    logger.error(f"Errore nell'invio alla chat {chat_id}: {e}")
    
    async def run_viral_hunter(self):
        """Esegue il viral news hunter principale"""
        if not await self.initialize():
            print("❌ Impossibile inizializzare Reddit!")
            return
            
        # Test Telegram
        test_telegram = await self.get_active_chats()
        if not test_telegram:
            print("❌ Errore con Telegram API! Controlla il token.")
            return
        
        print("✅ Bot Telegram: OK")
        print("🔥 VIRAL NEWS HUNTER avviato!")
        print("📱 Invia un messaggio al bot su Telegram per iniziare")
        print("🚀 Il bot cerca notizie che stanno diventando virali ADESSO")
        print("⏰ Scansione ogni 15 minuti per catturare il momentum virale")
        
        while True:
            try:
                # Rileva nuove chat
                await self.get_active_chats()
                
                # Se ci sono chat attive, cerca viral news
                if self.active_chats:
                    print(f"🔍 Cacciando notizie virali... (Chat attive: {len(self.active_chats)})")
                    viral_data = await self.hunt_viral_news()
                    
                    if viral_data and viral_data['viral_posts']:
                        # Filtra solo i post non ancora inviati
                        new_viral = [p for p in viral_data['viral_posts'] if p['id'] not in self.sent_posts]
                        
                        if new_viral:
                            # Aggiorna la lista dei post inviati
                            for post in new_viral:
                                self.sent_posts.add(post['id'])
                            
                            # Aggiorna i dati con solo i nuovi post
                            viral_data['viral_posts'] = new_viral
                            
                            message = self.format_viral_message(viral_data)
                            await self.send_to_telegram(message)
                            print(f"🔥 Inviate {len(new_viral)} notizie virali!")
                            
                            # Mostra anteprima
                            print("📋 Top viral trovate:")
                            for post in new_viral[:3]:
                                print(f"  {post['category']}: {post['title'][:50]}... (Score: {post['viral_score']})")
                        else:
                            print("⚠️ Nessuna nuova notizia virale (già inviate)")
                    else:
                        print("⚠️ Nessuna notizia virale rilevata")
                else:
                    print("⏳ In attesa di chat attive... (invia un messaggio al bot)")
                
                # Pulisci la cache ogni tanto per evitare memory leak
                if len(self.sent_posts) > 1000:
                    self.sent_posts.clear()
                    print("🧹 Cache pulita")
                
                # Attendi 15 minuti per la prossima scansione
                print("⏱️ Prossima caccia virale tra 15 minuti...\n")
                await asyncio.sleep(900)  # 15 minuti
                
            except KeyboardInterrupt:
                print("\n🛑 Viral Hunter fermato dall'utente")
                break
            except Exception as e:
                logger.error(f"Errore nel ciclo principale: {e}")
                await asyncio.sleep(180)  # Attendi 3 minuti prima di riprovare
        
        # Chiudi la connessione Reddit
        if self.reddit:
            await self.reddit.close()

async def main():
    """Funzione principale per Render.com"""
    try:
        print("🚀 Avvio Viral News Hunter...")
        print("📖 Lettura credenziali dalle variabili d'ambiente...")
        
        # Crea e avvia il bot
        bot = ViralNewsHunter()
        await bot.run_viral_hunter()
        
    except Exception as e:
        print(f"❌ Errore critico: {e}")
        print("🔧 Controlla le variabili d'ambiente su Render.com")

# Avvia il bot quando eseguito direttamente
if __name__ == "__main__":
    asyncio.run(main())
