import os
import requests

TELEGRAM_TOKEN   = os.environ.get('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

def send_alert(message: str, image_bytes: bytes = None):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(f'[Telegram] Sin configurar — mensaje: {message}')
        return
    try:
        base = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}'
        if image_bytes:
            # Enviar foto con el mensaje como caption
            requests.post(f'{base}/sendPhoto', data={
                'chat_id':    TELEGRAM_CHAT_ID,
                'caption':    message,
                'parse_mode': 'HTML',
            }, files={
                'photo': ('frame.jpg', image_bytes, 'image/jpeg'),
            }, timeout=15)
        else:
            requests.post(f'{base}/sendMessage', json={
                'chat_id':    TELEGRAM_CHAT_ID,
                'text':       message,
                'parse_mode': 'HTML',
            }, timeout=10)
        print(f'[Telegram] Alerta enviada: {message}')
    except Exception as e:
        print(f'[Telegram] Error al enviar: {e}')