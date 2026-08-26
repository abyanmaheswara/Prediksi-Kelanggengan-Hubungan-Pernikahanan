from pyngrok import ngrok
import subprocess
import time
import sys

print("="*60)
print("🚀 MENJALANKAN STREAMLIT DENGAN NGROK TUNNEL")
print("="*60)

import os

# 1. Jalankan Streamlit
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app_path = os.path.join(BASE_DIR, "app.py")

print("\n>> Memulai server Streamlit pada port 8501...")
cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.port", "8501", "--server.headless", "true"]
process = subprocess.Popen(cmd)

# Tunggu server siap
time.sleep(3)

try:
    # 2. Buka tunnel Ngrok
    public_url = ngrok.connect(8501)
    print("\n" + "="*60)
    print("✅ BERHASIL FORWARDING KE INTERNET!")
    print(f"🔗 URL PUBLIK : {public_url}")
    print("="*60)
    print("👉 Salin URL di atas dan bagikan ke Dosen / Penguji.")
    print("ℹ️  Jika muncul halaman konfirmasi ngrok, klik tombol 'Visit Site'.")
    print("🛑 Tekan Ctrl + C untuk menghentikan server.\n")

    process.wait()
except KeyboardInterrupt:
    print("\n>> Menghentikan server dan tunnel...")
    ngrok.kill()
    process.terminate()
    print(">> Selesai!")
except Exception as e:
    print(f"\n❌ Terjadi kesalahan: {e}")
    print("Pastikan kamu sudah memasukkan authtoken ngrok:")
    print("python -c \"from pyngrok import ngrok; ngrok.set_auth_token('TOKEN_KAMU')\"")
    ngrok.kill()
    process.terminate()
