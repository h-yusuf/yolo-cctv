# yolo-cctv

Deteksi dan baca plat nomor (ALPR) dari stream CCTV publik memakai YOLO + [fast-alpr](https://github.com/ankandrew/fast-alpr).

Repo: https://github.com/h-yusuf/yolo-cctv

```
stream CCTV (HLS/RTSP/file)
  → YOLO11n + ByteTrack        deteksi + tracking kendaraan
  → zona poligon               OCR hanya di area plat paling besar (dekat garis stop)
  → fast-alpr per kendaraan    deteksi plat (YOLOv9) + OCR (CCT)
  → regex plat Indonesia       buang bacaan yang bukan format plat
  → voting per track id        ambil bacaan terbanyak, dedupe misread
  → SQLite + snapshot plat     out/plates.db, out/snapshots/
```

Overlay video: box sudut per kendaraan, label plat, zona, panel kiri berisi plat terakhir, dan FPS.

## Setup

Butuh [uv](https://docs.astral.sh/uv/) dan Python 3.12+.

```bash
git clone https://github.com/h-yusuf/yolo-cctv.git
cd yolo-cctv
uv sync
cp .env.example .env
```

Model (`yolo11n.pt`, detector plat, OCR) diunduh otomatis saat run pertama.

## Konfigurasi `.env`

| Variabel | Isi |
|---|---|
| `CCTV_URL` | URL stream (`.m3u8` / `rtsp://`) atau path file video |
| `ZONE` | Poligon `x,y x,y ...` dalam piksel sumber. Kosongkan untuk seluruh frame |

Variabel di shell menang atas `.env`, jadi bisa ganti kamera sekali jalan:

```bash
CCTV_URL=https://example.com/cam/index.m3u8 ZONE= uv run alpr.py --show
```

`ZONE` harus disesuaikan per kamera. Taruh di area plat paling besar dan tajam, biasanya sekitar garis stop. Koordinat pakai resolusi asli stream. Ambil satu frame untuk mengukurnya:

```bash
ffmpeg -i "$CCTV_URL" -frames:v 1 frame.jpg
```

### Cari URL CCTV Semarang

Portal [PANTAUSEMAR](https://pantausemar.semarangkota.go.id) menyimpan semua kamera sebagai JSON di HTML halamannya. Cari nama kameranya, lalu ambil field `url` (`https://livepantau.semarangkota.go.id/<uuid>/index.m3u8`). Bisa juga lewat DevTools → Network → filter `m3u8` saat kamera dibuka.

Kamera yang layak untuk ALPR: resolusi minimal 1080p, dekat lajur, dan plat di zona minimal sekitar 60 px lebarnya. CCTV yang dipasang tinggi dengan sudut lebar biasanya cuma cukup untuk deteksi kendaraan.

## Menjalankan

```bash
uv run alpr.py --show                               # preview window, tekan q untuk berhenti
uv run alpr.py --duration 120 --save out/demo.mp4   # 2 menit, simpan video beranotasi
uv run alpr.py --source rekaman.mp4 --save out/hasil.mp4
uv run python test_alpr.py                          # self-check
```

Opsi lain (lihat `uv run alpr.py -h`):

| Flag | Default | Fungsi |
|---|---|---|
| `--stride` | 3 | Live: proses tiap frame ke-N (25 fps / 3 ≈ 8 fps) |
| `--min-width` | 100 | Lewati OCR untuk kendaraan lebih sempit dari ini (px) |
| `--min-votes` | 2 | Jumlah bacaan sama yang dibutuhkan untuk menyimpan plat |
| `--min-conf` | 0.7 | Rata-rata confidence OCR minimum |
| `--dedupe-sec` | 60 | Lewati plat yang sama/mirip yang sudah tersimpan dalam jendela ini |
| `--plate-model` | `yolo-v9-s-608-license-plate-end2end` | Detector plat fast-alpr |
| `--ocr-model` | `cct-s-v2-global-model` | Model OCR fast-alpr |

## Output

Tabel `plates` di `out/plates.db`:

| Kolom | Isi |
|---|---|
| `plate` | Teks plat, misal `H1234AB` |
| `votes`, `conf` | Jumlah bacaan pemenang dan rata-rata confidence |
| `vehicle` | `car` / `motorcycle` / `bus` / `truck` |
| `track_id`, `source` | ID track ByteTrack dan URL sumber |
| `first_seen`, `last_seen` | Waktu kendaraan pertama/terakhir terlihat |
| `snapshot` | Path crop plat terbaik di `out/snapshots/` |

```bash
sqlite3 out/plates.db "select plate, votes, conf, vehicle, last_seen from plates order by id desc limit 20"
```

## Hasil uji (Karangayu ke timur, Mac M2)

- Sekitar 5–8 FPS dengan `--stride 3`.
- Sekitar 4–5 plat unik per menit, kebanyakan mobil dan truk yang berhenti di garis stop.
- Plat motor (sekitar 35 px) belum terbaca.
- Masih ada salah baca 1 karakter. Akurasi belum diukur terhadap ground truth.

## Catatan teknis

- HLS mengirim satu segmen sekaligus. Reader menyimpan frame ke buffer (bukan cuma frame terakhir) supaya tidak kehilangan hampir semua frame di tiap segmen.
- Stream yang putus akan reconnect otomatis.
- Di macOS, onnxruntime memakai CoreML. CoreML melempar error saat crop tanpa plat; log itu disenyapkan karena hasilnya tetap list kosong.
- Dedupe memakai kemiripan string ≥ 0,75, jadi dua mobil berbeda dengan plat sangat mirip dalam 60 detik bisa tergabung.

## Privasi

Plat nomor termasuk data pribadi (UU PDP No. 27/2022). Folder `out/` (database dan snapshot) di-ignore dari git. Jangan publikasikan hasilnya. Hormati juga ketentuan penggunaan portal CCTV yang dipakai.

Lisensi dependency: Ultralytics YOLO memakai AGPL-3.0.
