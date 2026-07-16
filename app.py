# Aplikasi Flask untuk klasifikasi motif batik menggunakan model VGG16 hasil transfer learning

import json
import os
import time
import uuid
from datetime import datetime

import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for
from PIL import Image
from werkzeug.utils import secure_filename

import config
from batik_info import BATIK_INFO, get_batik_info

app = Flask(__name__)
app.secret_key = "batik-vgg16-secret-key"
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)

# Status model diisi oleh muat_model() saat aplikasi pertama kali dijalankan
model = None
class_names = config.DEFAULT_CLASS_NAMES
model_mode = "belum_dimuat"


def muat_model():
    # Pakai model hasil training jika tersedia, jika belum jatuh ke mode demo dengan VGG16 + ImageNet
    global model, class_names, model_mode
    import tensorflow as tf

    if os.path.exists(config.MODEL_PATH) and os.path.exists(config.CLASS_INDEX_PATH):
        model = tf.keras.models.load_model(config.MODEL_PATH)
        with open(config.CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
            index_to_class = json.load(f)
        class_names = [index_to_class[str(i)] for i in range(len(index_to_class))]
        model_mode = "terlatih"
        print(f"[INFO] Model batik dimuat. Kelas: {class_names}")
    else:
        print("[PERINGATAN] Model batik belum ditemukan, aplikasi berjalan dalam mode demo.")
        from tensorflow.keras.applications import VGG16

        model = VGG16(weights="imagenet", include_top=True)
        class_names = None
        model_mode = "demo"


def berkas_diizinkan(nama_file: str) -> bool:
    # Cek ekstensi file yang diunggah termasuk yang diizinkan
    return "." in nama_file and nama_file.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


def praproses_gambar(path_gambar: str) -> np.ndarray:
    # Ubah file gambar menjadi array numpy 224x224 yang sudah dinormalisasi untuk VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input

    gambar = Image.open(path_gambar).convert("RGB").resize(config.IMG_SIZE)
    array_gambar = np.expand_dims(np.array(gambar, dtype=np.float32), axis=0)
    return preprocess_input(array_gambar)


def prediksi_gambar(path_gambar: str):
    # Jalankan model dan kembalikan top-3 label motif beserta persentase keyakinannya
    array_gambar = praproses_gambar(path_gambar)
    hasil_prediksi = model.predict(array_gambar)[0]
    top3_index = np.argsort(hasil_prediksi)[-3:][::-1]

    hasil = []
    for idx in top3_index:
        if model_mode == "terlatih":
            label = class_names[idx]
        else:
            from tensorflow.keras.applications.vgg16 import decode_predictions

            label = decode_predictions(np.expand_dims(hasil_prediksi, axis=0), top=1000)[0][idx][1]
            label = label.replace("_", " ").title()
        hasil.append({"label": label, "confidence": round(float(hasil_prediksi[idx]) * 100, 2)})
    return hasil


def hitung_statistik_dataset():
    # Hitung jumlah gambar per kelas langsung dari folder dataset/train, val, test.
    # Kalau folder dataset tidak ada di server (memang sengaja tidak ikut di-deploy
    # karena isinya ribuan foto), pakai snapshot statistik dari model/dataset_stats.json
    # yang dibuat sekali lewat generate_dataset_stats.py di komputer lokal.
    if not os.path.isdir(config.TRAIN_DIR):
        return _muat_statistik_dataset_dari_snapshot()

    statistik = []
    nama_kelas_di_disk = sorted(
        d for d in os.listdir(config.TRAIN_DIR) if os.path.isdir(os.path.join(config.TRAIN_DIR, d))
    )

    for nama_kelas in nama_kelas_di_disk:
        jumlah = {}
        for split_name, split_dir in [("train", config.TRAIN_DIR), ("val", config.VAL_DIR), ("test", config.TEST_DIR)]:
            folder_kelas = os.path.join(split_dir, nama_kelas)
            if os.path.isdir(folder_kelas):
                jumlah[split_name] = len(
                    [f for f in os.listdir(folder_kelas) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                )
            else:
                jumlah[split_name] = 0
        statistik.append(
            {
                "nama": nama_kelas,
                "train": jumlah["train"],
                "val": jumlah["val"],
                "test": jumlah["test"],
                "total": jumlah["train"] + jumlah["val"] + jumlah["test"],
            }
        )
    return statistik


def _muat_statistik_dataset_dari_snapshot():
    # Baca statistik dataset dari snapshot JSON (dibuat oleh generate_dataset_stats.py)
    # supaya halaman /dataset tetap menampilkan angka asli walau folder dataset
    # tidak ikut di-deploy ke server.
    snapshot_path = os.path.join(config.MODEL_DIR, "dataset_stats.json")
    if os.path.exists(snapshot_path):
        with open(snapshot_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def muat_metrik_evaluasi():
    # Baca ringkasan hasil training dari model/metrics.json apabila sudah ada
    if os.path.exists(config.METRICS_PATH):
        with open(config.METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@app.route("/")
def dashboard():
    # Halaman beranda: statistik ringkas, grafik dataset, dan galeri motif
    statistik_dataset = hitung_statistik_dataset()
    total_gambar = sum(item["total"] for item in statistik_dataset)
    metrik = muat_metrik_evaluasi()
    return render_template(
        "dashboard.html",
        model_mode=model_mode,
        class_names=(class_names or []),
        batik_info=BATIK_INFO,
        total_kelas=len(class_names) if (model_mode == "terlatih" and class_names) else len(BATIK_INFO),
        total_gambar=total_gambar,
        metrik=metrik,
        statistik_dataset=statistik_dataset,
    )


@app.route("/klasifikasi")
def klasifikasi():
    # Halaman unggah gambar untuk diklasifikasikan
    daftar_nama_tampilan = [nama.replace("_", " ") for nama in (class_names or config.DEFAULT_CLASS_NAMES)]
    return render_template("klasifikasi.html", model_mode=model_mode, class_names_display=daftar_nama_tampilan)


def analisis_performa(metrik):
    # Cari motif dengan recall tertinggi/terendah dan pasangan motif yang paling sering tertukar
    laporan = metrik.get("laporan_per_kelas")
    if not laporan:
        return None

    recall_per_kelas = [(nama, laporan[nama]["recall"]) for nama in metrik["nama_kelas"] if nama in laporan]
    if not recall_per_kelas:
        return None

    kelas_terbaik = max(recall_per_kelas, key=lambda x: x[1])
    kelas_terlemah = min(recall_per_kelas, key=lambda x: x[1])

    pasangan_tertukar = None
    cm = metrik.get("confusion_matrix")
    if cm:
        nama_kelas = metrik["nama_kelas"]
        skor_tertinggi = 0
        for i, baris in enumerate(cm):
            for j, nilai in enumerate(baris):
                if i != j and nilai > skor_tertinggi:
                    skor_tertinggi = nilai
                    pasangan_tertukar = (nama_kelas[i], nama_kelas[j], nilai)

    return {
        "kelas_terbaik": {"nama": kelas_terbaik[0], "recall": round(kelas_terbaik[1] * 100, 1)},
        "kelas_terlemah": {"nama": kelas_terlemah[0], "recall": round(kelas_terlemah[1] * 100, 1)},
        "pasangan_tertukar": pasangan_tertukar,
    }


@app.route("/evaluasi-model")
def evaluasi_model_view():
    # Halaman performa model: metrik, grafik training, dan confusion matrix
    metrik = muat_metrik_evaluasi()
    ada_grafik_riwayat = os.path.exists(config.HISTORY_PLOT_PATH)
    ada_confusion_matrix = os.path.exists(config.CONFUSION_MATRIX_PATH)
    analisis = analisis_performa(metrik) if metrik else None
    return render_template(
        "evaluasi.html",
        model_mode=model_mode,
        metrik=metrik,
        ada_grafik_riwayat=ada_grafik_riwayat,
        ada_confusion_matrix=ada_confusion_matrix,
        analisis=analisis,
        konfigurasi_training={
            "batch_size": config.BATCH_SIZE,
            "epoch_feature_extraction": config.EPOCHS_FEATURE_EXTRACTION,
            "epoch_fine_tuning": config.EPOCHS_FINE_TUNING,
            "ukuran_input": f"{config.IMG_SIZE[0]} x {config.IMG_SIZE[1]} piksel",
        },
    )


@app.route("/dataset")
def dataset_view():
    # Halaman informasi dan statistik dataset yang digunakan
    statistik_dataset = hitung_statistik_dataset()
    return render_template(
        "dataset.html",
        statistik_dataset=statistik_dataset,
        sumber_nama=config.DATASET_SOURCE_NAME,
        sumber_url=config.DATASET_SOURCE_URL,
    )


@app.route("/tentang")
def tentang():
    # Halaman penjelasan metode transfer learning VGG16
    return render_template("tentang.html", class_names=(class_names or []), model_mode=model_mode)


@app.route("/predict", methods=["POST"])
def predict():
    # Terima file yang diunggah, jalankan prediksi, lalu tampilkan hasilnya
    if "file" not in request.files:
        flash("Tidak ada file yang dipilih.", "danger")
        return redirect(url_for("klasifikasi"))

    file = request.files["file"]
    if file.filename == "":
        flash("Silakan pilih file gambar terlebih dahulu.", "danger")
        return redirect(url_for("klasifikasi"))

    if not berkas_diizinkan(file.filename):
        flash("Format file tidak didukung. Gunakan JPG, JPEG, atau PNG.", "danger")
        return redirect(url_for("klasifikasi"))

    # Nama file dibuat unik agar tidak tertimpa file lain
    ekstensi = secure_filename(file.filename).rsplit(".", 1)[1].lower()
    nama_file_unik = f"{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d%H%M%S')}.{ekstensi}"
    path_simpan = os.path.join(app.config["UPLOAD_FOLDER"], nama_file_unik)
    file.save(path_simpan)

    try:
        waktu_mulai = time.perf_counter()
        daftar_hasil = prediksi_gambar(path_simpan)
        waktu_proses_ms = round((time.perf_counter() - waktu_mulai) * 1000)
        dimensi_asli = Image.open(path_simpan).size
    except Exception as e:  # noqa: BLE001
        flash(f"Terjadi kesalahan saat memproses gambar: {e}", "danger")
        return redirect(url_for("klasifikasi"))

    prediksi_utama = daftar_hasil[0]
    info_motif = get_batik_info(prediksi_utama["label"]) if model_mode == "terlatih" else None
    return render_template(
        "result.html",
        gambar_url=url_for("static", filename=f"uploads/{nama_file_unik}"),
        prediksi_utama=prediksi_utama,
        daftar_hasil=daftar_hasil,
        info_motif=info_motif,
        model_mode=model_mode,
        waktu_proses_ms=waktu_proses_ms,
        dimensi_asli=f"{dimensi_asli[0]} x {dimensi_asli[1]} piksel",
        ukuran_file_kb=round(os.path.getsize(path_simpan) / 1024),
    )


@app.errorhandler(404)
def halaman_tidak_ditemukan(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def kesalahan_server(e):
    return render_template("500.html"), 500


# Model dimuat sekali saat modul diimpor, baik lewat "python app.py" maupun gunicorn
muat_model()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
