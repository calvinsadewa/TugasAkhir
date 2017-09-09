README:
0. Copy semua isi folder program ke harddisk/flashdisk/ storage yang dapat di tulis bebas.
1.Install database postgreSQL versi 9.4.6
2.Pastikan port postgreSQL dibuka di port 5432, dengan username 'postgres' dan password 'postgres' (Alternatifnya adalah update konfigurasi postgreSQL di timeeries_db.py)
3.Buat database baru bernama 'tugas_akhir', dan lakukan restore database dengan file data/db_tugas_akhir.backup
4.python, pastikan install versi 3.5, lebih baik jika lewat anaconda karena berbagai library telah langsung disediakan
5.Install library python dari file library.txt
6.jalankan di console "python test-server.py"
7.Tunggu sebentar sampai muncul tulisan "Event loop running forever, press Ctrl+C to interrupt."
8.buka http://localhost:8080/ di browser, akan ada 3 opsi teratas : home, option, series
9. Home untuk simulasi trading
10. option untuk opsi simulasi
11. series untuk manajemen timeseries