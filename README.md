# IF3230_Tubes_1

1. Saat berjalan program akan menerima setiap input-input yang diperlukan. Program pada thread ke 0 kemudian ditentukan sebagai master, thread master akan membaca & menyimpan matrix masukan. Setelah itu matrix kernel akan di broadast ke thread-thread lainnya sehingga setiap thread memiliki data matrix kernel. Kemudian, untuk setiap matrix target pada array matriks akan dibagi rata untuk dijalankan pada setiap thread, misal jika terdapat 8 buah matriks target & 4 thread maka setiap thread akan menjalankan 2 buah matrix target. Jika tidak dapat dibagi rata (terdapat sisa) maka matriks sisa tersebut dialokasikan terurut mulai dari thread ke 0, misal jika terdapat 7 buah matriks target & 4 thread maka sisa 3 buah matriks akan dialokasikan 1 buah pada thread ke 0, 1 buah padda thread ke 1, dan 1 buah pada thread ke 2. Setelah dibagikan ke setiap thread, setiap threadnya kemudian akan menjalankan fungsi konvolusi matriks. Setelah proses konvolusi matriks selesai, hasil konvolusi matriks kemudian akan dibandingkan saat pengiriman kembali ke thread master & menyimpan nilai minimum & maksimum. Pada akhir proses pengiriman, program kemudian mencari nilai tengah & mean. Setelah selesai, program kemudian menampilkan hasil.

<!-- Jelaskan cara kerja program Anda, terutama pada paralelisasi yang Anda implementasikan berdasarkan skema di atas. -->

2. Berdasarkan waktu eksekusi yang diperoleh, program yang dieksekusi secara paralel lebih ___ dibanding program yang dieksekusi secara sekuensial. Hal tersebut karena program yang dieksekusi secara paralel ___ ,  sedangkan program yang dieksekusi secara sekuensial ___.

<!-- Dari waktu eksekusi terbaik program paralel Anda, bandingkan dengan waktu eksekusi program sekuensial yang diberikan. Analisis mengapa waktu eksekusi program Anda bisa lebih lambat / lebih cepat / sama saja.-->


3. Jelaskan secara singkat apakah ada perbedaan antara hasil program serial dan program paralel Anda. <!-- Tidak terdapat perbedaan hasil dari eksekusi program serial & paralel. Hal tersebut berarti kedua program sama-sama efektif untuk digunakan dalam melakukan proses konvolusi matriks. Perbedaan antara program lebih terlihat pada waktu eksekusi program-->


4. Berikut merupakan hasil variasi jumlah node OpenMPI yang berpartisipasi dan jumlah thread OpenMP yang digunakan

| Node      | Thread      | Hasil Serial | Hasil Paralel |
| --------- | ----------- | ------------ | ------------- |
| 2         | 5           |  2           |               |
| 2         | 16          |              |               |
| 3         | 5           |              |               |
| 3         | 16          |              |               |
| 4         | 5           |              |               |
| 4         | 16          |              |               |


<!-- Variasikan jumlah node OpenMPI yang berpartisipasi dan jumlah thread OpenMP yang digunakan. Gunakan percobaan-percobaan dengan parameter berikut: -->
