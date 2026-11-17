# Devir notu — sıradaki adımlar

Branch: `feature/physical-realism` (`main` üzerine kurulu, **push edilmedi**).

## Şu anki durum

- `gr_physics.py`: CUDA gerektirmeyen referans fizik (jeodezik, Novikov–Thorne akısı, g-faktörü, limb darkening, kara cisim → sRGB, gökyüzü dokusu).
- `gr_raytracer.py`: CUDA çekirdekleri, `Renderer`, CLI, etkileşimli görüntüleyici (eski `gpu_disk_viewer_v062.py` yerine).
- Doğrulama (hepsi geçiyor, RTX 5060 Laptop, Python 3.13):
  - `py -3.13 -m unittest test_physics.py test_gpu.py` → 42 test
  - `py -3.13 gr_raytracer.py --validate` → 9 kontrol
  - `py -3.13 gr_raytracer.py --benchmark-frames 60` → 1080p'de ~5,6 ms/kare (eski kod 7,1 ms)

## Çalışma kuralı (bozma)

Yeni fizik önce `gr_physics.py` içinde Python olarak yazılır ve test edilir, GPU'ya **tablo (LUT)** olarak yüklenir, ardından `test_gpu.py` içinde CPU tahmini ile piksel eşlik testi eklenir. Formülü CUDA içinde elle çoğaltma; eski kodun testleri bu yüzden çekirdeği hiç sınamıyordu.

## Doğrulanmamış olanlar

- Etkileşimli pencere gerçek ekranda denenmedi; yalnızca `SDL_VIDEODRIVER=dummy` ile tüm tuşlar çalıştırıldı ve pencere içeriği PNG olarak incelendi.
- Python 3.13 dışındaki sürümler ve başka GPU'lar denenmedi.
- `F12`, ilk karede (henüz render yokken) basılırsa sessizce hiçbir şey yapmaz — zararsız, ama düzeltilebilir.

## Planlanan ama yapılmayan işler (öncelik sırasıyla)

1. **Yüksek mertebeli görüntülerin çözünürlüğü.** Tablolar `mu`'da düzgün örnekleniyor (`build_transfer`, `build_path_table`); kritik çarpma parametresi yakınında (foton halkası, mertebe ≥ 2) örnek aralığı ~0,011 M. `mu` yerine kritik değere doğru sıklaşan bir parametre (ör. `log(b − b_c)`) kullan; `path_u`, `rout_crossing` ve `shade_sky` içindeki indeks hesabını buna göre değiştir. Test: `--fov 15` ile `--order-colors` görüntüsünde 2. mertebe halkanın sürekliliği.
2. **Dışa bakan ışınlar (`mu > 0`).** `trace` içinde `mu` `[-1, 0]` aralığına kırpılıyor; kamera hep deliğe baktığı için şimdilik sorun yok. Serbest bakış (kamerayı delikten çevirme) eklenecekse tablolar `mu ∈ [-1, 1]` olmalı.
3. **Disk yapısı ve zaman.** Disk şu an pürüzsüz ve durağan. Kepler dönüşüyle kayan yoğunluk/sıcaklık dalgalanmaları (prosedürel gürültü, `Ω = r^-3/2` ile kaydırılmış) ve bir zaman parametresi ekle. Işık yolu gecikmesi için tabloya `t(φ)` eklemek gerekir (`dt/dφ = b / (u² (1 − 2u))`).
4. **Parlama (bloom).** Doygun bölgeler için doğrusal ışıkta, sRGB kodlamasından önce ayrık Gauss bulanıklığı; ayrı bir çekirdek geçişi gerekir (`render` şu an tek geçişte doğrudan 8 bit yazıyor → önce float tampona yazacak şekilde ayır).
5. **Gökyüzü üretim süresi.** `make_sky_texture` açılışta ~2 sn sürüyor (yarısı `srgb_encode` ve bulut gürültüsü). `float32`'ye geçir veya `seed`'e göre `.npy` önbelleği ekle (`.gitignore`'a eklemeyi unutma).
6. **Limb darkening rengi.** Şu an yalnızca parlaklığı ölçekliyor; kenara doğru kızarma yok.
7. **Kerr.** Kapsam dışı bırakıldı; `u(φ)` düzlemsel yörünge tablosu Kerr'de geçerli değil, ayrı bir integratör ve tablo düzeni gerekir. Büyük iş, ayrı branch.

## Davranış değişiklikleri (kullanıcı geri isteyebilir)

- Varsayılan renk modu kara cisim, `T_max = 4500 K`; eski görünüme en yakın: `--color-mode classic --background-r 100`.
- Yıldızlar sonsuzda (eskiden 100 M küre); gökyüzü kutbu disk ekseninde (+y).
- Gri/klasik modda parlaklık `g⁴` (eskiden `g^2.6`).
- `-`/`=` pozlama, `,`/`.` sıcaklık (eskiden kalınlık); kalınlık yalnızca `Z`/`X`.
- `--disk-rin < 6` ve `--camera-r <= --disk-rout` artık reddediliyor.
