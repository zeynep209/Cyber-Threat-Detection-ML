#  Cyber Threat Detection & Classification

Makine öğrenmesi yöntemleri kullanılarak siber güvenlik olaylarının analiz edilmesi ve saldırı türlerinin sınıflandırılması amacıyla geliştirilmiş bir veri analizi ve makine öğrenmesi projesidir.

Proje kapsamında farklı kaynaklardan elde edilen siber güvenlik verileri bir araya getirilerek saldırı olayları, finansal etkiler ve pazar üzerindeki etkiler birlikte incelenmiştir. Makine öğrenmesi modellerinin performansları karşılaştırılmış ve saldırıların sınıflandırılmasında etkili olan özellikler analiz edilmiştir.

##  Projenin Amacı

Bu projenin temel amacı, geçmiş siber güvenlik olaylarından elde edilen verileri kullanarak saldırı türlerinin makine öğrenmesi yöntemleriyle analiz edilmesi ve sınıflandırılmasıdır.

Bunun yanında siber olayların finansal ve sektörel etkilerinin incelenmesi ve saldırı sınıflandırmasında etkili olan değişkenlerin belirlenmesi amaçlanmıştır.

##  Proje Kapsamı

- Farklı siber güvenlik veri kaynaklarının birleştirilmesi
- Veri ön işleme ve veri setinin makine öğrenmesine hazırlanması
- Siber saldırıların özelliklerine göre analiz edilmesi
- Makine öğrenmesi modelleri ile saldırı sınıflandırması
- Farklı modellerin performanslarının karşılaştırılması
- Model sonuçlarının değerlendirilmesi
- Feature Importance analizi
- Siber olayların finansal etkilerinin incelenmesi
- Siber olayların pazar üzerindeki etkilerinin incelenmesi

##  Makine Öğrenmesi Süreci

Proje kapsamında farklı kaynaklardan elde edilen veriler ortak bir veri yapısında birleştirilmiştir.

Hazırlanan veri seti üzerinde makine öğrenmesi modelleri uygulanarak siber saldırıların sınıflandırılması hedeflenmiştir. Elde edilen model sonuçları karşılaştırılarak modellerin saldırı türlerini ayırt etme performansları değerlendirilmiştir.

Ayrıca Feature Importance analizi kullanılarak model tahminlerinde hangi değişkenlerin daha etkili olduğu incelenmiştir.

##  Veri ve Analiz Dosyaları

| Dosya | Açıklama |
|---|---|
| `incidents_master.csv` | Birleştirilmiş siber olay veri seti |
| `financial_impact.csv` | Siber olayların finansal etkilerine ilişkin veriler |
| `market_impact.csv` | Siber olayların pazar etkilerine ilişkin veriler |
| `model_results.csv` | Makine öğrenmesi modellerinden elde edilen sonuçlar |
| `feature_importance.csv` | Özellik önem derecelerine ilişkin analiz sonuçları |
| `step1_merge.py` | Veri kaynaklarını birleştirmek için kullanılan Python kodu |

## Kullanılan Teknolojiler

- Python
- Pandas
- NumPy
- Scikit-learn
- Machine Learning
- Data Preprocessing
- Feature Importance
- Veri Analizi

##  Proje Yapısı

```text
Cyber-Threat-Detection-ML/
├── feature_importance.csv
├── financial_impact.csv
├── incidents_master.csv
├── market_impact.csv
├── model_results.csv
├── step1_merge.py
├── .gitignore
└── README.md
```

##  Çıktılar

Proje sonucunda farklı siber güvenlik verileri tek bir veri setinde birleştirilmiş, makine öğrenmesi modellerinin sınıflandırma performansları karşılaştırılmış ve saldırıların belirlenmesinde etkili olan özellikler analiz edilmiştir.

Elde edilen sonuçlar `model_results.csv` ve `feature_importance.csv` dosyalarında saklanmıştır.

##  Geliştirici

**Zeynep Başocak**  
Bilgisayar Mühendisliği