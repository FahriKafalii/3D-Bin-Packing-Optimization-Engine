from django.urls import path
from . import views
from .views import home_view  

app_name = 'palet_app'

urlpatterns = [
    path('', home_view, name='home'),  # Ana sayfa için URL 
    path('urun-listesi/', views.urun_listesi, name='urun_listesi'),  # Ürün listesi sayfası
    path('isleniyor/', views.processing, name='processing'),
    path('analiz/', views.analysis, name='analysis'),
    path('palet-detay/<int:palet_id>/', views.palet_detail, name='palet_detail'),
    path('palet-detay/<int:palet_id>/3d-data/', views.palet_3d_data, name='palet_3d_data'),
    # AJAX işlemleri için
    path('yukle-sonuc/', views.upload_result, name='upload_result'),
    path('yerlestirme-baslat/', views.start_placement, name='start_placement'),
    path('optimizasyon-durumu/', views.optimization_status, name='optimization_status'),
]