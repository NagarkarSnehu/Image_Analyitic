from django.urls import path
from . import views as capsule_view
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path('upload_capsule', capsule_view.upload_capsule, name='upload_capsule'),
    path('result_capsule', capsule_view.result_capsule, name='result_capsule'),
    path('detail_capsule', capsule_view.detail_capsule, name='detail_capsule'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)