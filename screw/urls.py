from django.urls import path
from . import views as screw_view
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path('upload_screw', screw_view.upload_screw, name='upload_screw'),
    path('result_screw', screw_view.result_screw, name='result_screw'),
    path('detail_screw', screw_view.detail_screw, name='detail_screw'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)