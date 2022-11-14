from django.urls import path
from . import views as transistor_view
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path('upload_transistor', transistor_view.upload_transistor, name='upload_transistor'),
    path('result_transistor', transistor_view.result_transistor, name='result_transistor'),
    path('detail_transistor', transistor_view.detail_transistor, name='detail_transistor'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)