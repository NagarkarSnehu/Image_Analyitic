from django.urls import path
from . import views as bottle_views
# from ..capsule import views as capsule_view
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('', bottle_views.login_user, name='login'),
    path('select', bottle_views.select, name='select'),
    path('introduction', bottle_views.introduction, name='introduction'),
    path('upload_file', bottle_views.upload_file, name='upload_file'),
    path('result', bottle_views.result, name='result'),
    path('detail', bottle_views.detail, name='detail'),
    path('prac', bottle_views.prac, name='prac'),
    path('test', bottle_views.test, name='test'),
    # path('upload_transistor', transistor_view.upload_transistor, name='upload_transistor'),
    path('logout', bottle_views.logout_user, name='logout'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)