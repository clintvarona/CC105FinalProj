from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict-form/', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
]
