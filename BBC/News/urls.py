from django.urls import path
from News import views 

urlpatterns = [
    path('',views.index,name="index"),
    path('res/',views.res, name='result')
]
