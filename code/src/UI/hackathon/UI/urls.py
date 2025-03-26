from django.urls import path
from .views import index, handle_upload, search_entity

urlpatterns = [
    path("", index, name="home"),
    path("upload/<str:folder>/", handle_upload, name="handle_upload"),
    path("search/", search_entity, name="search_entity"),  # Search API route
]
