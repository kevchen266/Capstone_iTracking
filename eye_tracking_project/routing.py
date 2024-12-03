from django.urls import path
from . import test_consumers, consumers

websocket_urlpatterns = [
    path('ws/dispatcher/', consumers.DispatcherConsumer.as_asgi()),
    path('ws/test_dispatcher/', test_consumers.TestDispatcherConsumer.as_asgi()),
]