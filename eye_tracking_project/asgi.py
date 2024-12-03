"""
ASGI config for eye_calibration_project project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/asgi/
"""

# import os

# from django.core.asgi import get_asgi_application

# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'eye_calibration_project.settings')

# application = get_asgi_application()
import os
from django.core.asgi import get_asgi_application
from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
import eye_tracking_project.routing


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'eye_tracking_project.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            eye_tracking_project.routing.websocket_urlpatterns
            # calibration.routing.websocket_urlpatterns +
            # prediction.routing.websocket_urlpatterns
        )
    ),
})