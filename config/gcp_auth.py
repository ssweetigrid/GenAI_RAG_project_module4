"""
config/gcp_auth.py
──────────────────
Sets up Vertex AI authentication using your service-account JSON key.
Call `init_vertex()` once at the start of every script.
"""

import os
import vertexai
from config.settings import GCP_CREDENTIALS, GCP_PROJECT_ID, GCP_LOCATION


def init_vertex():
    """
    Point Google's auth library at your JSON key file,
    then initialise the Vertex AI SDK.
    """
    # Tell the Google SDK where your key is
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS

    # Initialise Vertex AI with your project & region
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    print(f"✅ Vertex AI ready  |  project={GCP_PROJECT_ID}  location={GCP_LOCATION}")
