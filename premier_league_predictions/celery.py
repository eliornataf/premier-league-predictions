from __future__ import absolute_import, unicode_literals
import os
from celery import Celery
from celery.schedules import crontab


# Set the default Django settings module for the 'celery' program.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "premier_league_predictions.settings")

# Create a Celery instance for the 'premier_league_predictions' project
app = Celery("premier_league_predictions")

# Set the local time zone to Jerusalem
app.conf.timezone = "Asia/Jerusalem"

# Load task modules from all registered Django app configs.
app.config_from_object("django.conf:settings", namespace="CELERY")

# Autodiscover tasks in all apps included in INSTALLED_APPS setting
app.autodiscover_tasks()

# Schedule tasks: fetch clubs details in July 2024, process football match fixtures in August 2024, and weekly process
# football match results every Tuesday morning
app.conf.beat_schedule = {
    "process-football-match-results": {
        "task": "website.tasks.process_football_match_results",
        "schedule": crontab(
            hour="9", minute="0", day_of_week="2"
        ),  # Tuesday at 9:00 AM (0-indexed day, 0=Monday)
    },
}
