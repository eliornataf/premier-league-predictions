from celery import shared_task
from django.core.management import call_command


@shared_task
def process_football_match_results() -> None:
    """
    Celery task to process football match results.

    This task invokes the Django management command 'process_football_match_results' to process football match results.

    :return: None
    """
    call_command("process_football_match_results")
