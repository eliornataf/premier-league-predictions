from django.core.management.base import BaseCommand
from website.football_match_scraper import FootballMatchScraper
from website.match_outcome_predictor import MatchOutcomePredictor


class Command(BaseCommand):
    """
    Custom management command to process football match fixtures HTML and update the database.
    """

    help = "Process football match fixtures HTML and update the database."

    def handle(self, *args, **options) -> None:
        """
        Executes the custom management command to process football match fixtures HTML and add them to the database.

        :param args: Command arguments.
        :param options: Command options.
        :return: None
        """
        scraper: FootballMatchScraper = FootballMatchScraper()
        predictor: MatchOutcomePredictor = MatchOutcomePredictor()

        # Inform the user that the process of processing football match fixtures HTML has started
        self.stdout.write("Processing football match fixtures HTML.")

        # Process football match fixtures HTML and add them to the database
        scraper.process_fixtures_html()
        self.stdout.write("Processing football match fixtures HTML.")

        # Create the match outcome predictor
        predictor.create_match_outcome_predictor()
        self.stdout.write("Created match outcome predictor.")

        # Update predicted fixtures
        predictor.update_predicted_fixtures()
        self.stdout.write("Updated predicted fixtures.")

        # Inform the user that the football match fixtures have been processed and added to the database
        self.stdout.write(
            self.style.SUCCESS(
                "Football match fixtures processed and added to the database."
            )
        )
