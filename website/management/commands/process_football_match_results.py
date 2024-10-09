from typing import List, Any, Dict
from django.core.management.base import BaseCommand
from website.football_match_scraper import FootballMatchScraper
from website.match_outcome_predictor import MatchOutcomePredictor
from website.models import TableRow


class Command(BaseCommand):
    """
    Custom management command to process football match results HTML and update the database.

    This command fetches football match results HTML, processes the data to update the database with
    relevant information, prepares a match outcome predictor, and updates predicted fixtures.
    """

    help = "Process football match results HTML and update the database."

    def handle(self, *args, **options):
        """
        Executes the custom management command to process football match results HTML and update them in the database.

        :param args: Command arguments.
        :param options: Command options.
        :return: None
        """
        scraper: FootballMatchScraper = FootballMatchScraper()
        predictor: MatchOutcomePredictor = MatchOutcomePredictor()

        # Add a boolean variable to track if updating the table is needed
        update_table_needed: bool = False

        # Continuously process and update results until there are no more
        while True:
            # Process the results HTML and get processed games
            processed_games: List[Dict[str, Any]] = scraper.process_results_html()
            self.stdout.write("Processing football match results HTML.")

            # If there are no more games to process, set the flag to True and break the loop
            if not processed_games:
                break

            update_table_needed = True

            # Combine and save processed game dataframes
            predictor.combine_and_save_dataframes(processed_games=processed_games)
            self.stdout.write("Combined and saved processed game dataframes.")

            # Create the match outcome predictor
            predictor.create_match_outcome_predictor()
            self.stdout.write("Created match outcome predictor.")

            # Update predicted fixtures
            predictor.update_predicted_fixtures()
            self.stdout.write("Updated predicted fixtures.")

        # Check if updating the table is needed and run process_table_html if True
        if update_table_needed:
            # Remove all TableRow entries from the database
            TableRow.objects.all().delete()
            self.stdout.write("Removed all TableRow entries from the database.")

            # Call scraper.process_table_html()
            scraper.process_table_html()
            self.stdout.write("Updated table due to processing results.")

        # Display a message indicating the completion of processing
        self.stdout.write(
            self.style.SUCCESS(
                "Football match results processed and updated in the database."
            )
        )
