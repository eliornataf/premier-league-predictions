from django.core.management import BaseCommand
from website.football_club_details_fetcher import FootballClubDetailsFetcher


class Command(BaseCommand):
    """
    Fetches football club details and adds them to the database.
    """

    help = "Fetches football club details and adds them to the database."

    def handle(self, *args, **options) -> None:
        """
        Executes the custom management command to fetch football club details and add them to the database.

        :param args: Command arguments.
        :param options: Command options.
        :return: None
        """
        # Create an instance of FootballClubDetailsFetcher
        fetcher: FootballClubDetailsFetcher = FootballClubDetailsFetcher()

        # Inform the user that the process of fetching and adding club details has started
        self.stdout.write("Fetching and adding football club details to the database.")

        # Fetch football club details and add them to the database
        fetcher.add_clubs_to_database()

        # Inform the user that the club details have been added to the database
        self.stdout.write(
            self.style.SUCCESS("Football club details added to the database.")
        )
